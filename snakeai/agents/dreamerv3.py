"""DreamerV3 — aprende um modelo do mundo e treina o ator **dentro** dele.

O que o separa dos outros oito
------------------------------
AlphaZero e MuZero também têm modelo, mas o usam para **buscar no momento de jogar**: a
qualidade da ação vem de gastar computação em tempo de inferência. O Dreamer não busca
nada quando age — a rede olha o estado latente e escolhe. O modelo serve para **treinar**:
o ator aprende em rollouts imaginados, e o ambiente real só entrega dados e a medição.

Isso tem uma consequência que importa para a arena: o número do Dreamer na curva é o número
da política pura, sem asterisco. Já AlphaZero e MuZero aparecem na curva pela rede pura e a
versão com busca vai numa coluna à parte, porque busca é computação extra de inferência —
a mesma regra que mantém o filtro de flood-fill fora da curva.

O laço, em cinco passos
-----------------------
1. **Coleta.** Anda no ambiente com o ator, mantendo `(h, z)` por ambiente entre chamadas,
   e grava a sequência.
2. **Modelo.** Amostra `B` sequências de `T` passos, desenrola o RSSM, e treina
   reconstrução + recompensa + continuação + máscara + KL balanceada.
3. **Sonho.** Dos `B×T` estados posteriores, imagina `H` passos com o **prior** — sem
   observação nenhuma, que é o teste real de se o modelo aprendeu.
4. **Ator e crítico.** Retornos λ dentro do sonho; o ator por REINFORCE com retorno
   normalizado por percentis; o crítico em two-hot com alvo do próprio retorno.
5. Repete.

Três detalhes que decidem entre funcionar e não funcionar
---------------------------------------------------------
**KL balanceada.** A KL entre prior e posterior é otimizada com pesos diferentes nas duas
direções: 0,5 para aproximar o prior do posterior (o modelo aprende a prever) e 0,1 para
aproximar o posterior do prior (o encoder não guarda detalhe irrelevante). Um peso só nas
duas direções faz o posterior colapsar no prior e o modelo virar constante.

**Free bits.** A KL só entra na perda acima de 1 nat. Sem isso o termo de KL domina no
começo, quando reconstruir ainda é difícil, e o latente colapsa antes de aprender nada.

**Normalização do retorno por percentis.** A escala do retorno em Snake muda de ~1 para ~50
ao longo do treino. O ator divide a vantagem por `percentil(95) − percentil(5)` do retorno,
com média móvel — não pela variância, que é sensível a cauda. É o que faz um único
`ent_coef` servir do começo ao fim.

Custo
-----
É o mais caro dos nove por passo de ambiente: cada passo de treino desenrola o RSSM `T`
vezes e imagina `H` vezes. O `iterate()` reporta `train_ratio` — passos de gradiente por
passo de ambiente — porque essa é a variável que decide se o Dreamer está sendo eficiente
em dados ou só devagar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf

from ..env.vec_snake import N_ACTIONS, N_CHANNELS, VecSnake
from ..eval import MASK_NEG
from ..memory.sequencia import SequenceBuffer
from ..nets.dreamer import (PRESETS_DREAMER, CelulaRecorrente, amostra_straight_through,
                            bins_simetricos, build_ator, build_cabecas, build_critico,
                            build_decoder, build_encoder, build_rssm_post,
                            build_rssm_prior, de_two_hot, symexp, symlog, two_hot,
                            unimix)
from ..otimizadores import cria_otimizador
from .base import AgentBase, BaseConfig

__all__ = ["DreamerV3Config", "DreamerV3", "PoliticaRecorrente"]


def _percentil(x, q):
    """Percentil por ordenação. Sem dependência extra, e exato para o tamanho daqui.

    Percentis, e não desvio padrão: a distribuição de retorno em Snake tem cauda longa
    (um episódio que come 40 vezes convive com dezenas que morrem em 5 passos), e o desvio
    padrão dessa distribuição é dominado pela cauda.
    """
    plano = tf.sort(tf.reshape(x, [-1]))
    n = tf.cast(tf.shape(plano)[0], tf.float32)
    i = tf.cast(tf.clip_by_value(q / 100.0 * (n - 1.0), 0.0, n - 1.0), tf.int32)
    return plano[i]


class PoliticaRecorrente:
    """A política do Dreamer no formato que `snakeai.eval` consome, **com memória**.

    O `keras_policy` padrão assume uma rede sem estado: recebe `obs`, devolve logits. O
    Dreamer age sobre `(h, z)`, que dependem de toda a história do episódio. Avaliá-lo sem
    a recorrência não daria erro — daria um número **mais baixo**, e a conclusão "modelo do
    mundo não funciona aqui" viria de um defeito da medição, não do algoritmo.

    O contrato entre esta classe e `evaluate` tem duas metades:

    * `__call__(obs, mask)` avança o latente e devolve logits;
    * `apos_passo(acoes, done)` recebe o que de fato aconteceu — a ação escolhida (que
      pode ter passado pelo filtro de segurança e não ser o argmax) e onde o episódio
      terminou, para zerar o latente ali.

    Sem a segunda metade, o latente atravessaria a morte da cobra e carregaria o estado de
    uma partida para dentro da próxima.
    """

    def __init__(self, agente):
        self.ag = agente
        self.n = None
        #: O mesmo grafo da coleta. A avaliação são 1.000 episódios de centenas de passos
        #: cada; em eager, o custo é dominado pelo despacho e o protocolo oficial demora
        #: mais que um pedaço do treino.
        self._grafo = tf.function(agente._passo_de_politica, reduce_retracing=True)

    def _garante(self, n):
        if self.n != n:
            self.n = n
            self.h = tf.zeros([n, self.ag.deter])
            self.z = tf.zeros([n, self.ag.dim_z])
            self.a = np.zeros(n, np.int32)
            self.first = np.ones(n, bool)

    def __call__(self, obs, mask):
        self._garante(obs.shape[0])
        _, h, z = self._grafo(
            self.h, self.z, tf.one_hot(self.a, N_ACTIONS),
            tf.convert_to_tensor(self.first),
            tf.convert_to_tensor(obs, tf.float32), tf.convert_to_tensor(mask))
        self.h, self.z, self.first = h, z, np.zeros(self.n, bool)
        logits = self.ag.ator(tf.concat([h, z], axis=-1), training=False)
        return tf.where(tf.convert_to_tensor(mask), logits,
                        tf.fill(tf.shape(logits), MASK_NEG)).numpy()

    def apos_passo(self, acoes, done):
        self.a = np.asarray(acoes, np.int32)
        self.first = np.asarray(done, bool)


@dataclass
class DreamerV3Config(BaseConfig):
    preset: str = "dreamer_small"
    num_envs: int = 64

    #: Sequências por lote e comprimento de cada uma.
    batch_size: int = 16
    seq_len: int = 32

    #: Capacidade da memória, por ambiente.
    memory_size: int = 8_000
    warmup_steps: int = 20_000

    #: **Transições reaproveitadas por passo de ambiente.** É o parâmetro que a literatura
    #: do Dreamer chama de *train ratio*, e é o que decide se o agente aprende ou só anda.
    #:
    #: Ele é derivado, não independente: `train_steps` sai daqui, de `batch_size`, de
    #: `seq_len` e de quantos passos a coleta dá. Expor a razão em vez do número de passos
    #: é o que faz o valor continuar significando a mesma coisa quando `num_envs` muda —
    #: com `train_steps` fixo, dobrar os ambientes **metade** o aprendizado, em silêncio.
    #:
    #: O DreamerV3 do paper usa entre 32 e 1024, mas para orçamentos de ~100 mil passos.
    #: Aqui o contrato dá 5 milhões, e razão 4 já são 20 milhões de transições revisitadas.
    train_ratio: float = 4.0

    #: Deixe `None` para derivar de `train_ratio`. Fixar aqui é para ablação.
    train_steps: int = None
    collect_steps: int = 16

    #: Sonho.
    horizonte: int = 15
    gamma: float = 0.997
    lam: float = 0.95

    #: Modelo do mundo.
    kl_free: float = 1.0
    kl_dyn: float = 0.5
    kl_rep: float = 0.1
    unimix: float = 0.01
    n_bins: int = 41

    #: Ator e crítico.
    ent_coef: float = 3e-4
    critico_ema: float = 0.98
    critico_reg: float = 1.0
    ret_ema: float = 0.99

    lr_modelo: float = 1e-4
    lr_ator: float = 3e-5
    lr_critico: float = 3e-5
    optimizer: str = "adamw"
    max_grad_norm: float = 100.0

    net: str = "dreamer_small"

    def __post_init__(self):
        super().__post_init__()
        if self.preset not in PRESETS_DREAMER:
            raise ValueError(
                f"preset {self.preset!r} desconhecido. Use um de "
                f"{sorted(PRESETS_DREAMER)}")
        if self.seq_len < 4:
            raise ValueError("seq_len curto demais para uma recorrência aprender algo")
        if self.train_steps is None:
            por_iter = self.collect_steps * self.num_envs
            self.train_steps = max(
                1, round(self.train_ratio * por_iter / (self.batch_size * self.seq_len)))
        self.net = self.preset


class DreamerV3(AgentBase):
    algo = "dreamerv3"

    def __init__(self, cfg: DreamerV3Config = None, model=None, variant=None):
        cfg = cfg or DreamerV3Config()
        super().__init__(cfg, variant=variant or cfg.preset)
        keras.utils.set_random_seed(cfg.seed)

        d = PRESETS_DREAMER[cfg.preset]
        self.deter, self.grupos, self.classes = d["deter"], d["grupos"], d["classes"]
        self.dim_z = self.grupos * self.classes
        self.dim_estado = self.deter + self.dim_z
        self.bins = bins_simetricos(cfg.n_bins)

        self.encoder = build_encoder(cfg.board_size, d["canais"])
        self.dim_emb = self.encoder.output_shape[-1]
        self.gru = CelulaRecorrente(self.deter, name="gru")
        self.entrada_gru = keras.layers.Dense(d["largura"], use_bias=False,
                                              name="gru_in")
        self.prior = build_rssm_prior(self.deter, self.grupos, self.classes, d["largura"])
        self.post = build_rssm_post(self.deter, self.dim_emb, self.grupos, self.classes,
                                    d["largura"])
        self.decoder = build_decoder(self.dim_estado, cfg.board_size, d["canais"])
        self.cabecas = build_cabecas(self.dim_estado, d["largura"], cfg.n_bins)
        self.ator = build_ator(self.dim_estado, d["largura"])
        self.critico = build_critico(self.dim_estado, d["largura"], cfg.n_bins)
        self.critico_alvo = build_critico(self.dim_estado, d["largura"], cfg.n_bins,
                                         nome="critico_alvo")

        #: `self.model` é o **ator** porque é ele que `snakeai.eval` consome e é ele que o
        #: contrato mede. O resto do Dreamer é maquinaria de treino: um modelo do mundo
        #: excelente com um ator ruim vale zero na arena, e o registro tem que refletir isso.
        self.model = self.ator

        self.opt_modelo = cria_otimizador(cfg.optimizer, cfg.lr_modelo,
                                          clipnorm=cfg.max_grad_norm)
        self.opt_ator = cria_otimizador(cfg.optimizer, cfg.lr_ator,
                                        clipnorm=cfg.max_grad_norm)
        self.opt_critico = cria_otimizador(cfg.optimizer, cfg.lr_critico,
                                          clipnorm=cfg.max_grad_norm)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.memoria = SequenceBuffer(cfg.num_envs, cfg.memory_size,
                                      (cfg.board_size, cfg.board_size, N_CHANNELS),
                                      N_ACTIONS, seed=cfg.seed)
        self._h = tf.zeros([cfg.num_envs, self.deter])
        self._z = tf.zeros([cfg.num_envs, self.dim_z])
        self._primeiro = np.ones(cfg.num_envs, dtype=bool)
        self._ultima_acao = np.zeros(cfg.num_envs, dtype=np.int32)
        #: `tf.Variable`, e não `float`, porque o passo de treino roda em **grafo**: um
        #: atributo Python só seria atualizado na traçagem, uma vez, e depois congelaria.
        self._escala_ret = tf.Variable(1.0, trainable=False, name="escala_retorno")
        self._grad_steps = 0
        self._construido = False
        self._grafo = tf.function(self._passo_de_gradiente, reduce_retracing=True)
        self._grafo_politica = tf.function(self._passo_de_politica, reduce_retracing=True)

    # -------------------------------------------------------------- variáveis
    def _vars_modelo(self):
        partes = [self.encoder, self.entrada_gru, self.gru, self.prior, self.post,
                  self.decoder, self.cabecas]
        return [v for m in partes for v in m.trainable_variables]

    # ------------------------------------------------------------------- RSSM
    def _avanca(self, h, z, acao_onehot, primeiro=None):
        """Um passo determinístico do RSSM: `(h, z, a) → h'`.

        `primeiro` zera `h` e `z` onde um episódio novo começou. Sem esse reset, o modelo
        carrega o estado latente de uma cobra que já morreu para dentro da próxima
        partida — e aprende uma dinâmica que não existe.
        """
        if primeiro is not None:
            manter = 1.0 - tf.cast(primeiro, tf.float32)[:, None]
            h, z = h * manter, z * manter
            acao_onehot = acao_onehot * manter
        x = self.entrada_gru(tf.concat([z, acao_onehot], axis=-1))
        return self.gru(x, h)

    def _posterior(self, h, obs, seed=None):
        emb = self.encoder(obs)
        lg = unimix(self.post([h, emb]), self.grupos, self.classes, self.cfg.unimix)
        z, lg2d = amostra_straight_through(lg, self.grupos, self.classes, seed=seed)
        return z, lg2d

    def _prior(self, h, seed=None):
        lg = unimix(self.prior(h), self.grupos, self.classes, self.cfg.unimix)
        z, lg2d = amostra_straight_through(lg, self.grupos, self.classes, seed=seed)
        return z, lg2d

    @staticmethod
    def _kl(logits_q, logits_p):
        """KL(q‖p) somada sobre os grupos, em nats. Entradas em `(..., grupos, classes)`."""
        q = tf.nn.softmax(logits_q, axis=-1)
        lq = tf.nn.log_softmax(logits_q, axis=-1)
        lp = tf.nn.log_softmax(logits_p, axis=-1)
        return tf.reduce_sum(q * (lq - lp), axis=[-1, -2])

    # ------------------------------------------------------------------ coleta
    def _passo_de_politica(self, h, z, a_ant, primeiro, obs, mask):
        """`(estado, observação) → (ação, estado novo)`. **Pura**, para caber num grafo.

        Um passo de coleta chama encoder, posterior, GRU e ator — meia dúzia de submodelos
        sobre um lote pequeno. Em eager, cada um é um despacho separado, e medido aqui isso
        é **99% do custo da coleta**: o `VecSnake` gasta 7,7 ms enquanto as chamadas de
        modelo gastam 603 ms. Não é cálculo, é overhead de chamada.

        Numa GPU isso é pior, porque cada despacho espera o Python: o treino já está em
        grafo e ficou rápido, então a coleta vira o gargalo e a placa fica ociosa
        — exatamente o que o painel de uso do Kaggle mostrava com GPU em 4%.

        A função é pura de propósito: `(h, z)` entram e saem como tensores em vez de virar
        atributo. Efeito colateral em Python dentro de `tf.function` acontece só na
        traçagem, e o estado latente congelaria no da primeira iteração.
        """
        h = self._avanca(h, z, a_ant, primeiro)
        z, _ = self._posterior(h, obs)
        logits = self.ator(tf.concat([h, z], axis=-1))
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        return tf.random.categorical(logits, 1)[:, 0], h, z

    def _escolher(self, obs, mask):
        """Ação amostrada da política, sobre o estado latente atual. Atualiza `(h, z)`."""
        acoes, h, z = self._grafo_politica(
            self._h, self._z, tf.one_hot(self._ultima_acao, N_ACTIONS),
            tf.convert_to_tensor(self._primeiro),
            tf.convert_to_tensor(obs, tf.float32), tf.convert_to_tensor(mask))
        self._h, self._z = h, z
        self._ultima_acao = acoes.numpy().astype(np.int32)
        return self._ultima_acao

    def collect(self):
        cfg = self.cfg
        scores, vitorias = [], 0
        for _ in range(cfg.collect_steps):
            obs_ant, mask_ant, primeiro = self.obs, self.mask, self._primeiro.copy()
            acoes = self._escolher(obs_ant, mask_ant)
            self.obs, self.mask, r, d, info = self.env.step(acoes)

            self.memoria.add(obs_ant, acoes, r, 1.0 - d.astype(np.float32),
                             primeiro, mask_ant)
            self._primeiro = d.copy()
            self.global_step += cfg.num_envs
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]
        self.episodes += len(scores)
        return scores, vitorias

    # --------------------------------------------------------- modelo do mundo
    def _desenrola(self, obs, act, first):
        """Desenrola o RSSM sobre uma sequência `(B, T, ...)`. Devolve estados e logits.

        O laço é em Python sobre `T` porque `T` é fixo e conhecido na configuração — o
        `tf.while_loop` só ganharia em compilação, e aqui a legibilidade da recorrência
        vale mais que os milissegundos.
        """
        B, T = tf.shape(obs)[0], obs.shape[1]
        h = tf.zeros([B, self.deter])
        z = tf.zeros([B, self.dim_z])
        a_onehot = tf.one_hot(act, N_ACTIONS)

        hs, zs, post_lg, prior_lg = [], [], [], []
        for t in range(T):
            a_ant = a_onehot[:, t - 1] if t > 0 else tf.zeros([B, N_ACTIONS])
            h = self._avanca(h, z, a_ant, first[:, t])
            lg_p = unimix(self.prior(h), self.grupos, self.classes, self.cfg.unimix)
            z, lg_q = self._posterior(h, obs[:, t])
            hs.append(h)
            zs.append(z)
            post_lg.append(lg_q)
            prior_lg.append(tf.reshape(lg_p, tf.shape(lg_q)))
        empilha = lambda xs: tf.stack(xs, axis=1)
        return empilha(hs), empilha(zs), empilha(post_lg), empilha(prior_lg)

    def _perda_modelo(self, lote):
        cfg = self.cfg
        obs = tf.convert_to_tensor(lote["obs"])
        act = tf.convert_to_tensor(lote["act"])
        first = tf.convert_to_tensor(lote["first"])

        h, z, lg_q, lg_p = self._desenrola(obs, act, first)
        estado = tf.concat([h, z], axis=-1)
        plano = tf.reshape(estado, [-1, self.dim_estado])

        recon = self.decoder(plano)
        alvo_obs = tf.reshape(obs, tf.shape(recon))
        # A observação é contínua e limitada; erro quadrático em symlog é o que o
        # DreamerV3 usa e evita ter que escolher uma verossimilhança por ambiente.
        p_recon = tf.reduce_mean(
            tf.reduce_sum(tf.square(recon - symlog(alvo_obs)), axis=[1, 2, 3]))

        r_lg, c_lg, m_lg = self.cabecas(plano)
        alvo_r = two_hot(symlog(tf.reshape(lote["rew"], [-1])), self.bins)
        p_rec = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(alvo_r, r_lg))
        p_cont = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.reshape(lote["cont"], [-1, 1]), c_lg))
        p_mask = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.cast(tf.reshape(lote["mask"], [-1, N_ACTIONS]), tf.float32), m_lg))

        # KL balanceada, com free bits. Cada direção tem seu próprio `stop_gradient`:
        # é isso que faz "aprender a prever" e "não guardar lixo" serem dois termos.
        kl_dyn = tf.maximum(cfg.kl_free, tf.reduce_mean(
            self._kl(tf.stop_gradient(lg_q), lg_p)))
        kl_rep = tf.maximum(cfg.kl_free, tf.reduce_mean(
            self._kl(lg_q, tf.stop_gradient(lg_p))))

        perda = (p_recon + p_rec + p_cont + p_mask
                 + cfg.kl_dyn * kl_dyn + cfg.kl_rep * kl_rep)
        partes = {"recon": p_recon, "rec": p_rec, "cont": p_cont, "mask": p_mask,
                  "kl_dyn": kl_dyn, "kl_rep": kl_rep}
        return perda, partes, tf.stop_gradient(plano)

    # ------------------------------------------------------------------- sonho
    def _sonha(self, estado0):
        """Imagina `H` passos a partir de `estado0`, usando **só** o prior.

        Nenhuma observação entra aqui. Se o modelo do mundo estiver ruim, é exatamente
        neste ponto que o treino do ator degenera — e o jeito de ver isso é comparar a
        recompensa imaginada com a real, que é o que `stats["rew_sonho"]` reporta.
        """
        cfg = self.cfg
        h, z = tf.split(estado0, [self.deter, self.dim_z], axis=-1)
        estados, logps, entropias = [], [], []

        for _ in range(cfg.horizonte):
            estado = tf.concat([h, z], axis=-1)
            _, _, m_lg = self.cabecas(estado)
            mask = m_lg > 0.0
            # um estado sem nenhuma ação viável é um artefato do modelo, não do jogo:
            # nesse caso libera todas, senão o softmax vira NaN
            mask = tf.where(tf.reduce_any(mask, axis=-1, keepdims=True), mask,
                            tf.ones_like(mask))

            logits = self.ator(estado)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp_all = tf.nn.log_softmax(logits)
            a = tf.random.categorical(logits, 1)[:, 0]
            logps.append(tf.gather(logp_all, a[:, None], batch_dims=1)[:, 0])
            p = tf.exp(logp_all)
            entropias.append(-tf.reduce_sum(
                p * tf.where(mask, logp_all, tf.zeros_like(logp_all)), axis=-1))
            estados.append(estado)

            h = self._avanca(h, z, tf.one_hot(a, N_ACTIONS))
            z, _ = self._prior(h)

        estados.append(tf.concat([h, z], axis=-1))
        return (tf.stack(estados, axis=0), tf.stack(logps, axis=0),
                tf.stack(entropias, axis=0))

    def _retornos_lambda(self, rew, cont, valores):
        """`R_t = r_t + γc_t[(1-λ)V_{t+1} + λR_{t+1}]`, com `R_H = V_H`."""
        cfg = self.cfg
        R = valores[-1]
        saida = []
        for t in reversed(range(rew.shape[0])):
            R = rew[t] + cfg.gamma * cont[t] * (
                (1 - cfg.lam) * valores[t + 1] + cfg.lam * R)
            saida.append(R)
        return tf.stack(saida[::-1], axis=0)

    def _perda_ator_critico(self, estado0):
        cfg = self.cfg
        estados, logps, entropias = self._sonha(estado0)
        plano = tf.reshape(estados, [-1, self.dim_estado])

        r_lg, c_lg, _ = self.cabecas(plano)
        rew = tf.reshape(symexp(de_two_hot(r_lg, self.bins)), tf.shape(estados)[:2])
        cont = tf.reshape(tf.sigmoid(c_lg[:, 0]), tf.shape(estados)[:2])

        v_alvo = tf.reshape(symexp(de_two_hot(self.critico_alvo(plano), self.bins)),
                            tf.shape(estados)[:2])
        R = self._retornos_lambda(rew[:-1], cont[:-1], v_alvo)

        # --- ator: REINFORCE com vantagem normalizada por percentis
        bruta = tf.maximum(_percentil(R, 95.0) - _percentil(R, 5.0), 1.0)
        self._escala_ret.assign(cfg.ret_ema * self._escala_ret
                                + (1 - cfg.ret_ema) * bruta)
        escala = tf.maximum(self._escala_ret, 1.0)
        vantagem = tf.stop_gradient((R - v_alvo[:-1]) / escala)
        p_ator = -tf.reduce_mean(logps * vantagem) - cfg.ent_coef * tf.reduce_mean(
            entropias)

        # --- crítico: two-hot no retorno, mais regularização para o alvo EMA
        estados_v = tf.reshape(estados[:-1], [-1, self.dim_estado])
        lg_v = self.critico(estados_v)
        alvo = two_hot(symlog(tf.stop_gradient(tf.reshape(R, [-1]))), self.bins)
        p_critico = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(alvo, lg_v))
        alvo_ema = tf.nn.softmax(tf.stop_gradient(self.critico_alvo(estados_v)))
        p_critico += cfg.critico_reg * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(alvo_ema, lg_v))

        info = {"rew_sonho": tf.reduce_mean(rew), "retorno": tf.reduce_mean(R),
                "escala_ret": escala, "ent_sonho": tf.reduce_mean(entropias)}
        return p_ator, p_critico, info

    # ---------------------------------------------------------------- um passo
    def _passo_de_gradiente(self, obs, act, rew, cont, first, mask):
        """O passo inteiro em **um grafo**: desenrolar, sonhar, e aplicar os três gradientes.

        Isto é `@tf.function` por um motivo medido, não por hábito. O desenrolamento do RSSM
        é um laço Python de `seq_len` passos e o sonho é outro de `horizonte`, cada iteração
        chamando vários submodelos — em modo eager, isso vira **milhares de kernels
        minúsculos e sequenciais**. Numa GPU o custo disso é latência de lançamento, não
        cálculo: a placa fica ociosa esperando o Python. Medido nesta CPU, a perda do modelo
        caiu de 1.910 ms para 95 ms — 20×. Numa GPU a diferença é maior, porque é lá que a
        latência por kernel pesa mais.

        Consequência de projeto: nada aqui dentro pode ter efeito colateral em Python. Por
        isso `self._escala_ret` é `tf.Variable` — um `float` seria atualizado só na traçagem
        e congelaria no valor da primeira iteração, silenciosamente.
        """
        lote = {"obs": obs, "act": act, "rew": rew, "cont": cont,
                "first": first, "mask": mask}

        with tf.GradientTape() as tape:
            perda_m, partes, estado0 = self._perda_modelo(lote)
        vars_m = self._vars_modelo()
        self.opt_modelo.apply_gradients(zip(tape.gradient(perda_m, vars_m), vars_m))

        with tf.GradientTape(persistent=True) as tape:
            p_ator, p_critico, info = self._perda_ator_critico(estado0)
        self.opt_ator.apply_gradients(zip(
            tape.gradient(p_ator, self.ator.trainable_variables),
            self.ator.trainable_variables))
        self.opt_critico.apply_gradients(zip(
            tape.gradient(p_critico, self.critico.trainable_variables),
            self.critico.trainable_variables))
        del tape

        d = self.cfg.critico_ema
        for a, b in zip(self.critico_alvo.weights, self.critico.weights):
            a.assign(d * a + (1 - d) * b)

        return {"modelo": perda_m, "ator": p_ator, "critico": p_critico,
                **partes, **info}

    def _treina(self):
        cfg = self.cfg
        lote = self.memoria.sample(cfg.batch_size, cfg.seq_len)
        saida = self._grafo(
            tf.convert_to_tensor(lote["obs"], tf.float32),
            tf.convert_to_tensor(lote["act"], tf.int32),
            tf.convert_to_tensor(lote["rew"], tf.float32),
            tf.convert_to_tensor(lote["cont"], tf.float32),
            tf.convert_to_tensor(lote["first"], tf.bool),
            tf.convert_to_tensor(lote["mask"], tf.bool),
        )
        self._grad_steps += 1
        return {k: float(v) for k, v in saida.items()}

    def iterate(self):
        cfg = self.cfg
        scores, vitorias = self.collect()
        perdas = {}
        if self.global_step >= cfg.warmup_steps and self.memoria.pronto(cfg.seq_len):
            for _ in range(cfg.train_steps):
                perdas = self._treina()

        return {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "memoria": len(self.memoria),
            "train_ratio": self._grad_steps * cfg.batch_size * cfg.seq_len
                           / max(1, self.global_step),
            "train_steps": cfg.train_steps,
            **perdas,
        }

    # ------------------------------------------------------------- avaliação
    def politica(self):
        return PoliticaRecorrente(self)

    def on_model_reloaded(self):
        self.ator = self.model
