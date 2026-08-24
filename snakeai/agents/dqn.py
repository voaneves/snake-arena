"""DQN — a família inteira num agente só.

Os seis notebooks quase idênticos do `colab-rl` viram seis linhas de configuração. Cada
componente do Rainbow é uma flag independente, para que a pergunta "quanto o PER vale?"
possa ser respondida com o resto congelado:

======================  ================================================================
flag                    o que liga
======================  ================================================================
``double``              alvo Double DQN: a rede online escolhe a ação, a alvo avalia
``dueling``             cabeça `V + A − média(A)`
``per``                 memória priorizada com correção por importance sampling
``n_steps``             retornos de n passos
``noisy``               exploração por ruído aprendido (dispensa o ε-greedy)
``n_atoms``             C51 distribucional
======================  ================================================================

Ligar tudo é o Rainbow. Não ligar nada é o DQN de 2013.

A diferença estrutural em relação ao original
---------------------------------------------
O DQN antigo coletava de **um** jogo pygame por vez, o que fazia o ambiente ser o gargalo.
Aqui ele coleta de `num_envs` ambientes em paralelo, igual ao PPO — o que significa que os
dois algoritmos veem o mesmo número de passos de ambiente pelo mesmo custo de tempo, e a
comparação passa a ser sobre o algoritmo, não sobre quem tinha o coletor mais rápido.
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
from ..memory.replay import PrioritizedReplayBuffer, ReplayBuffer
from ..nets import build_q_network, q_de_distribuicao
from ..nets.heads import ruido_ligado
from ..otimizadores import cria_otimizador
from .base import AgentBase, BaseConfig

__all__ = ["DQNConfig", "DQN"]


@dataclass
class DQNConfig(BaseConfig):
    net: str = "resnet_small"
    num_envs: int = 64

    # --- componentes do Rainbow, cada um mensurável isolado
    double: bool = False
    dueling: bool = False
    per: bool = False
    noisy: bool = False
    n_steps: int = 1
    n_atoms: int = 0

    gamma: float = 0.995
    lr: float = 3e-4
    #: O eixo que substitui o K-FAC. Ver `snakeai/otimizadores.py`.
    optimizer: str = "adam"
    batch_size: int = 512
    memory_size: int = 200_000
    #: Iterações do laço de coleta entre atualizações. Cada iteração dá um passo em
    #: **todos** os `num_envs` ambientes, então o intervalo em passos de ambiente é
    #: `learn_every × num_envs` — 256 no padrão, não 4.
    learn_every: int = 4
    warmup_steps: int = 20_000

    #: Defasagem da rede alvo, em **atualizações de gradiente**. Contava passos de
    #: ambiente, e com uma atualização a cada 256 deles os 2.000 nominais viravam ~8 de
    #: defasagem real — 250× menos que as implementações de referência. Com o alvo colado
    #: na rede online, o `double` perde o efeito (a rede que escolhe e a que avalia são
    #: quase a mesma) e o alvo deixa de ser um ponto fixo.
    #:
    #: O valor não é o canônico 2.000: este orçamento tem ~19.500 atualizações no treino
    #: inteiro, e 2.000 deixariam só dez sincronizações. 250 é ~1,3% do orçamento — a
    #: mesma ordem de grandeza relativa das referências. É um eixo de ablação declarado,
    #: não uma constante óbvia. Ver `docs/REVISAO_ALGORITMOS.md` §2.4.
    target_update: int = 250
    max_grad_norm: float = 10.0

    #: Deixa o ε agendado valer **junto** com as noisy nets. Por padrão `False`, então a
    #: composição canônica do Rainbow não muda e as ablações de DQN com `noisy=True`
    #: continuam sem ε, como sempre. Existe porque a exploração é o suspeito aberto do
    #: Rainbow neste ambiente, e antes não havia como pedir um piso sem editar código:
    #: `epsilon()` devolvia zero incondicionalmente e `eps_start` virava um campo
    #: silenciosamente ignorado. Ver `docs/REVISAO_ALGORITMOS.md` §2.15.
    eps_mesmo_com_noisy: bool = False

    #: Um `ε` de noisy net **por ambiente** na coleta, em vez de um por passada.
    #: Ver `NoisyDense.por_amostra` e `docs/REVISAO_ALGORITMOS.md` §2.24.
    ruido_por_ambiente: bool = False

    # exploração ε-greedy — ignorada quando `noisy=True`, salvo `eps_mesmo_com_noisy`
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_frac: float = 0.2         # fração do treino gasta decaindo

    # PER
    per_alpha: float = 0.6
    per_beta0: float = 0.4

    # C51
    v_min: float = -2.0
    v_max: float = 60.0           # o retorno de Snake vai muito além do [-10, 10] do Atari


class DQN(AgentBase):
    algo = "dqn"

    def __init__(self, cfg: DQNConfig = None, variant=None):
        cfg = cfg or DQNConfig()
        super().__init__(cfg, variant=variant or self._nome_variante(cfg))
        keras.utils.set_random_seed(cfg.seed)

        construir = lambda: build_q_network(
            cfg.board_size, cfg.net, dueling=cfg.dueling, noisy=cfg.noisy,
            n_atoms=cfg.n_atoms,
        )
        self.model = construir()
        self.target = construir()
        self.target.set_weights(self.model.get_weights())

        self.optimizer = cria_otimizador(cfg.optimizer, cfg.lr,
                                         clipnorm=cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)

        Memoria = PrioritizedReplayBuffer if cfg.per else ReplayBuffer
        extra = {"alpha": cfg.per_alpha, "beta0": cfg.per_beta0} if cfg.per else {}
        self.memoria = Memoria(
            cfg.memory_size, (cfg.board_size, cfg.board_size, N_CHANNELS),
            n_actions=N_ACTIONS, n_steps=cfg.n_steps, gamma=cfg.gamma,
            num_envs=cfg.num_envs, rng=np.random.default_rng(cfg.seed + 2), **extra,
        )

        self.suporte = None
        if cfg.n_atoms:
            self.suporte = np.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms,
                                       dtype=np.float32)
            self.delta_z = float(self.suporte[1] - self.suporte[0])

        self._desde_alvo = 0
        #: Atualizações de gradiente acumuladas — a moeda em que `target_update` é medido.
        #: O contador do `AgentBase` só existe durante o `train`; este vale desde a
        #: construção, para que `iterate()` sozinho também sincronize certo.
        #:
        #: **O nome não pode ser `_atualizacoes`.** Era, e como o `AgentBase.train` também
        #: faz `self._atualizacoes += stats["atualizacoes"]` com o valor que `iterate()`
        #: devolve, o mesmo atributo era incrementado **duas vezes por iteração**. O
        #: `meta["atualizacoes"]` de toda execução de DQN e Rainbow saiu exatamente
        #: **2,00×** o número real de passos de gradiente — medido: 250 chamadas reais,
        #: 500 gravadas. O PPO não tem contador próprio e grava 1,00×, então a comparação
        #: entre famílias no `ORCAMENTO_DE_GRADIENTE.md` estava enviesada por um fator de
        #: dois **numa família só**. Ver `docs/REVISAO_ALGORITMOS.md` §2.18.
        self._passos_gradiente = 0

    @staticmethod
    def _nome_variante(cfg):
        partes = [p for p, on in (("double", cfg.double), ("dueling", cfg.dueling),
                                  ("per", cfg.per), ("noisy", cfg.noisy),
                                  (f"{cfg.n_steps}steps", cfg.n_steps > 1),
                                  ("c51", cfg.n_atoms > 0)) if on]
        nome = "+".join(partes) if partes else "base"
        # o otimizador só entra no nome quando não é o padrão — senão toda variante
        # ganharia um "+adam" que não informa nada
        if cfg.optimizer != "adam":
            nome += f"+{cfg.optimizer}"
        return nome

    def on_model_reloaded(self):
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        self.optimizer = cria_otimizador(self.cfg.optimizer, self.cfg.lr,
                                         clipnorm=self.cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)

    # ------------------------------------------------------------- exploração
    def epsilon(self):
        """O ε agendado. Zero com `noisy=True`, salvo `eps_mesmo_com_noisy=True`.

        O padrão não mudou: noisy nets substituem o ε, como no paper. O que mudou é que
        agora existe **um jeito de pedir os dois** — antes `epsilon()` devolvia `0.0`
        incondicionalmente e `eps_start` virava um campo silenciosamente ignorado, sem erro
        e sem efeito.

        Isto importa porque a exploração do Rainbow neste ambiente é o suspeito aberto: com
        `noisy=True` e ε zero, a entropia das ações de uma rede não treinada mede 0,949
        contra 1,099 do aleatório, e o `sigma` das `NoisyDense` fica **constante** ao longo
        do treino (0,02446 → 0,02421 em 100 mil passos) enquanto os gaps de Q crescem — ou
        seja, a exploração relativa encolhe sozinha. Ver `docs/REVISAO_ALGORITMOS.md` §2.16.
        """
        if self.cfg.noisy and not getattr(self.cfg, "eps_mesmo_com_noisy", False):
            return 0.0
        f = min(1.0, self.frac() / max(self.cfg.eps_frac, 1e-9))
        return self.cfg.eps_start + f * (self.cfg.eps_end - self.cfg.eps_start)

    def beta(self):
        """β da PER sobe até 1: pouca correção quando o erro de TD ainda é ruído."""
        return self.cfg.per_beta0 + self.frac() * (1.0 - self.cfg.per_beta0)

    # -------------------------------------------------------------------- Q
    def _q_valores(self, modelo, obs, training=False):
        saida = modelo(obs, training=training)
        if self.cfg.n_atoms:
            return q_de_distribuicao(saida, self.suporte)
        return saida

    def politica(self):
        """A política greedy que `snakeai.eval` consome — sem ε, sem ruído."""
        def fn(obs, mask):
            q = np.asarray(self._q_valores(self.model, tf.convert_to_tensor(obs)))
            return np.where(mask, q, MASK_NEG).astype(np.float32)
        return fn

    def politica_do_modelo(self, modelo):
        """A `politica()` acima, mas para um modelo vindo de fora — com o C51 colapsado.

        O `keras_policy` que o `AgentBase` usa assume saída `(lote, ações)` e faz
        `tf.where(mask, logits, ...)`. Com `n_atoms > 0` a rede devolve
        `(lote, ações, átomos)` e o broadcast quebra:

            ValueError: Dimensions must be equal, but are 250 and 3 ...
            input shapes: [250,3], [250,3,121], [250,3,121]

        A `politica()` deste agente já colapsava a distribuição pelo `_q_valores`; esta
        não, e como o único caminho que passa por aqui é `avaliar_melhor()`, o erro
        chegava **depois do orçamento inteiro gasto**. É o mesmo lugar do §2.14, um passo
        adiante: com o `Lambda` corrigido o checkpoint volta do disco, e aí quebra na
        primeira jogada. Ver `docs/REVISAO_ALGORITMOS.md` §2.17.
        """
        def fn(obs, mask):
            q = np.asarray(self._q_valores(modelo, tf.convert_to_tensor(obs)))
            return np.where(mask, q, MASK_NEG).astype(np.float32)
        return fn

    def _escolher(self, obs, mask):
        """A política de **comportamento** — e é aqui que a exploração precisa existir.

        Com `noisy=True` o ε é zero de propósito ("a exploração é responsabilidade da
        rede"), mas `NoisyDense` só sorteava ruído com `training=True`, que a coleta nunca
        ligava: o Rainbow passava o treino inteiro agindo por argmax determinístico. Ver
        `docs/REVISAO_ALGORITMOS.md` §2.3. A avaliação continua sem ruído — é outro
        caminho (`politica()`), e o contrato exige determinismo lá.
        """
        with ruido_ligado(self.model, ativo=bool(self.cfg.noisy),
                          por_amostra=bool(getattr(self.cfg, 'ruido_por_ambiente', False))):
            q = np.asarray(self._q_valores(self.model, tf.convert_to_tensor(obs)))
        q = np.where(mask, q, -np.inf)
        acoes = q.argmax(axis=1).astype(np.int32)

        eps = self.epsilon()
        if eps > 0:
            trocar = self.rng.random(len(acoes)) < eps
            if trocar.any():
                p = mask.astype(np.float64)
                p /= p.sum(axis=1, keepdims=True)
                aleatorias = (p.cumsum(axis=1) >
                              self.rng.random((len(acoes), 1))).argmax(axis=1)
                acoes[trocar] = aleatorias[trocar].astype(np.int32)
        return acoes

    # -------------------------------------------------------------- alvo de TD
    def _alvo(self, lote):
        cfg = self.cfg
        prox = tf.convert_to_tensor(lote["next_obs"])
        mascara = lote["next_mask"]
        # `γ**n_real`, não `γ**n_steps`: as janelas esvaziadas no fim de um episódio são
        # mais curtas, e com a fome sendo truncamento (`done=0`) elas bootstrapam de
        # verdade — descontar por 3 passos o que andou 2 desloca o alvo. Ver §2.13.
        n_real = np.asarray(lote.get("n_real", cfg.n_steps), dtype=np.float32)
        g = cfg.gamma ** n_real

        if cfg.n_atoms:
            return self._alvo_c51(lote, prox, mascara, g)

        q_alvo = np.asarray(self._q_valores(self.target, prox))
        if cfg.double:
            # a rede online escolhe, a alvo avalia — corta o viés otimista do max
            q_online = np.asarray(self._q_valores(self.model, prox))
            melhor = np.where(mascara, q_online, -np.inf).argmax(axis=1)
        else:
            melhor = np.where(mascara, q_alvo, -np.inf).argmax(axis=1)

        v_prox = q_alvo[np.arange(len(melhor)), melhor]
        return lote["rew"] + g * v_prox * (1.0 - lote["done"])

    def _alvo_c51(self, lote, prox, mascara, g):
        """Projeção categórica do C51: desloca o suporte e redistribui a massa."""
        cfg = self.cfg
        logits = np.asarray(self.target(prox, training=False))
        p_prox = np.asarray(tf.nn.softmax(logits, axis=-1))

        q_prox = (p_prox * self.suporte).sum(-1)
        if cfg.double:
            q_online = np.asarray(self._q_valores(self.model, prox))
            melhor = np.where(mascara, q_online, -np.inf).argmax(axis=1)
        else:
            melhor = np.where(mascara, q_prox, -np.inf).argmax(axis=1)

        n = len(melhor)
        p_sel = p_prox[np.arange(n), melhor]                       # (n, n_atoms)
        # `g` virou por amostra (γ**n_real), então precisa de eixo para transmitir contra
        # o suporte — ver §2.13
        g = np.reshape(g, (-1, 1)) if np.ndim(g) else g
        tz = lote["rew"][:, None] + g * (1.0 - lote["done"])[:, None] * self.suporte
        tz = np.clip(tz, cfg.v_min, cfg.v_max)
        b = (tz - cfg.v_min) / self.delta_z
        # `tz` já está preso a [v_min, v_max], mas `delta_z` é float32 e a divisão pode
        # devolver 50,0000001 para o átomo do topo — aí `ceil` dá 51 e o `np.add.at`
        # abaixo estoura o eixo. Prender `b` ao índice válido é o que a implementação
        # canônica do C51 faz, e sem isto o agente só não quebra por sorte da aritmética
        # do suporte que estiver configurado.
        b = np.clip(b, 0.0, cfg.n_atoms - 1)
        l, u = np.floor(b).astype(np.int64), np.ceil(b).astype(np.int64)

        alvo = np.zeros_like(p_sel)
        # quando b cai exatamente num átomo, l == u e a massa iria para o vazio
        eq = l == u
        np.add.at(alvo, (np.arange(n)[:, None], l), p_sel * (u - b + eq))
        np.add.at(alvo, (np.arange(n)[:, None], u), p_sel * (b - l))
        return alvo.astype(np.float32)

    # ------------------------------------------------------------------ update
    @tf.function(reduce_retracing=True)
    def _passo_treino(self, obs, act, alvo, pesos, distribucional):
        with tf.GradientTape() as tape:
            saida = self.model(obs, training=True)
            if distribucional:
                logits = tf.gather(saida, act, batch_dims=1)      # (n, n_atoms)
                logp = tf.nn.log_softmax(logits, axis=-1)
                por_amostra = -tf.reduce_sum(alvo * logp, axis=-1)
                surpresa = por_amostra          # o H(alvo) sai fora do grafo
            else:
                q = tf.gather(saida, act, batch_dims=1)
                erro = alvo - q
                # Huber: quadrática perto de zero, linear longe — um outlier de TD não
                # domina o lote inteiro
                por_amostra = tf.where(tf.abs(erro) <= 1.0,
                                       0.5 * tf.square(erro),
                                       tf.abs(erro) - 0.5)
                # a PER quer |δ|, não a perda: ver `_prioridades`
                surpresa = tf.abs(erro)
            perda = tf.reduce_mean(pesos * por_amostra)
        grads = tape.gradient(perda, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return perda, surpresa

    def _aprender(self):
        cfg = self.cfg
        if cfg.per:
            lote, idx, pesos = self.memoria.sample(cfg.batch_size, beta=self.beta())
        else:
            lote, idx, pesos = self.memoria.sample(cfg.batch_size)

        alvo = self._alvo(lote)
        perda, por_amostra = self._passo_treino(
            tf.convert_to_tensor(lote["obs"]),
            tf.convert_to_tensor(lote["act"].astype(np.int32)),
            tf.convert_to_tensor(np.asarray(alvo, dtype=np.float32)),
            tf.convert_to_tensor(pesos),
            bool(cfg.n_atoms),
        )
        if cfg.per:
            self.memoria.update_priorities(idx, self._prioridades(por_amostra, alvo))
        return float(perda)

    def _prioridades(self, por_amostra, alvo):
        """A surpresa de cada transição — que **não** é a perda que o gradiente usa.

        No ramo C51 a perda é a entropia cruzada, e `CE = KL(alvo‖pred) + H(alvo)`. O
        `H(alvo)` não mede erro nenhum: mede quão difusa a rede alvo está no estado
        sucessor, e com 121 átomos ele fica preso perto de `ln 121 = 4,796`. Medido num
        lote real de 512:

            CE (usada antes)  média 4,7922  desvio 0,0178
            KL (a correta)    média 0,0363  desvio 0,2382
            correlação CE × KL = **−0,9066**

        Não era só ruído: era **anticorrelação**. A amostra de maior erro do lote recebia
        prioridade 2,4903 e a de menor erro 2,5581 — a mais surpreendente era amostrada
        *menos*. A massa dos 10% maiores dava 0,100, exatamente uniforme; com KL dá 0,212.
        Um dos seis componentes do Rainbow não fazia nada, e o pouco que fazia era ao
        contrário.

        No ramo escalar a perda é o Huber. Como `(δ²/2)**α ∝ |δ|**2α`, usá-la como
        prioridade dobra o expoente efetivo da PER na região quadrática — `α=0,6` vira
        1,2, e a ablação "quanto a PER vale" mede um `α` que não é o do `config`. A
        prioridade certa é `|δ|`, e o `_passo_treino` já devolve o erro absoluto para isso.

        Ver `docs/REVISAO_ALGORITMOS.md` §2.19.
        """
        p = np.asarray(por_amostra, dtype=np.float64)
        if self.cfg.n_atoms:
            a = np.asarray(alvo, dtype=np.float64)
            entropia = -(a * np.log(np.clip(a, 1e-12, None))).sum(-1)
            p = p - entropia                      # CE − H = KL
        return np.maximum(p, 0.0)

    # ------------------------------------------------------------------ passo
    def iterate(self):
        cfg = self.cfg
        scores, perdas, vitorias = [], [], 0
        passos_por_iter = max(1, cfg.learn_every)

        for _ in range(passos_por_iter):
            obs_ant, mask_ant = self.obs, self.mask
            acoes = self._escolher(obs_ant, mask_ant)
            self.obs, self.mask, r, d, info = self.env.step(acoes)
            self.registra_fim(info)

            # fome é truncamento: o estado final verdadeiro entra no lugar da
            # observação já resetada, e `done` volta a 0 para o bootstrap acontecer
            prox_obs, prox_mask, prox_done = self.desfaz_truncamento(
                info, self.obs, self.mask, d.astype(np.float32))
            # `prox_done` é 0 na fome (truncamento, o alvo bootstrapa); `d` é a fronteira
            # real do episódio, e é ela que corta a janela de n passos. Ver §2.13.
            self.memoria.add_batch(obs_ant, acoes, r, prox_obs, prox_done, prox_mask,
                                   fim=d.astype(np.float32))
            self.global_step += cfg.num_envs
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        self.episodes += len(scores)

        if self.global_step >= cfg.warmup_steps and len(self.memoria) >= cfg.batch_size:
            perdas.append(self._aprender())

        # em atualizações de gradiente, não em passos de ambiente — ver `target_update`
        self._passos_gradiente += len(perdas)
        self._desde_alvo += len(perdas)
        if self._desde_alvo >= cfg.target_update:
            self.target.set_weights(self.model.get_weights())
            self._desde_alvo = 0

        return {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "atualizacoes": len(perdas),
            "loss": float(np.mean(perdas)) if perdas else None,
            "epsilon": self.epsilon(),
            "memoria": len(self.memoria),
        }
