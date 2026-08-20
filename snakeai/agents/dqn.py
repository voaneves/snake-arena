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

    # exploração ε-greedy — ignorada quando `noisy=True`
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
        self._atualizacoes = 0

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
        """Zero quando `noisy=True`: a exploração vira responsabilidade da rede."""
        if self.cfg.noisy:
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

    def _escolher(self, obs, mask):
        """A política de **comportamento** — e é aqui que a exploração precisa existir.

        Com `noisy=True` o ε é zero de propósito ("a exploração é responsabilidade da
        rede"), mas `NoisyDense` só sorteava ruído com `training=True`, que a coleta nunca
        ligava: o Rainbow passava o treino inteiro agindo por argmax determinístico. Ver
        `docs/REVISAO_ALGORITMOS.md` §2.3. A avaliação continua sem ruído — é outro
        caminho (`politica()`), e o contrato exige determinismo lá.
        """
        with ruido_ligado(self.model, ativo=bool(self.cfg.noisy)):
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
        g = cfg.gamma ** cfg.n_steps

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
        tz = lote["rew"][:, None] + g * (1.0 - lote["done"])[:, None] * self.suporte
        tz = np.clip(tz, cfg.v_min, cfg.v_max)
        b = (tz - cfg.v_min) / self.delta_z
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
            else:
                q = tf.gather(saida, act, batch_dims=1)
                erro = alvo - q
                # Huber: quadrática perto de zero, linear longe — um outlier de TD não
                # domina o lote inteiro
                por_amostra = tf.where(tf.abs(erro) <= 1.0,
                                       0.5 * tf.square(erro),
                                       tf.abs(erro) - 0.5)
            perda = tf.reduce_mean(pesos * por_amostra)
        grads = tape.gradient(perda, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return perda, por_amostra

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
            self.memoria.update_priorities(idx, np.asarray(por_amostra))
        return float(perda)

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
            self.memoria.add_batch(obs_ant, acoes, r, prox_obs, prox_done, prox_mask)
            self.global_step += cfg.num_envs
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        self.episodes += len(scores)

        if self.global_step >= cfg.warmup_steps and len(self.memoria) >= cfg.batch_size:
            perdas.append(self._aprender())

        # em atualizações de gradiente, não em passos de ambiente — ver `target_update`
        self._atualizacoes += len(perdas)
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
