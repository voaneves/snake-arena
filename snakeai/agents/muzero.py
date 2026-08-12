"""MuZero — a mesma busca, sobre um modelo aprendido.

O contraste com o AlphaZero deste repositório é a razão de ele estar aqui: **o algoritmo de
busca é literalmente o mesmo objeto**, o `MCTS`. Muda só o que a árvore percorre — a
`DinamicaReal` (o `VecSnake`) vira `DinamicaAprendida` (a rede `g`). Toda diferença de
resultado entre os dois é atribuível a isso, e a nada mais.

Vale dizer com todas as letras: **em Snake, o MuZero deveria perder para o AlphaZero.** O
simulador está disponível, é exato e é rápido; trocá-lo por uma aproximação aprendida só
pode piorar a busca. O MuZero existe para domínios onde o simulador *não* está disponível
durante o jogo — e medi-lo aqui é medir quanto custa não ter o simulador. Esse é um número
interessante, e é uma pergunta que o benchmark pode responder justamente por ter os dois
lado a lado, sob o mesmo contrato.

O desenrolar de K passos
------------------------
O treino não olha transições isoladas. Ele parte de uma posição da trajetória, aplica `h`
uma vez, e depois `g` `K` vezes seguindo as ações que foram realmente tomadas. Em cada um
dos `K+1` passos há três perdas:

* **política** ← visitas do MCTS naquele passo,
* **valor** ← retorno de n passos com bootstrap no valor da busca,
* **recompensa** ← a recompensa que o ambiente de fato deu.

A perda de recompensa é a única âncora que liga o estado oculto ao mundo. Sem ela o modelo
pode inventar qualquer dinâmica internamente consistente e a busca vira ficção.
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
from ..nets.muzero import build_dinamica, build_predicao, build_representacao
from ..nets.resnet import PRESETS
from ..search import MCTS, DinamicaAprendida
from .base import AgentBase, BaseConfig

__all__ = ["MuZeroConfig", "MuZero"]


@dataclass
class MuZeroConfig(BaseConfig):
    net: str = "resnet_small"
    num_envs: int = 64
    rollout: int = 16

    num_simulations: int = 24
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25

    gamma: float = 0.997
    n_step: int = 10
    #: Passos do desenrolar no treino. É o que obriga o modelo a ser útil por mais de um
    #: passo à frente — com K=1 ele vira um crítico caro.
    unroll: int = 5

    lr: float = 3e-4
    max_grad_norm: float = 5.0
    batch_size: int = 256
    memory_size: int = 50_000
    epochs_por_iter: int = 1

    temp_inicio: float = 1.0
    temp_fim: float = 0.25
    temp_frac: float = 0.5

    coef_valor: float = 0.25
    coef_recompensa: float = 1.0

    sims_avaliacao: int = 24


class MuZero(AgentBase):
    algo = "muzero"

    def __init__(self, cfg: MuZeroConfig = None, variant=None):
        cfg = cfg or MuZeroConfig()
        super().__init__(cfg, variant=variant or f"unroll{cfg.unroll}")
        keras.utils.set_random_seed(cfg.seed)

        self.h = build_representacao(cfg.board_size, cfg.net)
        self.g = build_dinamica(cfg.board_size, cfg.net)
        self.f = build_predicao(cfg.board_size, cfg.net)
        self.largura = PRESETS[cfg.net][0]

        self.optimizer = keras.optimizers.Adam(cfg.lr, clipnorm=cfg.max_grad_norm)
        self.optimizer.build(self._variaveis())

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)

        self.mcts = MCTS(self._avaliar_oculto, board_size=cfg.board_size,
                         gamma=cfg.gamma, num_simulations=cfg.num_simulations,
                         c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                         dirichlet_frac=cfg.dirichlet_frac,
                         dinamica=DinamicaAprendida(self._passo_dinamica),
                         rng=np.random.default_rng(cfg.seed + 2))

        forma = (cfg.board_size, cfg.board_size, N_CHANNELS)
        M, K = cfg.memory_size, cfg.unroll
        self._buf_obs = np.zeros((M, *forma), dtype=np.float32)
        self._buf_mask = np.ones((M, N_ACTIONS), dtype=bool)
        self._buf_act = np.zeros((M, K), dtype=np.int32)
        self._buf_pi = np.zeros((M, K + 1, N_ACTIONS), dtype=np.float32)
        self._buf_z = np.zeros((M, K + 1), dtype=np.float32)
        self._buf_r = np.zeros((M, K), dtype=np.float32)
        self._pos, self._cheio = 0, 0

    def _variaveis(self):
        return (self.h.trainable_variables + self.g.trainable_variables
                + self.f.trainable_variables)

    def on_model_reloaded(self):
        self.optimizer = keras.optimizers.Adam(self.cfg.lr,
                                               clipnorm=self.cfg.max_grad_norm)
        self.optimizer.build(self._variaveis())

    # -------------------------------------------------------- modelo -> busca
    @property
    def model(self):
        """O `AgentBase` salva `self.model`; para o MuZero o que interessa é `h`+`f`."""
        return self._modelo_politica()

    @model.setter
    def model(self, _):
        pass          # o estado real vive em h, g, f

    def _modelo_politica(self):
        """Modelo `observação → [logits, valor]`, para a política pura e o checkpoint."""
        if getattr(self, "_mp", None) is None and hasattr(self, "h"):
            inp = keras.Input(shape=(self.cfg.board_size, self.cfg.board_size,
                                     N_CHANNELS))
            logits, valor = self.f(self.h(inp))
            self._mp = keras.Model(inp, [logits, valor], name="muzero_politica")
        return getattr(self, "_mp", None)

    @tf.function(reduce_retracing=True)
    def _repr_predicao(self, obs, mask):
        s = self.h(obs, training=False)
        logits, valor = self.f(s, training=False)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        return s, tf.nn.softmax(logits), tf.squeeze(valor, -1)

    @tf.function(reduce_retracing=True)
    def _predicao(self, s):
        logits, valor = self.f(s, training=False)
        return tf.nn.softmax(logits), tf.squeeze(valor, -1)

    @tf.function(reduce_retracing=True)
    def _dinamica_tf(self, s, planos):
        novo, r = self.g([s, planos], training=False)
        return novo, tf.squeeze(r, -1)

    def _planos_de_acao(self, acoes, n):
        b = self.cfg.board_size
        planos = np.zeros((n, b, b, N_ACTIONS), dtype=np.float32)
        planos[np.arange(n), :, :, acoes] = 1.0
        return planos

    def _passo_dinamica(self, estados, acoes):
        """A interface que `DinamicaAprendida` consome."""
        planos = self._planos_de_acao(acoes, len(acoes))
        novo, r = self._dinamica_tf(tf.convert_to_tensor(estados),
                                    tf.convert_to_tensor(planos))
        return novo.numpy(), r.numpy()

    def _avaliar_oculto(self, estados, mask):
        """Priors e valores a partir de estados ocultos — o que o MCTS chama."""
        p, v = self._predicao(tf.convert_to_tensor(np.asarray(estados,
                                                              dtype=np.float32)))
        p = p.numpy()
        m = np.asarray(mask)
        if m.shape == p.shape and not m.all():
            p = np.where(m, p, 0.0)
            p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        return p, v.numpy()

    def politica(self):
        """Política pura de `h`+`f`, sem busca — a curva oficial do contrato."""
        modelo = self._modelo_politica()

        def fn(obs, mask):
            logits, _ = modelo(obs, training=False)
            return np.where(mask, np.asarray(logits), MASK_NEG).astype(np.float32)
        return fn

    def _busca(self, obs, mask, ruido=False):
        """Roda o MCTS a partir da observação: `h` uma vez, depois só a dinâmica."""
        s, priors, valores = self._repr_predicao(tf.convert_to_tensor(obs),
                                                 tf.convert_to_tensor(mask))
        return self.mcts.run(s.numpy(), mask, s.numpy(), adicionar_ruido=ruido)

    # -------------------------------------------------------------------- coleta
    def temperatura(self):
        f = min(1.0, self.frac() / max(self.cfg.temp_frac, 1e-9))
        return self.cfg.temp_inicio + f * (self.cfg.temp_fim - self.cfg.temp_inicio)

    def collect(self):
        cfg = self.cfg
        T, N, K = cfg.rollout, cfg.num_envs, cfg.unroll
        temp = self.temperatura()

        obs_b = np.empty((T, N, cfg.board_size, cfg.board_size, N_CHANNELS), np.float32)
        mask_b = np.empty((T, N, N_ACTIONS), bool)
        pi_b = np.empty((T, N, N_ACTIONS), np.float32)
        v_b = np.empty((T, N), np.float32)
        act_b = np.empty((T, N), np.int32)
        rew_b = np.empty((T, N), np.float32)
        done_b = np.empty((T, N), np.float32)

        scores, vitorias = [], 0
        for t in range(T):
            obs_b[t], mask_b[t] = self.obs, self.mask
            visitas, valores = self._busca(self.obs, self.mask, ruido=True)
            pi = MCTS.politica_das_visitas(visitas, temp)
            pi_b[t], v_b[t] = pi, valores
            a = (pi.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)
            act_b[t] = a
            self.obs, self.mask, r, d, info = self.env.step(a)
            rew_b[t], done_b[t] = r, d.astype(np.float32)
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        # alvo de valor por passo: n passos + bootstrap no valor da busca
        z = np.zeros((T, N), np.float32)
        for t in range(T):
            g = np.zeros(N, np.float32)
            desc = np.ones(N, np.float32)
            vivo = np.ones(N, bool)
            k = 0
            for k in range(min(cfg.n_step, T - t)):
                g += desc * rew_b[t + k] * vivo
                vivo &= done_b[t + k] < 0.5
                desc *= cfg.gamma
            if t + k + 1 < T:
                g += desc * v_b[t + k + 1] * vivo
            z[t] = g

        # cada amostra guarda o desenrolar de K passos que vem depois dela
        validos = max(0, T - K)
        if validos:
            idx = np.arange(validos)
            self._guardar(
                obs_b[idx].reshape(-1, *obs_b.shape[2:]),
                mask_b[idx].reshape(-1, N_ACTIONS),
                np.stack([act_b[idx + k] for k in range(K)], axis=-1).reshape(-1, K),
                np.stack([pi_b[idx + k] for k in range(K + 1)], axis=1)
                  .transpose(0, 2, 1, 3).reshape(-1, K + 1, N_ACTIONS),
                np.stack([z[idx + k] for k in range(K + 1)], axis=-1).reshape(-1, K + 1),
                np.stack([rew_b[idx + k] for k in range(K)], axis=-1).reshape(-1, K),
            )

        self.global_step += T * N
        self.episodes += len(scores)
        return {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "temperatura": temp,
            "valor_busca": float(v_b.mean()),
            "memoria": self._cheio,
        }

    def _guardar(self, obs, mask, act, pi, z, r):
        k = len(obs)
        idx = (self._pos + np.arange(k)) % self.cfg.memory_size
        self._buf_obs[idx] = obs
        self._buf_mask[idx] = mask
        self._buf_act[idx] = act
        self._buf_pi[idx] = pi
        self._buf_z[idx] = z
        self._buf_r[idx] = r
        self._pos = int((self._pos + k) % self.cfg.memory_size)
        self._cheio = min(self._cheio + k, self.cfg.memory_size)

    # -------------------------------------------------------------------- treino
    @tf.function(reduce_retracing=True)
    def _passo(self, obs, mask, act, pi_alvo, z, r_alvo, coef_v, coef_r):
        K = tf.shape(act)[1]
        with tf.GradientTape() as tape:
            s = self.h(obs, training=True)
            logits, valor = self.f(s, training=True)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp = tf.nn.log_softmax(logits)

            perda_pi = -tf.reduce_mean(tf.reduce_sum(pi_alvo[:, 0] * logp, -1))
            perda_v = tf.reduce_mean(tf.square(tf.squeeze(valor, -1) - z[:, 0]))
            perda_r = tf.constant(0.0)

            for k in range(self.cfg.unroll):
                planos = tf.one_hot(act[:, k], N_ACTIONS)[:, None, None, :]
                planos = tf.tile(planos, [1, self.cfg.board_size,
                                          self.cfg.board_size, 1])
                s, rec = self.g([s, planos], training=True)
                # escala de gradiente de 1/2: sem ela o gradiente que chega em `h` cresce
                # com o número de passos do desenrolar
                s = s * 0.5 + tf.stop_gradient(s) * 0.5

                logits_k, valor_k = self.f(s, training=True)
                logp_k = tf.nn.log_softmax(logits_k)
                perda_pi += -tf.reduce_mean(
                    tf.reduce_sum(pi_alvo[:, k + 1] * logp_k, -1))
                perda_v += tf.reduce_mean(
                    tf.square(tf.squeeze(valor_k, -1) - z[:, k + 1]))
                # a âncora do modelo no mundo real: sem ela a dinâmica pode inventar
                # qualquer física internamente consistente
                perda_r += tf.reduce_mean(
                    tf.square(tf.squeeze(rec, -1) - r_alvo[:, k]))

            perda = perda_pi + coef_v * perda_v + coef_r * perda_r

        variaveis = (self.h.trainable_variables + self.g.trainable_variables
                     + self.f.trainable_variables)
        grads = tape.gradient(perda, variaveis)
        self.optimizer.apply_gradients(zip(grads, variaveis))
        return perda_pi, perda_v, perda_r

    def _aprender(self):
        cfg = self.cfg
        if self._cheio < cfg.batch_size:
            return None
        saidas = []
        for _ in range(cfg.epochs_por_iter):
            i = self.rng.integers(0, self._cheio, size=cfg.batch_size)
            p, v, r = self._passo(
                tf.convert_to_tensor(self._buf_obs[i]),
                tf.convert_to_tensor(self._buf_mask[i]),
                tf.convert_to_tensor(self._buf_act[i]),
                tf.convert_to_tensor(self._buf_pi[i]),
                tf.convert_to_tensor(self._buf_z[i]),
                tf.convert_to_tensor(self._buf_r[i]),
                cfg.coef_valor, cfg.coef_recompensa,
            )
            saidas.append((float(p), float(v), float(r)))
        p, v, r = (float(np.mean(x)) for x in zip(*saidas))
        return {"perda_pi": p, "perda_v": v, "perda_r": r}

    def iterate(self):
        stats = self.collect()
        treino = self._aprender()
        if treino:
            stats.update(treino)
        return stats

    # ---------------------------------------------------------------- checkpoint
    def salvar(self, tag="last"):
        for nome, rede in (("h", self.h), ("g", self.g), ("f", self.f)):
            rede.save(os.path.join(self.cfg.ckpt_dir,
                                   f"{self.algo}_{tag}_{nome}.keras"))
        super().salvar(tag)

    def retomar(self, tag="last"):
        caminhos = {n: os.path.join(self.cfg.ckpt_dir, f"{self.algo}_{tag}_{n}.keras")
                    for n in ("h", "g", "f")}
        if not all(os.path.exists(c) for c in caminhos.values()):
            return False
        self.h = keras.models.load_model(caminhos["h"])
        self.g = keras.models.load_model(caminhos["g"])
        self.f = keras.models.load_model(caminhos["f"])
        self._mp = None
        return super().retomar(tag)
