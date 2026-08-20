"""ACER — *Actor-Critic with Experience Replay*.

O algoritmo que o repositório antigo tentou três vezes e nunca fez rodar. Reescrito do
zero em Keras 3, com as quatro peças que o definem:

1. **Retrace(λ)** para estimar o retorno a partir de dados velhos, com a recursão para trás
   no tempo e os pesos de importância truncados em 1.
2. **Gradiente de política com IS truncado + correção de viés.** Truncar a razão `π/μ`
   controla a variância mas introduz viés; o segundo termo devolve a parte cortada,
   somando sobre todas as ações. É o que torna o ACER não-enviesado *e* estável.
3. **Região de confiança contra a política média.** Uma cópia Polyak-média da política
   serve de âncora: se o passo proposto afastaria demais a política dela, ele é projetado
   de volta. Sem isso o ACER diverge com dados off-policy.
4. **Replay ratio.** Um update on-policy seguido de `k` updates sobre trajetórias
   guardadas — é daí que vem a eficiência amostral que justifica o algoritmo.

Os dois bugs legados, e por que não podem voltar
-------------------------------------------------
O ACER do `colab-rl` morria de duas formas distintas:

* ``TypeError: You are passing KerasTensor(...) to a TF API that does not allow
  registering custom dispatchers`` — a lógica do ACER estava sendo montada **dentro do
  grafo funcional do Keras**, onde os tensores são simbólicos. Aqui toda a matemática
  acontece em `tf.function` sobre tensores concretos; o modelo é só `entrada -> [logits,
  Q]`, e nada mais.
* ``ValueError: expected shape=(None, 256, 100), found shape=(None, 100)`` — a dimensão
  de tempo tinha se perdido. Aqui o rollout é `(T, N, ...)` explícito do começo ao fim, e
  o `TrajectoryBuffer` recusa qualquer coisa com outra forma.

Critério de desistência
-----------------------
ACER é o algoritmo mais difícil deste repositório e o que tem mais chance de não convergir
neste domínio. Se depois de um esforço delimitado ele não superar o piso aleatório de forma
consistente, a curva entra no benchmark assim mesmo, com a nota — um resultado negativo
medido vale mais que uma pasta chamada "Not Working".
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
from ..memory.trajectory import TrajectoryBuffer
from ..nets import build_policy_q
from .base import AgentBase, BaseConfig

__all__ = ["ACERConfig", "ACER", "retrace"]


@dataclass
class ACERConfig(BaseConfig):
    num_envs: int = 64
    rollout: int = 32

    gamma: float = 0.995
    lr: float = 7e-4
    max_grad_norm: float = 10.0

    #: Truncamento do peso de importância no Retrace. O paper usa 1.
    c_retrace: float = 1.0
    #: Truncamento no gradiente de política. Acima disso entra a correção de viés.
    c_trunc: float = 10.0

    ent_coef: float = 0.01
    q_coef: float = 0.5

    #: Região de confiança: raio `delta` e taxa da média de Polyak da política âncora.
    trust_region: bool = True
    delta: float = 1.0
    polyak: float = 0.99

    #: Updates off-policy por update on-policy. É a razão de existir do ACER.
    replay_ratio: int = 4
    memory_size: int = 500
    warmup_segments: int = 8


def retrace(rew, done, q_a, v, rho_barra, ultimo_v, gamma):
    """Alvo Retrace(λ), recursão para trás no tempo.

    `Q^ret_t = r_t + γ (1−d_t) [ ρ̄_{t+1} (Q^ret_{t+1} − Q(s_{t+1}, a_{t+1})) + V(s_{t+1}) ]`

    Cortar o peso de importância em 1 (`ρ̄ = min(1, ρ)`) é o que torna o estimador seguro
    para dados arbitrariamente velhos: a correção nunca amplifica, só encolhe. É por isso
    que o ACER pode reusar trajetórias que um A2C teria de jogar fora.

    Formas: tudo `(T, N)`; `ultimo_v` é `(N,)`.
    """
    T, N = rew.shape
    ret = np.zeros((T, N), dtype=np.float32)
    prox_ret = ultimo_v.astype(np.float32)
    prox_q = ultimo_v.astype(np.float32)
    prox_v = ultimo_v.astype(np.float32)
    prox_rho = np.ones(N, dtype=np.float32)

    for t in reversed(range(T)):
        continua = 1.0 - done[t]
        ret[t] = rew[t] + gamma * continua * (
            prox_rho * (prox_ret - prox_q) + prox_v
        )
        prox_ret = ret[t]
        prox_q = q_a[t]
        prox_v = v[t]
        prox_rho = rho_barra[t]
    return ret


class ACER(AgentBase):
    algo = "acer"

    def __init__(self, cfg: ACERConfig = None, variant=None):
        cfg = cfg or ACERConfig()
        super().__init__(cfg, variant=variant or cfg.net)
        keras.utils.set_random_seed(cfg.seed)

        self.model = build_policy_q(cfg.board_size, cfg.net)
        self.media = build_policy_q(cfg.board_size, cfg.net)     # política âncora
        self.media.set_weights(self.model.get_weights())

        self.optimizer = keras.optimizers.Adam(cfg.lr, clipnorm=cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)
        self.memoria = TrajectoryBuffer(cfg.memory_size,
                                        rng=np.random.default_rng(cfg.seed + 2))

    def on_model_reloaded(self):
        self.media = keras.models.clone_model(self.model)
        self.media.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(self.cfg.lr,
                                               clipnorm=self.cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)

    # ------------------------------------------------------------------ política
    def politica(self):
        """Greedy sobre os logits — a mesma régua dos outros agentes."""
        @tf.function(reduce_retracing=True)
        def frente(obs, mask):
            logits, _ = self.model(obs, training=False)
            return tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))

        def fn(obs, mask):
            return frente(tf.convert_to_tensor(obs), tf.convert_to_tensor(mask)).numpy()
        return fn

    @tf.function(reduce_retracing=True)
    def _probs(self, obs, mask):
        logits, q = self.model(obs, training=False)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        return tf.nn.softmax(logits), q

    # ------------------------------------------------------------------ rollout
    def collect(self):
        cfg = self.cfg
        T, N, b = cfg.rollout, cfg.num_envs, cfg.board_size

        obs_buf = np.empty((T, N, b, b, N_CHANNELS), dtype=np.float32)
        mask_buf = np.empty((T, N, N_ACTIONS), dtype=bool)
        act_buf = np.empty((T, N), dtype=np.int32)
        mu_buf = np.empty((T, N, N_ACTIONS), dtype=np.float32)
        rew_buf = np.empty((T, N), dtype=np.float32)
        done_buf = np.empty((T, N), dtype=np.float32)

        scores, vitorias = [], 0
        for t in range(T):
            obs_buf[t], mask_buf[t] = self.obs, self.mask
            pi, _ = self._probs(tf.convert_to_tensor(self.obs),
                                tf.convert_to_tensor(self.mask))
            pi = pi.numpy()
            mu_buf[t] = pi
            a = (pi.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)
            act_buf[t] = a

            self.obs, self.mask, r, d, info = self.env.step(a)
            self.registra_fim(info)
            if info["trunc_idx"].size:       # fome é truncamento, não terminação
                pi_f, q_f = self._probs(tf.convert_to_tensor(info["final_obs"]),
                                        tf.convert_to_tensor(info["final_mask"]))
                # V(s) = Σ_a π(a|s)·Q(s,a) — o ACER não tem cabeça de valor separada
                v_f = np.sum(pi_f.numpy() * q_f.numpy(), axis=1)
                r = self.bootstrap_truncados(info, r, v_f, cfg.gamma)
            rew_buf[t], done_buf[t] = r, d.astype(np.float32)
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        self.global_step += T * N
        self.episodes += len(scores)
        # o estado logo após o último passo — é ele que faz o bootstrap do Retrace deste
        # segmento, hoje e daqui a mil iterações
        segmento = self.memoria.add(obs_buf, mask_buf, act_buf, mu_buf, rew_buf, done_buf,
                                    obs_final=self.obs.copy(), mask_final=self.mask.copy())
        stats = {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "segmentos": len(self.memoria),
        }
        return segmento, stats

    # ------------------------------------------------------------------- update
    @tf.function(reduce_retracing=True)
    def _passo(self, obs, mask, act, mu, ret, ent_coef, q_coef, c_trunc, delta,
               trust_region):
        """Um passo de gradiente do ACER, em tensores concretos.

        Tudo aqui é `tf.function` sobre tensores reais — nunca `KerasTensor` dentro do
        grafo funcional, que era o `TypeError` que matava o ACER legado.
        """
        with tf.GradientTape() as tape_ext:
            logits, q = self.model(obs, training=True)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            pi = tf.nn.softmax(logits)
            logpi = tf.nn.log_softmax(logits)

            v = tf.reduce_sum(pi * q, axis=-1)                     # V = Σ π Q
            q_a = tf.gather(q, act, batch_dims=1)
            pi_a = tf.gather(pi, act, batch_dims=1)
            mu_a = tf.maximum(tf.gather(mu, act, batch_dims=1), 1e-8)
            rho = pi_a / mu_a
            rho_todas = pi / tf.maximum(mu, 1e-8)

            vantagem_ret = tf.stop_gradient(ret - v)
            vantagem_q = tf.stop_gradient(q - tf.expand_dims(v, -1))

            with tf.GradientTape() as tape_logits:
                tape_logits.watch(logits)
                lp = tf.nn.log_softmax(logits)
                lp_a = tf.gather(lp, act, batch_dims=1)

                # termo 1: IS truncado na ação tomada
                termo1 = tf.minimum(c_trunc, tf.stop_gradient(rho)) * lp_a * vantagem_ret
                # termo 2: correção de viés — devolve a parte que o truncamento cortou,
                # somando sobre TODAS as ações, ponderada pela política atual
                corte = tf.nn.relu(1.0 - c_trunc / tf.maximum(
                    tf.stop_gradient(rho_todas), 1e-8))
                termo2 = tf.reduce_sum(
                    corte * tf.stop_gradient(pi) * lp * vantagem_q, axis=-1)
                objetivo = tf.reduce_mean(termo1 + termo2)

            g = tape_logits.gradient(objetivo, logits)

            if trust_region:
                logits_media, _ = self.media(obs, training=False)
                logits_media = tf.where(mask, logits_media,
                                        tf.fill(tf.shape(logits_media), MASK_NEG))
                pi_media = tf.nn.softmax(logits_media)
                # k = ∇_logits KL(π_média || π) = π − π_média
                k = pi - pi_media
                kg = tf.reduce_sum(k * g, axis=-1, keepdims=True)
                kk = tf.reduce_sum(k * k, axis=-1, keepdims=True) + 1e-8
                escala = tf.nn.relu((kg - delta) / kk)
                z = g - escala * k
            else:
                z = g

            # devolve o gradiente projetado para a rede: d(perda)/d(logits) = −z
            perda_pi = -tf.reduce_sum(tf.stop_gradient(z) * logits)
            perda_q = q_coef * tf.reduce_mean(tf.square(tf.stop_gradient(ret) - q_a))
            entropia = -tf.reduce_mean(tf.reduce_sum(pi * logpi, axis=-1))
            perda = perda_pi + perda_q - ent_coef * entropia

        grads = tape_ext.gradient(perda, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return perda_q, entropia, tf.reduce_mean(rho)

    def _alvo_retrace(self, seg):
        """Recalcula π e Q com a rede **atual** sobre a trajetória guardada."""
        cfg = self.cfg
        T, N = seg["act"].shape
        obs = seg["obs"].reshape(T * N, *seg["obs"].shape[2:])
        mask = seg["mask"].reshape(T * N, N_ACTIONS)

        pi, q = self._probs(tf.convert_to_tensor(obs), tf.convert_to_tensor(mask))
        pi = pi.numpy().reshape(T, N, N_ACTIONS)
        q = q.numpy().reshape(T, N, N_ACTIONS)

        v = (pi * q).sum(-1)
        idx_t, idx_n = np.indices((T, N))
        q_a = q[idx_t, idx_n, seg["act"]]
        pi_a = pi[idx_t, idx_n, seg["act"]]
        mu_a = np.maximum(seg["mu"][idx_t, idx_n, seg["act"]], 1e-8)
        rho_barra = np.minimum(cfg.c_retrace, pi_a / mu_a).astype(np.float32)

        # bootstrap com o estado final DO SEGMENTO, nunca com o estado atual do ambiente:
        # num update off-policy os dois não têm relação nenhuma
        pi_f, q_f = self._probs(tf.convert_to_tensor(seg["obs_final"]),
                                tf.convert_to_tensor(seg["mask_final"]))
        ultimo_v = (pi_f.numpy() * q_f.numpy()).sum(-1).astype(np.float32)

        ret = retrace(seg["rew"], seg["done"], q_a, v, rho_barra, ultimo_v, cfg.gamma)
        return ret

    def _aprender(self, seg):
        cfg = self.cfg
        T, N = seg["act"].shape
        ret = self._alvo_retrace(seg)

        perda_q, ent, rho = self._passo(
            tf.convert_to_tensor(seg["obs"].reshape(T * N, *seg["obs"].shape[2:])),
            tf.convert_to_tensor(seg["mask"].reshape(T * N, N_ACTIONS)),
            tf.convert_to_tensor(seg["act"].reshape(T * N)),
            tf.convert_to_tensor(seg["mu"].reshape(T * N, N_ACTIONS)),
            tf.convert_to_tensor(ret.reshape(T * N)),
            cfg.ent_coef, cfg.q_coef, cfg.c_trunc, cfg.delta, cfg.trust_region,
        )

        # média de Polyak da política âncora
        if cfg.trust_region:
            p = cfg.polyak
            self.media.set_weights([
                p * a + (1.0 - p) * b
                for a, b in zip(self.media.get_weights(), self.model.get_weights())
            ])
        return float(perda_q), float(ent), float(rho)

    # -------------------------------------------------------------------- passo
    def iterate(self):
        cfg = self.cfg
        seg, stats = self.collect()

        perdas = [self._aprender(seg)]                      # on-policy
        if len(self.memoria) >= cfg.warmup_segments:        # off-policy
            for _ in range(cfg.replay_ratio):
                perdas.append(self._aprender(self.memoria.sample()))

        q, e, r = (float(np.mean(x)) for x in zip(*perdas))
        stats.update({"loss_q": q, "entropia": e, "rho_medio": r,
                      "updates": len(perdas)})
        return stats
