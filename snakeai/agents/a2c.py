"""A2C — actor-critic síncrono, o controle experimental do PPO.

O A2C é o PPO sem as duas coisas que definem o PPO: **sem clipping da razão** e **uma
única passada de gradiente por rollout**. Tudo o mais é igual — mesmo ambiente, mesma
rede, mesmo GAE, mesmo bootstrap de truncamento, mesmo agendamento de entropia.

Por isso ele é mais que "mais um algoritmo": é o **controle experimental**. A diferença
entre a curva do PPO e a do A2C mede exatamente quanto valem o clipping e o reaproveitamento
do rollout, com todo o resto congelado. Sem esse controle, o ganho do PPO poderia ser do
ambiente novo, da rede residual ou do shaping — e não haveria como saber.

A herança direta de `PPO` é deliberada: garante que o `collect()` seja *literalmente* o
mesmo código, não uma cópia que diverge com o tempo. Uma correção no rollout vale para os
dois na hora.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ..eval import MASK_NEG
from .ppo import PPO, PPOConfig, variancia_explicada

__all__ = ["A2CConfig", "A2C"]


@dataclass
class A2CConfig(PPOConfig):
    #: Rollouts curtos são o normal em A2C: sem clipping, dar passos grandes com dados
    #: velhos desestabiliza. O PPO aguenta 96 porque o clipping segura.
    rollout: int = 16
    lr_start: float = 7e-4
    lr_end: float = 1e-4
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.002
    vf_coef: float = 0.5

    #: Campos do PPO que não existem aqui. Ficam para o `dataclass` não brigar, mas o
    #: `A2C` ignora — e o teste `test_a2c_ignores_ppo_only_knobs` garante que ignora.
    epochs: int = 1
    minibatches: int = 1
    clip_eps: float = 0.0
    vf_clip: float = 0.0
    target_kl: float = 0.0


class A2C(PPO):
    algo = "a2c"

    def __init__(self, cfg: A2CConfig = None, model=None, variant=None):
        super().__init__(cfg or A2CConfig(), model=model, variant=variant)

    @staticmethod
    @tf.function(reduce_retracing=True)
    def _train_step_a2c(model, optimizer, obs, mask, act, adv, ret, ent_coef, vf_coef):
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
        with tf.GradientTape() as tape:
            logits, valor = model(obs, training=True)
            valor = tf.squeeze(valor, -1)
            # mesma regra do PPO: a máscara vale no update também
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp_all = tf.nn.log_softmax(logits)
            logp = tf.gather(logp_all, act, batch_dims=1)

            # o gradiente de política puro: sem razão, sem clipping
            pg_loss = -tf.reduce_mean(logp * adv)
            v_loss = 0.5 * tf.reduce_mean(tf.square(valor - ret))

            probs = tf.exp(logp_all)
            seguro = tf.where(mask, logp_all, tf.zeros_like(logp_all))
            entropia = -tf.reduce_mean(tf.reduce_sum(probs * seguro, axis=-1))

            perda = pg_loss + vf_coef * v_loss - ent_coef * entropia

        grads = tape.gradient(perda, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return pg_loss, v_loss, entropia

    def update(self, lote):
        """Uma passada de gradiente sobre o rollout inteiro, e o dado é descartado.

        É esta linha que separa o A2C do PPO: sem clipping, reaproveitar o rollout por
        várias épocas faria a política se afastar demais dos dados que a geraram.
        """
        cfg = self.cfg
        self.optimizer.learning_rate.assign(self.lr())
        ent = self.ent_coef()

        pg, vf, e = self._train_step_a2c(
            self.model, self.optimizer,
            tf.convert_to_tensor(lote["obs"]), tf.convert_to_tensor(lote["mask"]),
            tf.convert_to_tensor(lote["act"]), tf.convert_to_tensor(lote["adv"]),
            tf.convert_to_tensor(lote["ret"]),
            ent, cfg.vf_coef,
        )
        return {
            "pg": float(pg), "vf": float(vf), "ent": float(e),
            "lr": float(self.lr()), "ent_coef": ent, "epochs_done": 1,
            "atualizacoes": 1,
            "ev": variancia_explicada(lote["val"], lote["ret"]),
        }
