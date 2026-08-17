"""Onde exatamente a comida se perde: encoder, posterior, ou decoder?

Por que esta é a última pergunta
-------------------------------
A varredura de `tools/diag_latente.py` deu um resultado limpo e estranho. Com a
reconstrução pesada e sem free bits, o modelo do mundo aprende o tabuleiro **muito bem** —
erro do canal do corpo cai de 1,397 para 0,373, cabeça de 0,483 para 0,105, decaimento de
0,958 para 0,145. E a comida não sai do lugar em **nenhuma** variante::

    atual              argmax 0,0122   kl 1,07 nats    comida 0,494
    sem_free_bits      argmax 0,0113   kl 0,35 nats    comida 0,488
    recon_x10          argmax 0,0111   kl 10,29 nats   comida 0,499
    lr_3e-4            argmax 0,0108   kl 2,09 nats    comida 0,496
    recon_x10_sem_fb   argmax 0,0104   kl 3,81 nats    comida 0,489
                                       (acaso = 0,0100)

Isso mata a hipótese de escala como explicação **da comida**: com 10 nats de orçamento o
latente aprende tudo, menos ela. E mata também "o decoder não sabe pintar uma célula
sozinha" — a cabeça é uma célula sozinha e ele acerta.

Então a comida é categoricamente diferente, e há três lugares onde ela pode desaparecer.
Este script mede os três **no mesmo modelo treinado**, com uma sonda supervisionada de
100 classes (a célula da comida) em cima de cada representação:

===================  ==========================================================
sonda                 se ela falhar, o culpado é
===================  ==========================================================
`enc(obs)`            o encoder — a informação morre antes de chegar ao RSSM
`(h, z)` posterior    o gargalo do posterior: `Dense(256)` para 100 posições
`argmax` do decoder   só o decoder; o latente sabe e ele não pinta
===================  ==========================================================

A sonda é treinada com os pesos do modelo **congelados**: ela mede o que a representação
contém, não o que ela poderia aprender a conter.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf
from keras import layers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag_latente import I_COMIDA, tensores
from snakeai.agents.dreamerv3 import DreamerV3, DreamerV3Config
from snakeai.env.vec_snake import N_ACTIONS
from snakeai.nets.dreamer import symlog, two_hot

PASSOS_MODELO = 2500
PESO_RECON, KL_FREE, LR = 10.0, 0.0, 3e-4       # a melhor variante da varredura
LOTES_SONDA = 250


def perda(ag, lote):
    h, z, lg_q, lg_p = ag._desenrola(lote["obs"], lote["act"], lote["first"])
    plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado])
    recon = ag.decoder(plano)
    obs = tf.reshape(lote["obs"], tf.shape(recon))
    p_recon = tf.reduce_mean(tf.reduce_sum(tf.square(recon - symlog(obs)), [1, 2, 3]))
    r_lg, c_lg, m_lg = ag.cabecas(plano)
    p = (PESO_RECON * p_recon
         + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
             two_hot(symlog(tf.reshape(lote["rew"], [-1])), ag.bins), r_lg))
         + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             tf.reshape(lote["cont"], [-1, 1]), c_lg))
         + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             tf.cast(tf.reshape(lote["mask"], [-1, N_ACTIONS]), tf.float32), m_lg))
         + ag.cfg.kl_dyn * tf.maximum(KL_FREE, tf.reduce_mean(
             ag._kl(tf.stop_gradient(lg_q), lg_p)))
         + ag.cfg.kl_rep * tf.maximum(KL_FREE, tf.reduce_mean(
             ag._kl(lg_q, tf.stop_gradient(lg_p)))))
    return p, p_recon


def colhe(ag, lotes):
    """`enc(obs)`, `(h,z)`, o `argmax` do decoder, e a célula verdadeira da comida."""
    EMB, EST, PREV, ALVO = [], [], [], []
    for _ in range(lotes):
        lote = tensores(ag)
        h, z, _, _ = ag._desenrola(lote["obs"], lote["act"], lote["first"])
        obs2d = tf.reshape(lote["obs"], [-1, ag.cfg.board_size, ag.cfg.board_size, 5])
        EMB.append(ag.encoder(obs2d).numpy())
        plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado]).numpy()
        EST.append(plano)
        recon = ag.decoder(plano).numpy()
        PREV.append(recon[..., I_COMIDA].reshape(len(recon), -1).argmax(1))
        ALVO.append(obs2d.numpy()[..., I_COMIDA].reshape(len(recon), -1).argmax(1))
    return [np.concatenate(x) for x in (EMB, EST, PREV, ALVO)]


def sonda(X, y, n_classes, epocas=15, lote=1024, largura=512):
    """Classificador de 100 classes em cima de uma representação **congelada**."""
    m = keras.Sequential([layers.Input((X.shape[1],)),
                          layers.Dense(largura, activation="silu"),
                          layers.Dense(largura, activation="silu"),
                          layers.Dense(n_classes)])
    m.compile(keras.optimizers.Adam(1e-3),
              keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              [keras.metrics.SparseCategoricalAccuracy()])
    n = len(X)
    corte = int(0.85 * n)
    h = m.fit(X[:corte], y[:corte], batch_size=lote, epochs=epocas, verbose=0,
              validation_data=(X[corte:], y[corte:]))
    return round(float(h.history["val_sparse_categorical_accuracy"][-1]), 4)


if __name__ == "__main__":
    ag = DreamerV3(DreamerV3Config(num_envs=64, eval_every_steps=10**12, lr_modelo=LR))
    print("coletando...", flush=True)
    while len(ag.memoria) < 120_000:
        ag.collect()

    @tf.function(reduce_retracing=True)
    def passo(lote):
        with tf.GradientTape() as tape:
            total, recon = perda(ag, lote)
        vs = ag._vars_modelo()
        ag.opt_modelo.apply_gradients(zip(tape.gradient(total, vs), vs))
        return recon

    print(f"treinando o modelo do mundo ({PASSOS_MODELO} passos, "
          f"peso_recon={PESO_RECON}, kl_free={KL_FREE})...", flush=True)
    for i in range(PASSOS_MODELO):
        r = passo(tensores(ag))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{PASSOS_MODELO} · recon {float(r):.3f}", flush=True)

    print("colhendo representações...", flush=True)
    EMB, EST, PREV, ALVO = colhe(ag, LOTES_SONDA)
    n_cel = ag.cfg.board_size ** 2
    print(f"  {len(EMB):,} estados", flush=True)

    r = {
        "acaso": round(1.0 / n_cel, 4),
        "sonda_no_encoder": sonda(EMB, ALVO, n_cel),
        "sonda_no_latente_posterior": sonda(EST, ALVO, n_cel),
        "argmax_do_decoder": round(float((PREV == ALVO).mean()), 4),
    }
    print(json.dumps(r, indent=1), flush=True)
    with open("/tmp/diag_comida.json", "w") as f:
        json.dump(r, f, indent=1)
