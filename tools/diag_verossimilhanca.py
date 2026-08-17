"""Erro quadrático em canal binário esparso é uma verossimilhança ruim. Trocá-la resolve?

O que já está medido
--------------------
O latente carrega ~1 nat (`kl_crua ≈ 1,07`) e o decoder é praticamente branco: o `argmax`
do canal de comida acerta 1,22% contra 1,00% do acaso, e o erro de cada canal é o de prever
zero em todo lugar. Localizar a comida entre 100 células exige `ln 100 = 4,6` nats — o
latente não tem nem um quarto disso. Nada no sonho pode funcionar a partir daí.

Por que a reconstrução não empurra o latente
--------------------------------------------
`p_recon = Σ (recon − symlog(obs))²` sobre 10×10×5. Três dos cinco canais são **binários e
de suporte mínimo**: corpo (≈3 células acesas), cabeça (1) e comida (1). Num canal com 1
célula acesa em 100, prever zero em todo lugar custa `symlog(1)² = 0,48` — e acertar a
célula economiza no máximo esses 0,48. Ao lado de `0,5·kl_dyn + 0,1·kl_rep`, que valem 0,6
sozinhos, guardar a posição da comida no latente simplesmente **não se paga**.

O DreamerV3 não tem esse problema em Atari porque lá são 64×64×3 = 12.288 elementos densos
e a reconstrução vale centenas de nats — as mesmas constantes 0,5 e 0,1 ficam
insignificantes ao lado dela. A diferença não é de hiperparâmetro, é de **qual
verossimilhança combina com o dado**: o próprio paper manda escolher a distribuição da
observação, e a de um canal binário é Bernoulli, não gaussiana.

As variantes
------------
=====================  ==========================================================
`bernoulli`             entropia cruzada nos canais binários (corpo, cabeça,
                        comida) e erro quadrático nos contínuos (decaimento,
                        comprimento) — a verossimilhança que combina com o dado
`recon_x10`             o controle bruto: mesma perda de hoje, peso 10. Se ele
                        empatar com `bernoulli`, era só escala
`bernoulli_x3`          se a direção estiver certa mas fraca
=====================  ==========================================================

Métrica: `acerto_argmax_comida`. Acaso = 0,0100.

Cuidado ao ler a saída: `erro_por_canal` vem de `diag_latente.mede`, que compara a saída do
decoder com `symlog(obs)`. No modo `bernoulli` o decoder passa a emitir **logits**, então
esse campo fica na casa dos milhares e **não quer dizer nada** — não é regressão, é unidade
diferente. O `acerto_argmax_comida` é invariante a escala e é o número que decide.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag_latente import I_COMIDA, NOMES, PASSOS, mede, tensores
from snakeai.agents.dreamerv3 import DreamerV3, DreamerV3Config
from snakeai.env.vec_snake import N_ACTIONS
from snakeai.nets.dreamer import symlog, two_hot

#: Canais binários da observação do contrato: corpo, cabeça, comida.
BINARIOS = [0, 1, 3]
CONTINUOS = [2, 4]

VARIANTES = {
    "bernoulli":     dict(modo="bernoulli", peso=1.0, lr=3e-4),
    "recon_x10":     dict(modo="mse", peso=10.0, lr=3e-4),
    "bernoulli_x3":  dict(modo="bernoulli", peso=3.0, lr=3e-4),
}


def erro_de_reconstrucao(recon, obs, modo):
    """Soma sobre os elementos da observação, como no paper — mas com a lei certa por canal."""
    if modo == "mse":
        return tf.reduce_sum(tf.square(recon - symlog(obs)), axis=[1, 2, 3])
    b = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.gather(obs, BINARIOS, axis=-1), tf.gather(recon, BINARIOS, axis=-1)),
        axis=[1, 2, 3])
    c = tf.reduce_sum(
        tf.square(tf.gather(recon, CONTINUOS, axis=-1)
                  - symlog(tf.gather(obs, CONTINUOS, axis=-1))), axis=[1, 2, 3])
    return b + c


def perda(ag, lote, modo, peso, kl_free=1.0):
    h, z, lg_q, lg_p = ag._desenrola(lote["obs"], lote["act"], lote["first"])
    plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado])

    recon = ag.decoder(plano)
    obs = tf.reshape(lote["obs"], tf.shape(recon))
    p_recon = tf.reduce_mean(erro_de_reconstrucao(recon, obs, modo))

    r_lg, c_lg, m_lg = ag.cabecas(plano)
    p_rec = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        two_hot(symlog(tf.reshape(lote["rew"], [-1])), ag.bins), r_lg))
    p_cont = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.reshape(lote["cont"], [-1, 1]), c_lg))
    p_mask = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.cast(tf.reshape(lote["mask"], [-1, N_ACTIONS]), tf.float32), m_lg))

    kl_cru = tf.reduce_mean(ag._kl(tf.stop_gradient(lg_q), lg_p))
    total = (peso * p_recon + p_rec + p_cont + p_mask
             + ag.cfg.kl_dyn * tf.maximum(kl_free, kl_cru)
             + ag.cfg.kl_rep * tf.maximum(
                 kl_free, tf.reduce_mean(ag._kl(lg_q, tf.stop_gradient(lg_p)))))
    return total, {"recon": p_recon, "kl_cru": kl_cru}


def roda(memoria, nome, v):
    ag = DreamerV3(DreamerV3Config(num_envs=64, eval_every_steps=10**12,
                                   lr_modelo=v["lr"]))
    ag.memoria = memoria

    @tf.function(reduce_retracing=True)
    def passo(lote):
        with tf.GradientTape() as tape:
            total, partes = perda(ag, lote, v["modo"], v["peso"])
        vs = ag._vars_modelo()
        ag.opt_modelo.apply_gradients(zip(tape.gradient(total, vs), vs))
        return partes

    for i in range(PASSOS):
        partes = passo(tensores(ag))
        if (i + 1) % 500 == 0:
            print(f"  [{nome}] {i + 1}/{PASSOS} · recon {float(partes['recon']):.3f} "
                  f"· kl_crua {float(partes['kl_cru']):.3f}", flush=True)
    return mede(ag, 25)


if __name__ == "__main__":
    base = DreamerV3(DreamerV3Config(num_envs=64, eval_every_steps=10**12))
    print("coletando...", flush=True)
    while len(base.memoria) < 120_000:
        base.collect()
    print(f"  {len(base.memoria):,} transições", flush=True)

    saida = {}
    for nome, v in VARIANTES.items():
        print(f"--- {nome}: {v} ---", flush=True)
        saida[nome] = {**v, **roda(base.memoria, nome, v)}
        print(json.dumps(saida[nome], indent=1), flush=True)
        with open("/tmp/diag_verossimilhanca.json", "w") as f:
            json.dump(saida, f, indent=1)

    print("=== acerto do argmax da comida (acaso = 0,0100) ===", flush=True)
    for nome, r in saida.items():
        print(f"  {nome:<16} {r['acerto_argmax_comida']:.4f}   "
              f"kl_crua {r['kl_crua_nats']:>8.3f} nats", flush=True)
