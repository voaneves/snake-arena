"""O latente do Dreamer não sabe onde está a comida. Qual termo da perda é o culpado?

O que já está medido
--------------------
`tools/diag_cabecas.py` fechou a pergunta anterior: o `argmax` do canal de comida
reconstruído acerta **1,18%** das células contra 1,00% do puro acaso, e o erro somado de
cada canal é praticamente o de prever **zero em todo lugar** (comida 0,505 contra 0,480 do
branco; corpo 1,448 contra ~1,44). Uma cabeça de recompensa treinada por 25 épocas sobre
30 mil estados prevê `0,0010` quando a recompensa verdadeira é `+1` e `0,0010` quando é
`0` — e **dar a ação a ela não muda nada**, o que só pode significar que o latente não
carrega o tabuleiro. O resto do colapso é consequência: sonho vazio → vantagem é ruído →
entropia presa em `ln 3` → deriva → score abaixo do piso aleatório.

A hipótese sobre o porquê
-------------------------
É de **escala entre os termos da perda**, e ela nasce de uma diferença entre este ambiente e
o do paper. O DreamerV3 soma o erro de reconstrução sobre os pixels: em Atari são
64×64×3 = 12.288 elementos e a reconstrução vale centenas. Aqui são 10×10×5 = 500
elementos quase todos **zero**, e a reconstrução inteira vale ~3,5 — cem vezes menos. Ao
lado dela, `kl_free = 1,0` nat: enquanto a KL fica abaixo de 1 nat, o `tf.maximum` zera o
gradiente do termo (é o objetivo do free bits), e o único termo que empurraria o latente a
carregar informação é justamente o mais fraco. O latente fica podendo ser inútil de graça.

O que este script decide
------------------------
Treina o modelo do mundo a partir do **mesmo buffer** em variantes que mexem em um termo
cada, e mede em cada uma a única coisa que importa: o latente sabe onde está a comida?

==================  ============================================================
variante             a pergunta que responde
==================  ============================================================
`atual`              a linha de base, para os números serem comparáveis
`sem_free_bits`      `kl_free=0`: o free bits está deixando o latente vazio?
`recon_x10`          é escala? Se sim, pesar a reconstrução resolve
`lr_3e-4`            é só lentidão? Ou 1e-4 é lento demais para 500 elementos
`recon_x10_sem_fb`   os dois juntos, se cada um sozinho não bastar
==================  ============================================================

Métrica principal: `acerto_argmax_comida`. É binária e não tem como enganar — ou o modelo
aponta a célula certa acima do acaso, ou não aprendeu o tabuleiro.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakeai.agents.dreamerv3 import DreamerV3, DreamerV3Config
from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS
from snakeai.nets.dreamer import symlog, two_hot

NOMES = ["corpo", "cabeca", "decaimento", "comida", "comprimento"]
I_COMIDA = NOMES.index("comida")
PASSOS = 2500

VARIANTES = {
    "atual":             dict(peso_recon=1.0, kl_free=1.0, lr=1e-4),
    "sem_free_bits":     dict(peso_recon=1.0, kl_free=0.0, lr=1e-4),
    "recon_x10":         dict(peso_recon=10.0, kl_free=1.0, lr=1e-4),
    "lr_3e-4":           dict(peso_recon=1.0, kl_free=1.0, lr=3e-4),
    "recon_x10_sem_fb":  dict(peso_recon=10.0, kl_free=0.0, lr=3e-4),
}


def perda(ag, lote, peso_recon, kl_free):
    """`_perda_modelo` com os dois termos suspeitos como parâmetro, e a KL **sem clamp**."""
    cfg = ag.cfg
    h, z, lg_q, lg_p = ag._desenrola(lote["obs"], lote["act"], lote["first"])
    plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado])

    recon = ag.decoder(plano)
    alvo = symlog(tf.reshape(lote["obs"], tf.shape(recon)))
    p_recon = tf.reduce_mean(tf.reduce_sum(tf.square(recon - alvo), axis=[1, 2, 3]))

    r_lg, c_lg, m_lg = ag.cabecas(plano)
    p_rec = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        two_hot(symlog(tf.reshape(lote["rew"], [-1])), ag.bins), r_lg))
    p_cont = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.reshape(lote["cont"], [-1, 1]), c_lg))
    p_mask = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.cast(tf.reshape(lote["mask"], [-1, N_ACTIONS]), tf.float32), m_lg))

    kl_cru = tf.reduce_mean(ag._kl(tf.stop_gradient(lg_q), lg_p))
    kl_dyn = tf.maximum(kl_free, kl_cru)
    kl_rep = tf.maximum(kl_free, tf.reduce_mean(ag._kl(lg_q, tf.stop_gradient(lg_p))))

    total = (peso_recon * p_recon + p_rec + p_cont + p_mask
             + cfg.kl_dyn * kl_dyn + cfg.kl_rep * kl_rep)
    return total, {"recon": p_recon, "kl_cru": kl_cru, "rec": p_rec}


def mede(ag, lotes):
    """Erro por canal, acerto do `argmax` da comida, e a KL crua."""
    erros, kls = np.zeros(N_CHANNELS), []
    acertos = total = 0
    for _ in range(lotes):
        lote = tensores(ag)
        h, z, lg_q, lg_p = ag._desenrola(lote["obs"], lote["act"], lote["first"])
        plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado])
        recon = ag.decoder(plano).numpy()
        alvo = symlog(tf.reshape(lote["obs"], tf.shape(recon))).numpy()
        erros += ((recon - alvo) ** 2).sum(axis=(0, 1, 2)) / recon.shape[0]
        kls.append(float(tf.reduce_mean(ag._kl(lg_q, lg_p))))

        p = recon[..., I_COMIDA].reshape(len(recon), -1).argmax(1)
        r = alvo[..., I_COMIDA].reshape(len(alvo), -1).argmax(1)
        acertos += int((p == r).sum())
        total += len(r)
    return {
        "acerto_argmax_comida": round(acertos / total, 4),
        "kl_crua_nats": round(float(np.mean(kls)), 3),
        "erro_por_canal": {n: round(float(e / lotes), 3) for n, e in zip(NOMES, erros)},
    }


def tensores(ag):
    l = ag.memoria.sample(ag.cfg.batch_size, ag.cfg.seq_len)
    return {"obs": tf.convert_to_tensor(l["obs"], tf.float32),
            "act": tf.convert_to_tensor(l["act"], tf.int32),
            "rew": tf.convert_to_tensor(l["rew"], tf.float32),
            "cont": tf.convert_to_tensor(l["cont"], tf.float32),
            "first": tf.convert_to_tensor(l["first"], tf.bool),
            "mask": tf.convert_to_tensor(l["mask"], tf.bool)}


def roda_variante(memoria, nome, cfg_var, seed=0):
    cfg = DreamerV3Config(num_envs=64, eval_every_steps=10**12, seed=seed,
                          lr_modelo=cfg_var["lr"])
    ag = DreamerV3(cfg)
    ag.memoria = memoria                                   # mesmíssimos dados nas variantes

    @tf.function(reduce_retracing=True)
    def passo(lote):
        with tf.GradientTape() as tape:
            total, partes = perda(ag, lote, cfg_var["peso_recon"], cfg_var["kl_free"])
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
    print("coletando com política aleatória...", flush=True)
    while len(base.memoria) < 120_000:
        base.collect()
    print(f"  {len(base.memoria):,} transições · acaso = "
          f"{1 / base.cfg.board_size ** 2:.4f}", flush=True)

    saida = {}
    for nome, v in VARIANTES.items():
        print(f"--- {nome}: {v} ---", flush=True)
        saida[nome] = {**v, **roda_variante(base.memoria, nome, v)}
        print(json.dumps(saida[nome], indent=1), flush=True)
        with open("/tmp/diag_latente.json", "w") as f:
            json.dump(saida, f, indent=1)

    print("=== resumo: acerto do argmax da comida (acaso = 0,0100) ===", flush=True)
    for nome, r in saida.items():
        print(f"  {nome:<20} {r['acerto_argmax_comida']:.4f}   "
              f"kl_crua {r['kl_crua_nats']:>7.3f} nats   "
              f"comida {r['erro_por_canal']['comida']:.3f}", flush=True)
