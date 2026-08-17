"""O modelo do mundo do Dreamer sabe onde está a comida? E consegue prever comer?

Por que esta pergunta é a certa
------------------------------
O diagnóstico de treino mostrou o modelo reconstruindo a observação bem (erro somado ~3,3
sobre 500 elementos) e ao mesmo tempo prevendo recompensa ≈ 0 e continuação ≈ 0,99 em
**todo** estado, terminais inclusive. Num sonho onde nada acontece e nada acaba, toda ação
vale o mesmo: a vantagem é ruído e a entropia do ator fica presa em `ln 3 = 1,0986`.

Só que "reconstrói bem" é média sobre 5 canais de 100 células. A comida é **uma** célula
acesa em 100. Prever o canal de comida como zero em todo lugar custa `symlog(1)² = 0,48`
num total que começa perto de 28 — menos de 2%. Ou seja: o modelo pode **ignorar a comida
por completo** e ainda parecer ótimo no erro somado. E se o latente não sabe onde a comida
está, nenhuma cabeça de recompensa pode prever comer, com ação ou sem.

O que este script mede
----------------------
1. **Erro de reconstrução por canal**, e a taxa de acerto do `argmax` do canal de comida —
   a pergunta "o modelo põe a comida na célula certa?" tem resposta binária.
2. **A/B da cabeça de recompensa**: `(h,z)` contra `(h,z,a)`, treinadas sobre os mesmos
   estados latentes e os mesmos alvos. Comer é entrar na célula da comida: dado o estado,
   depende inteiramente de qual das 3 ações foi escolhida, e a cabeça de hoje não vê a
   ação. Se a versão com ação separa `r=+1` de `r=0` e a sem ação não, o desalinhamento de
   índice está medido. Se **nenhuma das duas** separa, o problema é anterior: está no
   latente, e é o item 1 que manda.

Cuidado de método: comer é ~1% das transições e morrer ~0,1%. A primeira versão deste
script usou 20 mil estados e 8 épocas e as duas cabeças ficaram na taxa base — não porque
sejam iguais, mas porque não havia dados nem passos suficientes para sair da base. Aqui são
~200 mil estados, treino mais longo, e as classes raras aparecem separadas na saída.
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

from snakeai.agents.dreamerv3 import DreamerV3, DreamerV3Config
from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS
from snakeai.nets.dreamer import bins_simetricos, de_two_hot, symexp, symlog, two_hot

NOMES_CANAIS = ["corpo", "cabeca", "decaimento", "comida", "comprimento"]
PASSOS_MODELO = 1500
LOTES_AVAL = 400


def treinador(ag):
    """Passo de treino do modelo do mundo em **grafo** — em eager isto leva ~2 s por passo."""
    @tf.function(reduce_retracing=True)
    def passo(obs, act, rew, cont, first, mask):
        lote = {"obs": obs, "act": act, "rew": rew, "cont": cont, "first": first,
                "mask": mask}
        with tf.GradientTape() as tape:
            perda, partes, _ = ag._perda_modelo(lote)
        vs = ag._vars_modelo()
        ag.opt_modelo.apply_gradients(zip(tape.gradient(perda, vs), vs))
        return partes
    return passo


def amostra(ag):
    lote = ag.memoria.sample(ag.cfg.batch_size, ag.cfg.seq_len)
    return (tf.convert_to_tensor(lote["obs"], tf.float32),
            tf.convert_to_tensor(lote["act"], tf.int32),
            tf.convert_to_tensor(lote["rew"], tf.float32),
            tf.convert_to_tensor(lote["cont"], tf.float32),
            tf.convert_to_tensor(lote["first"], tf.bool),
            tf.convert_to_tensor(lote["mask"], tf.bool)), lote


# ------------------------------------------------------- 1. reconstrução por canal
def reconstrucao_por_canal(ag, lotes=30):
    """Erro por canal e acerto do `argmax` da comida — o teste que a média esconde."""
    erros = np.zeros(N_CHANNELS)
    acertos = total = 0
    for _ in range(lotes):
        t, lote = amostra(ag)
        h, z, _, _ = ag._desenrola(t[0], t[1], t[4])
        plano = tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado])
        recon = ag.decoder(plano).numpy()
        alvo = symlog(tf.reshape(t[0], tf.shape(recon))).numpy()
        erros += ((recon - alvo) ** 2).sum(axis=(0, 1, 2)) / recon.shape[0]

        # a comida está numa célula só: o modelo aponta a certa?
        c = NOMES_CANAIS.index("comida")
        prev = recon[..., c].reshape(recon.shape[0], -1).argmax(axis=1)
        real = alvo[..., c].reshape(alvo.shape[0], -1).argmax(axis=1)
        acertos += int((prev == real).sum())
        total += len(real)
    return {
        "erro_somado_por_canal": {n: round(float(e / lotes), 4)
                                  for n, e in zip(NOMES_CANAIS, erros)},
        "acerto_argmax_comida": acertos / total,
        "acerto_ao_acaso": 1.0 / (ag.cfg.board_size ** 2),
        "estados": total,
    }


# ------------------------------------------------------------ 2. A/B da recompensa
def colhe(ag, lotes):
    E, A, R, C = [], [], [], []
    for _ in range(lotes):
        t, lote = amostra(ag)
        h, z, _, _ = ag._desenrola(t[0], t[1], t[4])
        E.append(tf.reshape(tf.concat([h, z], -1), [-1, ag.dim_estado]).numpy())
        A.append(lote["act"].reshape(-1))
        R.append(lote["rew"].reshape(-1))
        C.append(lote["cont"].reshape(-1))
    return (np.concatenate(E), np.concatenate(A).astype(np.int32),
            np.concatenate(R).astype(np.float32), np.concatenate(C).astype(np.float32))


def cabeca(dim, com_acao, n_bins, largura=256):
    e = keras.Input(shape=(dim,))
    entradas = [e]
    x = e
    if com_acao:
        a = keras.Input(shape=(N_ACTIONS,))
        entradas.append(a)
        x = layers.Concatenate()([x, a])
    for _ in range(2):
        x = layers.Dense(largura, use_bias=False)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("silu")(x)
    return keras.Model(entradas, [layers.Dense(n_bins, name="r")(x),
                                  layers.Dense(1, name="c")(x)])


def ajusta_e_mede(E, A, R, C, bins, com_acao, epocas=25, lote=1024):
    n = len(E)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[:int(0.85 * n)], idx[int(0.85 * n):]
    m = cabeca(E.shape[1], com_acao, len(bins))
    opt = keras.optimizers.Adam(3e-4)
    A1h = np.eye(N_ACTIONS, dtype=np.float32)[A]
    alvo_r = two_hot(symlog(tf.convert_to_tensor(R)), bins).numpy()

    @tf.function(reduce_retracing=True)
    def passo(e, a, ar, c):
        with tf.GradientTape() as tape:
            r_lg, c_lg = m([e, a] if com_acao else [e], training=True)
            perda = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ar, r_lg))
                     + tf.reduce_mean(
                         tf.nn.sigmoid_cross_entropy_with_logits(c[:, None], c_lg)))
        opt.apply_gradients(zip(tape.gradient(perda, m.trainable_variables),
                                m.trainable_variables))
        return perda

    rng = np.random.default_rng(1)
    for _ in range(epocas):
        rng.shuffle(tr)
        for i in range(0, len(tr) - lote, lote):
            b = tr[i:i + lote]
            passo(E[b], A1h[b], alvo_r[b], C[b])

    prev, p_cont = [], []
    for i in range(0, len(te), 4096):
        b = te[i:i + 4096]
        r_lg, c_lg = m([E[b], A1h[b]] if com_acao else [E[b]], training=False)
        prev.append(symexp(de_two_hot(r_lg, bins)).numpy())
        p_cont.append(tf.sigmoid(c_lg[:, 0]).numpy())
    prev, p_cont = np.concatenate(prev), np.concatenate(p_cont)
    Rte, Cte = R[te], C[te]

    por_r = {}
    for v in (-1.0, -0.5, 0.0, 1.0, 2.0):
        sel = np.isclose(Rte, v)
        por_r[f"r={v:+.1f}"] = {
            "n": int(sel.sum()),
            "previsao_media": round(float(prev[sel].mean()), 4) if sel.any() else None}
    return {
        "mae": round(float(np.abs(prev - Rte).mean()), 4),
        "previsao_por_recompensa_verdadeira": por_r,
        "p_cont_em_terminais": round(float(p_cont[Cte < 0.5].mean()), 4),
        "p_cont_em_continua": round(float(p_cont[Cte > 0.5].mean()), 4),
        "n_teste": int(len(te)),
    }


if __name__ == "__main__":
    cfg = DreamerV3Config(num_envs=64, eval_every_steps=10**12)
    ag = DreamerV3(cfg)
    print("coletando com política aleatória...", flush=True)
    while len(ag.memoria) < 120_000:
        ag.collect()
    print(f"  {len(ag.memoria):,} transições", flush=True)

    print(f"treinando o modelo do mundo ({PASSOS_MODELO} passos, em grafo)...", flush=True)
    passo = treinador(ag)
    for i in range(PASSOS_MODELO):
        t, _ = amostra(ag)
        partes = passo(*t)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{PASSOS_MODELO} · recon {float(partes['recon']):.3f} "
                  f"· rec {float(partes['rec']):.4f} "
                  f"· cont {float(partes['cont']):.4f}", flush=True)

    print("--- 1. reconstrução por canal ---", flush=True)
    print(json.dumps(reconstrucao_por_canal(ag), indent=1), flush=True)

    print("colhendo estados latentes...", flush=True)
    E, A, R, C = colhe(ag, LOTES_AVAL)
    print(f"  {len(E):,} estados · comeu {float((R > 0.5).mean()):.4f} · "
          f"terminou {float((C < 0.5).mean()):.4f}", flush=True)

    bins = bins_simetricos(cfg.n_bins)
    for com_acao in (False, True):
        print(f"--- 2. cabeça {'COM' if com_acao else 'SEM'} ação "
              f"{'(h,z,a)' if com_acao else '(h,z) — como está hoje'} ---", flush=True)
        print(json.dumps(ajusta_e_mede(E, A, R, C, bins, com_acao), indent=1), flush=True)
