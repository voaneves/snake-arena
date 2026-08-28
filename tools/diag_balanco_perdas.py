"""Quanto do gradiente do tronco compartilhado do AlphaZero vem do valor, e quanto da política.

A pergunta
----------
`06_alphazero` chega a 1 M de passos com a **busca** fazendo 17,8 e a **política pura** em
2,45 — mal acima do piso aleatório de 1,21. A busca funciona; a destilação é que não anda.

Uma causa possível é aritmética, não conceitual. O AlphaZero original treina o valor contra
o resultado da partida, em `[-1, 1]`: `perda_v` e `perda_pi` nascem na mesma ordem de
grandeza e `vf_coef = 1` é um número razoável. Aqui o alvo de valor é um retorno descontado
**não normalizado** — com `γ = 0,997` e uma maçã a cada ~37 passos ele vale ~9, e cresce
conforme o agente melhora. A perda de política é uma entropia cruzada sobre 3 ações, presa
perto de `ln 3 ≈ 1,10`. As duas dividem o mesmo tronco convolucional.

O PPO deste repositório não tem o problema porque normaliza a vantagem por minilote
(`ppo.py:303`), o que torna o gradiente de política invariante à escala do valor. O
AlphaZero não normaliza nada.

Este script mede a razão `‖∂perda_v/∂tronco‖ / ‖∂perda_pi/∂tronco‖` em função da escala do
alvo, com e sem `valor_symlog`. Só o tronco entra na conta: as cabeças têm parâmetros
próprios e não competem.

Uso::

    python tools/diag_balanco_perdas.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from snakeai.agents import AlphaZero, AlphaZeroConfig      # noqa: E402

#: Multiplicadores do alvo. `1x` é o regime dos primeiros milhares de passos, quando o
#: agente quase não come; `20x` é o de 1 M de passos com score de busca ~17.
ESCALAS = (1, 5, 10, 20, 40)
ITERACOES = 40
LOTE = 256


def _tronco(modelo):
    """Só o tronco: as cabeças têm parâmetros próprios e não disputam capacidade."""
    return [v for v in modelo.trainable_variables
            if not v.path.startswith(("logits", "value", "pi_", "v_"))]


def mede(valor_symlog, seed=0):
    cfg = AlphaZeroConfig(
        net="resnet_tiny", num_envs=32, rollout=16, num_simulations=8, batch_size=LOTE,
        memory_size=50_000, total_steps=10 ** 9, eval_every_steps=10 ** 9,
        log_every_steps=10 ** 9, salvar_gif=False, salvar_grafico=False, seed=seed,
        valor_symlog=valor_symlog)
    ag = AlphaZero(cfg)
    for _ in range(ITERACOES):        # enche o buffer com alvos de verdade
        ag.iterate()

    tronco = _tronco(ag.model)
    idx = np.arange(LOTE)
    obs = tf.convert_to_tensor(ag._buf_obs[idx])
    mask = tf.convert_to_tensor(ag._buf_mask[idx])
    pi_alvo = tf.convert_to_tensor(ag._buf_pi[idx])

    linhas = []
    for k in ESCALAS:
        z = tf.convert_to_tensor(ag._buf_z[idx] * k)
        with tf.GradientTape(persistent=True) as fita:
            logits, valor = ag.model(obs, training=True)
            valor = tf.squeeze(valor, -1)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e9))
            perda_pi = -tf.reduce_mean(
                tf.reduce_sum(pi_alvo * tf.nn.log_softmax(logits), axis=-1))
            alvo = ag._symlog(z) if valor_symlog else z
            perda_v = tf.reduce_mean(tf.square(valor - alvo))
        g_pi = float(tf.linalg.global_norm(
            [g for g in fita.gradient(perda_pi, tronco) if g is not None]))
        g_v = float(tf.linalg.global_norm(
            [g for g in fita.gradient(perda_v, tronco) if g is not None]))
        del fita
        linhas.append({"escala": k, "z_medio_abs": round(float(tf.reduce_mean(tf.abs(z))), 2),
                       "perda_pi": round(float(perda_pi), 3),
                       "perda_v": round(float(perda_v), 3),
                       "razao_grad_v_sobre_pi": round(g_v / max(g_pi, 1e-9), 1)})
    return linhas


def main():
    saida = {}
    for symlog in (False, True):
        chave = "com_symlog" if symlog else "sem_symlog"
        print(f"--- valor_symlog={symlog} ---", flush=True)
        print(f"{'escala':>7} {'|z| médio':>10} {'perda_pi':>9} {'perda_v':>10} "
              f"{'‖∇v‖/‖∇π‖':>12}", flush=True)
        saida[chave] = mede(symlog)
        for l in saida[chave]:
            print(f"{l['escala']:>6}x {l['z_medio_abs']:>10.2f} {l['perda_pi']:>9.3f} "
                  f"{l['perda_v']:>10.3f} {l['razao_grad_v_sobre_pi']:>11.1f}x", flush=True)
    destino = os.path.join(RAIZ, "docs", "diag_balanco_perdas.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=1, ensure_ascii=False)
    print("gravado em", destino)


if __name__ == "__main__":
    main()
