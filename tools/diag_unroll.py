"""Quanto da perda de política do MuZero pertence ao passo que a métrica oficial mede.

A pergunta
----------
`runs/muzero/unroll5/seed0` terminou em **49,26** com o melhor ponto em **66,05** — 16,8
pontos acima do final. Não é mínimo local: o `train_score`, que é o da **busca**, fica
estável em 58–60 do passo 2,5 M até o fim. Quem oscila é a rede pura, entre 33 e 66. E
`perda_pi` **sobe** de 2,42 (3,25 M) para 3,09 (5,0 M) enquanto o `lr` desce — ou seja, o
professor está bom e o aluno está piorando de destilá-lo, com passo cada vez menor.

A aritmética que ninguém olha
-----------------------------
`perda_pi` é uma **soma crua** sobre `K+1` termos: o passo 0, que sai de `f(h(o))` — a
observação real, o único caminho que `politica()` usa em avaliação —, e `K` passos
imaginados, que saem de `f(g^k(...))`. Nenhum peso separa os dois. Então a fatia do passo 0
dentro do gradiente de política é ~`1/(K+1)`, e **cresce ou encolhe com `K`**.

O pseudocódigo do paper não faz isso: ele aplica `scale_gradient(loss, 1/K)` aos passos do
desenrolar, deixando o passo 0 inteiro. Com isso o passo 0 vale ~metade do gradiente,
qualquer que seja `K`. O repositório tinha a escala de 1/2 no estado oculto (que controla o
gradiente que chega em `h`) mas não a das perdas.

A consequência prática é contraintuitiva: **aumentar `unroll` sem o peso do paper piora a
métrica oficial**, porque dilui justamente o termo que a produz.

Este script mede a fatia do passo 0, `perda_pi_0 / perda_pi`, em função de `K`, com e sem
`normaliza_unroll`, sobre lotes reais coletados pelo agente.

Uso::

    python tools/diag_unroll.py
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

from snakeai.agents import MuZero, MuZeroConfig            # noqa: E402

#: Os desenrolares que interessam: 2 e 5 (o padrão) delimitam a faixa em que o agente
#: rodou, 10 é o que o instinto sugere quando o resultado oscila.
DESENROLARES = (1, 2, 3, 5, 10)
#: Lotes medidos por configuração. O buffer é recoletado a cada um para não medir sempre
#: as mesmas posições.
LOTES = 8


def mede(unroll, normaliza):
    """Coleta com o agente e devolve a fatia média do passo 0 sobre `LOTES` minilotes."""
    cfg = MuZeroConfig(net="resnet_tiny", num_envs=16, rollout=8, unroll=unroll,
                       num_simulations=6, batch_size=64, memory_size=20_000,
                       total_steps=10**6, eval_every_steps=10**9, log_every_steps=10**9,
                       salvar_gif=False, salvar_grafico=False,
                       normaliza_unroll=normaliza)
    ag = MuZero(cfg)
    fatias, perdas, zeros = [], [], []
    for _ in range(LOTES):
        ag.collect()
        i = ag.rng.integers(0, ag._cheio, size=cfg.batch_size)
        p, _v, _r, p0 = ag._passo(
            tf.convert_to_tensor(ag._buf_obs[i]), tf.convert_to_tensor(ag._buf_mask[i]),
            tf.convert_to_tensor(ag._buf_act[i]), tf.convert_to_tensor(ag._buf_pi[i]),
            tf.convert_to_tensor(ag._buf_z[i]), tf.convert_to_tensor(ag._buf_r[i]),
            tf.convert_to_tensor(ag._buf_vivo[i]),
            cfg.coef_valor, cfg.coef_recompensa)
        p, p0 = float(p), float(p0)
        perdas.append(p)
        zeros.append(p0)
        fatias.append(p0 / max(p, 1e-9))
    return {"unroll": unroll, "normaliza_unroll": normaliza,
            "perda_pi": float(np.mean(perdas)), "perda_pi_0": float(np.mean(zeros)),
            "frac_pi_0": float(np.mean(fatias)),
            #: o que a soma crua produziria se todos os `K+1` termos fossem iguais — a
            #: referência contra a qual ler o número medido
            "frac_pi_0_uniforme": 1.0 / (unroll + 1)}


def main():
    linhas = []
    print(f"{'K':>3} {'peso':>10} {'perda_pi':>9} {'perda_pi_0':>11} "
          f"{'frac_pi_0':>10} {'1/(K+1)':>9}", flush=True)
    for normaliza in (False, True):
        for k in DESENROLARES:
            l = mede(k, normaliza)
            linhas.append(l)
            peso = "1/K" if normaliza else "soma crua"
            print(f"{k:>3} {peso:>10} {l['perda_pi']:>9.3f} {l['perda_pi_0']:>11.3f} "
                  f"{l['frac_pi_0']:>9.1%} {l['frac_pi_0_uniforme']:>8.1%}", flush=True)
    destino = os.path.join(RAIZ, "docs", "diag_unroll.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(linhas, f, indent=1, ensure_ascii=False)
    print("gravado em", destino)


if __name__ == "__main__":
    main()
