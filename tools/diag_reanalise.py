"""Quanto custa o Reanalyse, antes de alguém gastar sete horas para descobrir.

A pergunta
----------
O Apêndice H do MuZero introduz o Reanalyse porque reúso alto de amostra precisa de alvo
fresco. Este repositório está no regime de reúso do Reanalyse — `epochs_por_iter × batch`
sobre `num_envs × rollout` dá **2,0 amostras por estado**, exatamente o número do paper —
**sem** o Reanalyse. Ligá-lo custa busca, e a conta não é a óbvia.

Por que a conta não é a óbvia
-----------------------------
A intuição diz "buscas por iteração": a coleta faz `num_envs × rollout` buscas de raiz e o
Reanalyse faz `reanalise × batch_size × epochs_por_iter`. Com os números do contrato, 1024
contra 1638 — 1,6× a coleta.

Só que **as buscas são feitas em lote**. A coleta roda `rollout` buscas batelada de largura
`num_envs` (16 × 64); o Reanalyse roda `epochs_por_iter` buscas batelada de largura
`reanalise × batch_size` (8 × 205). Em trabalho de rede o Reanalyse é 1,6× a coleta; em
**iterações do laço de árvore em Python** ele é *metade* dela. Qual dos dois domina depende
do hardware — numa GPU a rede é barata e o laço é caro, numa CPU é o contrário.

Por isso este script mede as duas coisas: os contadores, que não dependem de máquina, e o
tempo de parede, que depende.

Uso::

    python tools/diag_reanalise.py            # a forma do contrato, poucas iterações
    python tools/diag_reanalise.py --iters 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from snakeai.agents import MuZero, MuZeroConfig      # noqa: E402

#: As frações medidas. 0,8 é a do paper; 0,25 é o ponto em que o custo ainda cabe num
#: orçamento de uma noite.
FRACOES = (0.0, 0.25, 0.5, 0.8)


def contadores(cfg, fracao):
    """O que não depende de máquina: buscas de raiz e buscas batelada por iteração."""
    coleta_raizes = cfg.num_envs * cfg.rollout
    rea_raizes = int(round(fracao * cfg.batch_size)) * cfg.epochs_por_iter
    return {
        "raizes_coleta": coleta_raizes,
        "raizes_reanalise": rea_raizes,
        "razao_raizes": rea_raizes / coleta_raizes,
        # cada busca batelada é um laço de árvore de `num_simulations` passos
        "lotes_coleta": cfg.rollout,
        "lotes_reanalise": cfg.epochs_por_iter if rea_raizes else 0,
        "razao_lotes": (cfg.epochs_por_iter / cfg.rollout) if rea_raizes else 0.0,
    }


def mede(fracao, iters, **kw):
    base = dict(net="resnet_small", num_envs=64, rollout=16, unroll=5,
                num_simulations=24, batch_size=256, memory_size=50_000,
                epochs_por_iter=8, total_steps=10**9, eval_every_steps=10**9,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False,
                reanalise=fracao)
    base.update(kw)
    cfg = MuZeroConfig(**base)
    ag = MuZero(cfg)
    ag.collect()                                  # enche o buffer e aquece os traços
    ag._aprender()
    t_col, t_apr = [], []
    for _ in range(iters):
        t0 = time.time()
        ag.collect()
        t1 = time.time()
        st = ag._aprender()
        t_apr.append(time.time() - t1)
        t_col.append(t1 - t0)
    return {"reanalise": fracao, "s_coleta": float(np.median(t_col)),
            "s_treino": float(np.median(t_apr)),
            "s_iter": float(np.median(t_col) + np.median(t_apr)),
            "reanalises": int(st.get("reanalises", 0)),
            **contadores(cfg, fracao)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args(argv)

    linhas = []
    print(f"{'reanalise':>10} {'raizes':>8} {'x coleta':>9} {'lotes':>6} {'x coleta':>9} "
          f"{'s/coleta':>9} {'s/treino':>9} {'s/iter':>8} {'x base':>7}", flush=True)
    base = None
    for f in FRACOES:
        l = mede(f, a.iters)
        linhas.append(l)
        base = base or l["s_iter"]
        print(f"{f:>10.2f} {l['raizes_reanalise']:>8} {l['razao_raizes']:>8.2f}x "
              f"{l['lotes_reanalise']:>6} {l['razao_lotes']:>8.2f}x "
              f"{l['s_coleta']:>9.1f} {l['s_treino']:>9.1f} {l['s_iter']:>8.1f} "
              f"{l['s_iter'] / base:>6.2f}x", flush=True)

    destino = os.path.join(RAIZ, "docs", "diag_reanalise.json")
    with open(destino, "w", encoding="utf-8") as fp:
        json.dump({"maquina": os.cpu_count(), "linhas": linhas}, fp, indent=1,
                  ensure_ascii=False)
    print("\ngravado em", destino)
    print("Os contadores nao dependem de maquina; os tempos sim. Numa GPU a rede e barata "
          "e o laco de arvore em Python e caro, entao a coluna que manda la e `x coleta` "
          "dos LOTES; numa CPU manda a das RAIZES.")


if __name__ == "__main__":
    main()
