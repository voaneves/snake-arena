"""Gráfico da ablação do canal de fome — `docs/CANAL_DE_FOME.md`.

Lê os seis `history.json` (`resnet_small` e `resnet_small_fome`, três sementes cada) e grava
`assets/canal_de_fome_{light,dark}.png`. Nenhum número é digitado aqui: se uma execução
for refeita, o gráfico muda sozinho.

A paleta e as convenções de leitura são as de `snakeai.plot` — mesmo eixo único, mesmo
piso desenhado, mesma cor por identidade. Uso::

    python tools/fig_canal_de_fome.py
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, ".")
from snakeai.plot import PALETA, PISO_ALEATORIO, SCORE_PERFEITO, _formata_passos

GRUPOS = [("resnet_small_esparso", "5 canais (contrato)"),
          ("resnet_small_fome_esparso", "6 canais (com fome)")]
SEEDS = ("seed0", "seed1", "seed2")

def carrega(variante, s):
    d = json.load(open(f"runs/ppo/{variante}/{s}/history.json", encoding="utf-8"))
    ev = [(p["global_step"], p["eval_score_mean"]) for p in d["curve"] if "eval_score_mean" in p]
    return np.array([e[0] for e in ev]), np.array([e[1] for e in ev]), d["final"]

def figura(mode="light"):
    p = PALETA[mode]
    cor = {"resnet_small_esparso": p["series"][0],
           "resnet_small_fome_esparso": p["series"][1]}
    fig = plt.figure(figsize=(12.6, 5.0), facecolor=p["plane"])
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1], wspace=0.22,
                          left=0.055, right=0.965, top=0.86, bottom=0.13)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    finais = {}
    for eixo in (ax, ax2):
        eixo.set_facecolor(p["surface"])
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eixo.spines[lado].set_color(p["axis"])
        eixo.tick_params(colors=p["ink2"], labelsize=9)

    # ---- painel 1: curvas de avaliação
    for variante, rotulo in GRUPOS:
        curvas = []
        for s in SEEDS:
            x, y, f = carrega(variante, s)
            curvas.append(y)
            finais.setdefault(variante, []).append(f)
            ax.plot(x, y, color=cor[variante], lw=0.9, alpha=0.32, zorder=2)
        c = np.array(curvas)
        ax.fill_between(x, c.min(0), c.max(0), color=cor[variante], alpha=0.11, lw=0, zorder=1)
        ax.plot(x, c.mean(0), color=cor[variante], lw=2.0, zorder=3,
                solid_capstyle="round", label=rotulo)
        ax.annotate(" " + f"{c.mean(0)[-1]:.1f}".replace(".", ","),
                    (x[-1], c.mean(0)[-1]), color=p["ink"],
                    fontsize=9.5, fontweight="bold", va="center", zorder=4)

    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax.annotate("piso aleatório 1,21", (x[-1] * 0.30, PISO_ALEATORIO + 1.6),
                color=p["muted"], fontsize=8.5)
    ax.set_ylim(-2, SCORE_PERFEITO * 0.82)
    ax.set_xlim(0, x[-1] * 1.075)
    ax.grid(axis="y", color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(_formata_passos))
    ax.set_xlabel("passos de ambiente", color=p["ink2"], fontsize=9.5)
    ax.set_ylabel("score de avaliação (1.000 episódios, greedy)", color=p["ink2"], fontsize=9.5)
    ax.set_title("Média das 3 sementes · faixa = amplitude entre sementes",
                 color=p["ink2"], fontsize=10, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=p["ink"])

    # ---- painel 2: score final por semente
    for i, (variante, rotulo) in enumerate(GRUPOS):
        v = [f["score_mean"] for f in finais[variante]]
        xs = np.full(len(v), i, dtype=float)
        ax2.scatter(xs, v, s=64, color=cor[variante], zorder=3, edgecolor=p["surface"], lw=1.6)
        ax2.plot([i - 0.19, i + 0.19], [np.mean(v)] * 2, color=cor[variante], lw=2.4, zorder=4,
                 solid_capstyle="round")
        ax2.annotate(f"{np.mean(v):.1f}".replace(".", ","), (i + 0.23, np.mean(v)), color=p["ink"],
                     fontsize=9.5, fontweight="bold", va="center")
        for j, y in enumerate(v):
            ax2.annotate(f"s{j}", (i - 0.24, y), color=p["muted"], fontsize=8,
                         va="center", ha="right")
    ax2.axhline(PISO_ALEATORIO, color=p["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["5 canais", "6 canais"], color=p["ink"], fontsize=9.5)
    ax2.set_xlim(-0.55, 1.62)
    ax2.set_ylim(-2, SCORE_PERFEITO * 0.82)
    ax2.grid(axis="y", color=p["grid"], lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("score final", color=p["ink2"], fontsize=9.5)
    ax2.set_title("Barra = média · a amplitude entre sementes\né maior que a diferença entre grupos",
                  color=p["ink2"], fontsize=10, loc="left", pad=8)

    fig.suptitle("O canal de fome valeu a pena?  PPO · resnet_small · 5 M passos · 3 sementes",
                 color=p["ink"], fontsize=13, fontweight="bold", x=0.055, ha="left", y=0.965)
    return fig

for mode in ("light", "dark"):
    f = figura(mode)
    caminho = f"assets/canal_de_fome_{mode}.png"
    f.savefig(caminho, dpi=150, facecolor=f.get_facecolor())
    plt.close(f)
    print("gravado:", caminho)
