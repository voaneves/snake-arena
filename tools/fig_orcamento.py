"""Gráfico da ablação de orçamento de gradiente — `docs/ORCAMENTO_DE_GRADIENTE.md`.

Lê os seis `history.json` (`resnet_small` e `resnet_small_denso`, três sementes cada) e
grava `assets/orcamento_{light,dark}.png`. Nenhum número é digitado aqui.

Três painéis, porque a média sozinha esconde o resultado: a curva mostra a eficiência de
amostra, os pontos por semente mostram o colapso da dispersão, e o painel de tabuleiro
cheio mostra a métrica que passa a mandar quando a mediana encosta no teto.

    python tools/fig_orcamento.py
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, ".")
from snakeai.plot import PALETA, PISO_ALEATORIO, SCORE_PERFEITO, _formata_passos

GRUPOS = [("resnet_small", "~2.400 atualizações (padrão)"),
          ("resnet_small_denso", "~38.000 atualizações (denso)")]
SEEDS = ("seed0", "seed1", "seed2")


def carrega(variante, s):
    d = json.load(open(f"runs/ppo/{variante}/{s}/history.json", encoding="utf-8"))
    ev = [(p["global_step"], p["eval_score_mean"])
          for p in d["curve"] if "eval_score_mean" in p]
    return np.array([e[0] for e in ev]), np.array([e[1] for e in ev]), d["final"]


def figura(mode="light"):
    p = PALETA[mode]
    cor = {"resnet_small": p["series"][0], "resnet_small_denso": p["series"][1]}
    fig = plt.figure(figsize=(13.6, 5.0), facecolor=p["plane"])
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.3,
                          left=0.05, right=0.97, top=0.84, bottom=0.13)
    ax, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    for eixo in (ax, ax2, ax3):
        eixo.set_facecolor(p["surface"])
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eixo.spines[lado].set_color(p["axis"])
        eixo.tick_params(colors=p["ink2"], labelsize=9)
        eixo.grid(axis="y", color=p["grid"], lw=0.8)
        eixo.set_axisbelow(True)

    finais = {}
    for variante, rotulo in GRUPOS:
        curvas, xs = [], None
        for s in SEEDS:
            x, y, f = carrega(variante, s)
            xs = x
            curvas.append(np.interp(np.linspace(0, 5e6, 200), x, y))
            finais.setdefault(variante, []).append(f)
            ax.plot(x, y, color=cor[variante], lw=0.9, alpha=0.3, zorder=2)
        malha = np.linspace(0, 5e6, 200)
        c = np.array(curvas)
        ax.fill_between(malha, c.min(0), c.max(0), color=cor[variante], alpha=0.11,
                        lw=0, zorder=1)
        ax.plot(malha, c.mean(0), color=cor[variante], lw=2.0, zorder=3,
                solid_capstyle="round", label=rotulo)
        ax.annotate(" " + f"{c.mean(0)[-1]:.1f}".replace(".", ","),
                    (malha[-1], c.mean(0)[-1]), color=p["ink"], fontsize=9.5,
                    fontweight="bold", va="center", zorder=4)

    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax.axhline(SCORE_PERFEITO, color=p["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax.annotate("jogo perfeito · 97", (1.4e6, SCORE_PERFEITO - 6), color=p["muted"],
                fontsize=8.5)
    ax.annotate("piso aleatório · 1,21", (1.4e6, PISO_ALEATORIO + 2), color=p["muted"],
                fontsize=8.5)
    ax.set_ylim(-3, SCORE_PERFEITO * 1.1)
    ax.set_xlim(0, 5.45e6)
    ax.xaxis.set_major_formatter(FuncFormatter(_formata_passos))
    ax.set_xlabel("passos de ambiente", color=p["ink2"], fontsize=9.5)
    ax.set_ylabel("score de avaliação (1.000 episódios, greedy)", color=p["ink2"],
                  fontsize=9.5)
    ax.set_title("Mesmo orçamento de ambiente · faixa = amplitude entre 3 sementes",
                 color=p["ink2"], fontsize=10, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=p["ink"])

    def pontos(eixo, valores, titulo, fmt, ymax):
        for i, (variante, _) in enumerate(GRUPOS):
            v = valores[variante]
            eixo.scatter(np.full(len(v), i, dtype=float), v, s=64, color=cor[variante],
                         zorder=3, edgecolor=p["surface"], lw=1.6)
            eixo.plot([i - 0.2, i + 0.2], [np.mean(v)] * 2, color=cor[variante], lw=2.4,
                      zorder=4, solid_capstyle="round")
            eixo.annotate(fmt(np.mean(v)), (i + 0.25, np.mean(v)), color=p["ink"],
                          fontsize=9.5, fontweight="bold", va="center")
        eixo.set_xticks([0, 1])
        eixo.set_xticklabels(["padrão", "denso"], color=p["ink"], fontsize=9.5)
        eixo.set_xlim(-0.5, 1.75)
        eixo.set_ylim(0, ymax)
        eixo.set_title(titulo, color=p["ink2"], fontsize=10, loc="left", pad=8)

    pontos(ax2, {v: [f["score_mean"] for f in finais[v]] for v, _ in GRUPOS},
           "Score final por semente", lambda x: f"{x:.1f}".replace(".", ","),
           SCORE_PERFEITO * 1.1)
    pontos(ax3, {v: [f["fim_tabuleiro_cheio"] * 100 for f in finais[v]] for v, _ in GRUPOS},
           "Tabuleiro cheio (%)", lambda x: f"{x:.0f}%", 100)

    fig.suptitle("O orçamento de gradiente, não o algoritmo  ·  PPO · resnet_small · "
                 "5 M passos · 3 sementes",
                 color=p["ink"], fontsize=13, fontweight="bold", x=0.05, ha="left",
                 y=0.965)
    return fig


for mode in ("light", "dark"):
    f = figura(mode)
    caminho = f"assets/orcamento_{mode}.png"
    f.savefig(caminho, dpi=150, facecolor=f.get_facecolor())
    plt.close(f)
    print("gravado:", caminho)
