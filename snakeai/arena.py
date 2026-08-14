"""A arena — junta todas as execuções e produz o gráfico e a tabela.

Um comando regenera tudo que o README mostra::

    python -m snakeai.arena --all

Ele lê todo `runs/**/history.json`, converte as curvas históricas de `results/legacy/`, e
grava `assets/arena_light.png`, `assets/arena_dark.png` e `docs/RESULTADOS.md`.

O portão está aqui, não na hora de escrever o registro: execuções com
`meta["contract_violations"]` são **listadas e excluídas** do gráfico. Excluir em silêncio
seria pior que incluir — quem olha a arena precisa saber que existe uma execução que não
entrou, e por quê.
"""

from __future__ import annotations

import argparse
import glob
import os

from .plot import arena_figure, arena_table, arena_tempo, mesmo_hardware
from .record import ORCAMENTO_OFICIAL, from_legacy_csv, load_all

__all__ = ["montar", "main"]


def carregar(runs="runs", legado="results/legacy"):
    """Todas as execuções + as curvas históricas. Devolve `(oficiais, fora, historicas)`."""
    registros = load_all(runs) if os.path.isdir(runs) else []
    oficiais = [r for r in registros if r.oficial]
    fora = [r for r in registros if not r.oficial and r.comparable]

    historicas = []
    if os.path.isdir(legado):
        for caminho in sorted(glob.glob(os.path.join(legado, "*.csv"))):
            try:
                historicas.append(from_legacy_csv(caminho))
            except ValueError:
                pass
    return oficiais, fora, historicas


def montar(runs="runs", legado="results/legacy", assets="assets", docs="docs",
           verbose=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oficiais, fora, historicas = carregar(runs, legado)

    # segunda linha de defesa: mesmo validando uma a uma, o conjunto tem que ser coerente
    orcamentos = {int(r.config.get("total_steps", 0)) for r in oficiais}
    if len(orcamentos) > 1:
        raise ValueError(
            f"as execuções oficiais têm orçamentos diferentes ({sorted(orcamentos)}). "
            "O gráfico seria uma comparação entre treinos de tamanhos distintos."
        )

    if verbose:
        print(f"{len(oficiais)} execuções oficiais, {len(historicas)} curvas históricas")
        if oficiais:
            print(f"orçamento: {orcamentos.pop():,} passos "
                  f"(oficial: {ORCAMENTO_OFICIAL:,})")
        for r in fora:
            print(f"  [fora da arena] {r.run_id}: "
                  + "; ".join(r.meta.get("contract_violations", [])))

    os.makedirs(assets, exist_ok=True)
    os.makedirs(docs, exist_ok=True)
    saidas = {}

    for modo in ("light", "dark"):
        fig, _ = arena_figure(oficiais + historicas, mode=modo)
        caminho = os.path.join(assets, f"arena_{modo}.png")
        fig.savefig(caminho, dpi=165, facecolor=fig.get_facecolor())
        plt.close(fig)
        saidas[modo] = caminho

    # O painel de custo. Ele não substitui o oficial: o eixo de passos iguala os dados
    # vistos, este iguala o esforço, e as duas perguntas têm respostas diferentes.
    for modo in ("light", "dark"):
        try:
            fig, _ = arena_tempo(oficiais, mode=modo)
            caminho = os.path.join(assets, f"arena_tempo_{modo}.png")
            fig.savefig(caminho, dpi=165, facecolor=fig.get_facecolor())
            plt.close(fig)
            saidas[f"tempo_{modo}"] = caminho
        except Exception as e:                    # nunca derrubar a arena pelo secundário
            saidas[f"tempo_{modo}_erro"] = repr(e)

    igual, hw = mesmo_hardware(oficiais)
    if verbose and oficiais and not igual:
        print("  [atenção] execuções de hardwares diferentes (" + " · ".join(sorted(hw))
              + "): o painel de tempo compara aceleradores, não algoritmos")

    tabela = arena_table(oficiais + historicas)
    linhas = [
        "# Resultados",
        "",
        "Gerado por `python -m snakeai.arena --all`. Não editar à mão.",
        "",
        "![arena](../assets/arena_light.png)",
        "",
        tabela,
        "",
        "## O mesmo resultado, no eixo do custo",
        "",
        "O gráfico acima iguala os **dados vistos**. Este iguala o **esforço gasto** — e a",
        "ordem muda, porque um passo de AlphaZero custa uma busca em árvore inteira e um",
        "de DQN custa uma passada de rede. São duas perguntas diferentes, e nenhuma das",
        "duas é a resposta da outra.",
        "",
        "![arena por tempo](../assets/arena_tempo_light.png)",
        "",
    ]
    if fora:
        linhas += [
            "## Execuções que não entraram na arena",
            "",
            "Estão registradas em `runs/`, com curva e artefatos, mas não competem — o",
            "motivo está em `meta[\"contract_violations\"]` de cada uma.",
            "",
        ]
        linhas += [f"- `{r.run_id}`: " + "; ".join(r.meta.get("contract_violations", []))
                   for r in fora]
        linhas.append("")

    caminho_md = os.path.join(docs, "RESULTADOS.md")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    saidas["tabela"] = caminho_md

    if verbose:
        for k, v in saidas.items():
            print(f"  {k}: {v}")
    return saidas


def main(argv=None):
    p = argparse.ArgumentParser(description="Monta o gráfico e a tabela da arena.")
    p.add_argument("--all", action="store_true", help="regenera tudo")
    p.add_argument("--runs", default="runs")
    p.add_argument("--legado", default="results/legacy")
    p.add_argument("--assets", default="assets")
    p.add_argument("--docs", default="docs")
    args = p.parse_args(argv)
    montar(args.runs, args.legado, args.assets, args.docs)


if __name__ == "__main__":
    main()
