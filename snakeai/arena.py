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

from .plot import (arena_figure, arena_table, arena_tempo, arena_vitorias,
                   mesmo_hardware, separa_principais)
from .record import (ORCAMENTO_OFICIAL, SEMENTES_OFICIAIS, configuracoes_incompletas,
                     from_legacy_csv, load_all, validate)

__all__ = ["montar", "main"]


def carregar(runs="runs", legado="results/legacy"):
    """Todas as execuções + as curvas históricas. Devolve `(oficiais, fora, historicas)`.

    O contrato é conferido **aqui e agora**, com esta versão do código, e não pelo carimbo
    que ficou em `meta["contract_violations"]`. A diferença não é acadêmica: aquele carimbo
    foi escrito por quem treinou, com o pacote daquele dia, e uma execução de 12/08 —
    medida antes da correção do protocolo de avaliação — continuou entrando na arena como
    oficial simplesmente porque a validação da época não sabia perguntar. Revalidar é o
    que faz uma regra nova alcançar as execuções antigas.

    Execuções `comparable=False` não somem: elas não competem, mas aparecem na lista
    `fora` com o `caveat` como motivo. Excluir em silêncio é o que o `COMPARABILITY.md`
    chama de pior que incluir.
    """
    registros = load_all(runs) if os.path.isdir(runs) else []
    for r in registros:
        problemas = validate(r) if r.comparable else [f"comparable=False: {r.caveat}"]
        if problemas:
            r.meta["contract_violations"] = problemas
        else:
            r.meta.pop("contract_violations", None)

    oficiais = [r for r in registros if r.comparable and not r.meta.get("contract_violations")]
    fora = [r for r in registros if r not in oficiais]

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

    # Sementes é propriedade do conjunto, não de uma execução: `validate` não tem como
    # ver. Sem este aviso, uma linha de uma semente é publicada com a mesma tipografia de
    # uma de três — e a amplitude entre sementes do PPO é de 19 pontos.
    incompletas = configuracoes_incompletas(oficiais)
    if verbose and incompletas:
        print(f"  [atenção] configurações com menos de {SEMENTES_OFICIAIS} sementes: "
              + "; ".join(f"{c['algo']}/{c['variant']} ({c['sementes']})"
                          for c in incompletas))

    # O gráfico mostra um braço por algoritmo; a tabela mostra tudo. A lista existe pelo
    # mesmo motivo da lista de `[fora da arena]`: uma execução que some da figura sem
    # aparecer em lugar nenhum é uma afirmação de que ela não existe.
    _, ablacoes = separa_principais(oficiais)
    if verbose and ablacoes:
        print("  [fora do gráfico] ablações — estão na tabela, com o controle de cada "
              "uma: " + "; ".join(sorted({f"{r.algo}/{r.variant}" for r in ablacoes})))

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

    # O painel de vitórias. Terceira pergunta, terceiro painel: média e taxa de vitória
    # são funcionais diferentes da mesma distribuição e **discordam** nestes dados.
    for modo in ("light", "dark"):
        try:
            fig, _ = arena_vitorias(oficiais, mode=modo)
            caminho = os.path.join(assets, f"arena_vitorias_{modo}.png")
            fig.savefig(caminho, dpi=165, facecolor=fig.get_facecolor())
            plt.close(fig)
            saidas[f"vitorias_{modo}"] = caminho
        except Exception as e:                    # nunca derrubar a arena pelo secundário
            saidas[f"vitorias_{modo}_erro"] = repr(e)

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
        "O gráfico mostra o **braço principal** de cada algoritmo — o que o notebook roda na",
        "configuração padrão. A tabela abaixo mostra **tudo**, ablações inclusive: a figura",
        "responde *quem vai mais longe com os mesmos dados*, e uma ablação desenhada ao lado",
        "do próprio controle, na mesma cor, responde outra pergunta.",
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
        "## E no eixo de quem fecha o tabuleiro",
        "",
        "A média e a taxa de vitória são dois funcionais da **mesma** distribuição —",
        "`E[X]` e `P(X = 97)` — e não têm obrigação de concordar. O limiar joga fora tudo",
        "abaixo do teto: um episódio de 96 conta igual a um de 3. A média joga fora o",
        "formato: não distingue \"sempre 78\" de \"metade perfeito, metade zero\". **Quando",
        "a ordem desta figura difere da ordem do gráfico oficial, é exatamente isso que",
        "está acontecendo** — e nenhuma das duas é \"a qualidade do modelo\".",
        "",
        "Por isso a barra não é a taxa de vitória sozinha: é a repartição inteira das",
        "causas de fim, com a vitória como primeiro segmento. Ela mostra o que nenhum dos",
        "dois números mostra — perder por fome e perder por colisão são fracassos",
        "diferentes, e a curva de score é idêntica nos dois casos.",
        "",
        "![quem fecha o tabuleiro](../assets/arena_vitorias_light.png)",
        "",
    ]
    if incompletas:
        linhas += [
            f"## Configurações com menos de {SEMENTES_OFICIAIS} sementes",
            "",
            "Entram no gráfico, mas **não sustentam comparação**: a amplitude entre",
            "sementes do PPO neste ambiente é de 19 pontos, maior que quase toda",
            "diferença entre algoritmos que a tabela mostra.",
            "",
        ]
        linhas += [f"- `{c['algo']}/{c['variant']}`: {c['sementes']} de "
                   f"{SEMENTES_OFICIAIS} — faltam {c['faltam']}" for c in incompletas]
        linhas.append("")

    if fora:
        linhas += [
            "## Execuções que não entraram na arena",
            "",
            "Estão registradas em `runs/`, com curva e artefatos, mas não competem. O",
            "motivo é conferido na hora de montar a arena, com esta versão do código —",
            "não é o carimbo que ficou gravado no dia do treino. Execuções marcadas",
            "`comparable=False` também aparecem aqui: elas não competem por construção,",
            "e some-las seria pior do que incluí-las.",
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
