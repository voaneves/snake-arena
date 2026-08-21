"""O gráfico da arena — onde os algoritmos finalmente ficam lado a lado.

Regras de leitura que este módulo impõe, e o porquê de cada uma:

* **Um eixo só.** Score de avaliação contra passos de ambiente. Nada de segundo eixo y:
  duas escalas empilhadas inventam correlação que não existe nos dados.
* **Cor é identidade, não posição.** Cada algoritmo recebe um slot fixo da paleta, sempre
  o mesmo. Filtrar a arena não repinta os sobreviventes — quem aprendeu que "PPO é azul"
  continua certo no gráfico seguinte.
* **Mediana com faixa interquartil**, nunca uma semente só. Uma curva de RL de execução
  única não é resultado, é anedota.
* **Curvas legadas em painel próprio.** Elas vêm de `comparable=False` e são medidas em
  *episódios*, não em passos de ambiente. Plotá-las no mesmo eixo x seria fabricar um eixo
  comum que não existe — o mesmo pecado do gráfico de dois eixos y, com outra roupa. Elas
  ganham um painel ao lado, com o próprio eixo rotulado, em cinza tracejado.
* **Piso e teto sempre visíveis.** Sem o piso aleatório de 1,21 desenhado, qualquer curva
  parece aprendizado; com ele, dá para ver quem só está tendo sorte.
* **Rótulo direto no fim de cada curva**, além da legenda. Três das cores da paleta clara
  ficam abaixo de 3:1 de contraste com o fundo, e a regra é que nesse caso a identidade
  não pode depender só da cor.

A paleta é a de referência do sistema de dataviz, validada para daltonismo nos dois modos
(pior par adjacente ΔE 9,1 no claro e 8,4 no escuro).
"""

from __future__ import annotations

import numpy as np

__all__ = ["PALETA", "cores_por_algoritmo", "arena_figure", "arena_familias",
           "arena_tempo", "arena_table", "plot_run", "mesmo_hardware"]

# ---------------------------------------------------------------------- paleta
PALETA = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                   "#008300", "#4a3aa7", "#e34948"],
        "legado": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                   "#008300", "#9085e9", "#e66767"],
        "legado": "#898781",
    },
}

#: Ordem fixa dos slots. Um algoritmo novo entra no fim; ninguém troca de cor por isso.
ORDEM_ALGORITMOS = ["ppo", "dqn", "rainbow", "a2c", "acer", "alphazero",
                    "muzero", "acktr", "dreamerv3", "dqn-legacy"]

#: Famílias, na ordem em que os painéis aparecem. Existem porque a arena passou de oito
#: algoritmos e **oito é o limite honesto de uma paleta categórica**: a nona cor seria
#: indistinguível de alguma das oito sob daltonismo. A saída não é gerar mais uma cor, é
#: mudar a forma do gráfico — *small multiples*, um painel por família.
#:
#: O agrupamento é o de sempre em RL, e não uma conveniência visual: o que o algoritmo
#: aprende (política, valor, ou um modelo do mundo) é a divisão que explica por que as
#: curvas têm formatos diferentes.
FAMILIAS = [
    ("política", "gradiente de política", ["ppo", "a2c", "acktr", "acer"]),
    ("valor", "função de valor", ["dqn", "rainbow"]),
    ("modelo", "modelo do mundo e busca", ["alphazero", "muzero", "dreamerv3"]),
]


def familia_de(algo):
    for chave, _, membros in FAMILIAS:
        if algo in membros:
            return chave
    return "outros"

PISO_ALEATORIO = 1.21
SCORE_PERFEITO = 97

#: Limiar padrão da coluna "passos até". 40 é bem acima do piso (1,21) e bem abaixo do teto
#: (97): alto o bastante para exigir que o agente jogue de verdade, baixo o bastante para
#: a maioria alcançar dentro do orçamento — um limiar que quase ninguém atinge não ordena
#: nada.
LIMIAR_PADRAO = 40.0


def cores_por_algoritmo(algoritmos, mode="light"):
    """Mapeia algoritmo -> cor, em ordem fixa. Nunca cicla nem gera hue nova.

    Passar do oitavo algoritmo é um erro deliberado: a nona cor seria
    indistinguível de alguma das oito sob daltonismo. Nesse ponto o gráfico
    precisa virar *small multiples*, não ganhar mais uma cor.
    """
    p = PALETA[mode]["series"]
    conhecidos = [a for a in ORDEM_ALGORITMOS if a in algoritmos]
    novos = sorted(a for a in algoritmos if a not in ORDEM_ALGORITMOS)
    ordenados = conhecidos + novos
    if len(ordenados) > len(p):
        raise ValueError(
            f"{len(ordenados)} algoritmos para {len(p)} slots de cor. "
            "Use `arena_figure(..., familias=True)` — small multiples por família — "
            "ou agrupe a cauda em 'outros'. Não gere cor nova."
        )
    return {a: p[i] for i, a in enumerate(ordenados)}


def cores_por_familia(mode="light"):
    """Cor de cada algoritmo **dentro do painel da sua família**.

    Nos *small multiples*, cada painel é uma unidade de leitura com no máximo quatro
    séries coloridas; as outras famílias aparecem em cinza, só para dar contexto. Duas
    famílias podem repetir um matiz — o que é seguro porque elas nunca aparecem coloridas
    no mesmo painel, e cada curva colorida ganha rótulo direto.

    A cor é presa ao algoritmo pela posição dele dentro da família, que é fixa. Filtrar
    execuções não repinta ninguém.
    """
    p = PALETA[mode]["series"]
    return {a: p[i] for _, _, membros in FAMILIAS for i, a in enumerate(membros)}


# ------------------------------------------------------------------ agregação
def agrega_sementes(registros, pontos=60):
    """Junta as sementes de uma mesma `(algo, variante)` numa mediana com faixa IQR.

    As sementes raramente avaliam nos mesmos passos, então interpolamos todas numa
    grade log-espaçada comum antes de tirar os quantis. A grade para no menor
    `max(step)` entre as sementes — extrapolar seria inventar dado.
    """
    curvas = []
    for r in registros:
        x, y = r.eval_curve()
        if x.size >= 2:
            curvas.append((x, y))
    if not curvas:
        return None

    x_min = max(1, max(c[0][0] for c in curvas))
    x_max = min(c[0][-1] for c in curvas)
    if x_max <= x_min:
        return None

    grade = np.unique(np.geomspace(x_min, x_max, pontos).astype(np.int64))
    empilhado = np.stack([np.interp(grade, x, y) for x, y in curvas])
    return {
        "x": grade,
        "mediana": np.median(empilhado, axis=0),
        "q1": np.percentile(empilhado, 25, axis=0),
        "q3": np.percentile(empilhado, 75, axis=0),
        "n_sementes": len(curvas),
    }


def mesmo_hardware(registros):
    """Todas as execuções vieram da mesma máquina? Devolve `(bool, conjunto)`.

    O eixo de tempo só significa alguma coisa dentro de um mesmo hardware. Uma curva feita
    numa P100 do Kaggle e outra numa T4 do Colab colocadas lado a lado em horas comparam os
    aceleradores, não os algoritmos — e o gráfico não avisaria.
    """
    hw = {r.hardware for r in registros}
    return len(hw) <= 1, hw


def agrega_tempo(registros, pontos=60):
    """Como `agrega_sementes`, no eixo de horas de GPU."""
    curvas = []
    for r in registros:
        h, y = r.eval_curve_tempo()
        ok = np.isfinite(h) & (h > 0)
        if ok.sum() >= 2:
            curvas.append((h[ok], y[ok]))
    if not curvas:
        return None

    x_min = max(1e-4, max(c[0][0] for c in curvas))
    x_max = min(c[0][-1] for c in curvas)
    if x_max <= x_min:
        return None
    grade = np.geomspace(x_min, x_max, pontos)
    emp = np.stack([np.interp(grade, h, y) for h, y in curvas])
    return {"x": grade, "mediana": np.median(emp, axis=0),
            "q1": np.percentile(emp, 25, axis=0), "q3": np.percentile(emp, 75, axis=0),
            "n_sementes": len(curvas)}


def _agrupa(registros):
    grupos = {}
    for r in registros:
        grupos.setdefault((r.algo, r.variant), []).append(r)
    return grupos


# -------------------------------------------------------------------- figuras
def arena_familias(registros, mode="light", figsize=(14.5, 4.8), titulo=None,
                   x_log=True, mostrar_legado=True):
    """*Small multiples*: um painel por família, com as demais em cinza ao fundo.

    Esta é a forma que a arena assume quando passa de oito algoritmos. Ela não é um
    consolo por não caber tudo num painel — é melhor para a pergunta que a arena de fato
    responde. Sobrepor nove curvas com faixa interquartil produz um emaranhado onde a
    comparação relevante ("o Rainbow supera o DQN?") fica *mais* difícil, não menos.

    Cada painel mostra a família em cor e **todas as outras curvas em cinza claro**, na
    mesma escala. Sem esse fundo, três painéis lado a lado seriam três gráficos
    independentes e a comparação entre famílias se perderia — que é justamente o que a
    arena existe para permitir.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    p = PALETA[mode]
    cores = cores_por_familia(mode)
    comparaveis = [r for r in registros if r.oficial]

    agregados = {}
    for (algo, variante), rs in sorted(_agrupa(comparaveis).items()):
        ag = agrega_sementes(rs)
        if ag is not None:
            agregados[(algo, variante)] = ag

    presentes = [f for f in FAMILIAS
                 if any(a in f[2] for a, _ in agregados)] or FAMILIAS
    legado = [r for r in registros if not r.comparable] if mostrar_legado else []

    # O painel legado entra com largura menor e **eixo x próprio**: ele mede episódios, e
    # pendurá-lo no eixo de passos seria fabricar um eixo comum que não existe.
    larguras = [1.0] * len(presentes) + ([0.62] if legado else [])
    fig = plt.figure(figsize=figsize, facecolor=p["plane"])
    gs = fig.add_gridspec(1, len(larguras), width_ratios=larguras, wspace=.08)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0], sharey=axes[0])
             for i in range(1, len(presentes))]
    ax_leg = fig.add_subplot(gs[-1], sharey=axes[0]) if legado else None

    topo = max((max(ag["mediana"]) for ag in agregados.values()), default=0.0)
    topo = max(topo * 1.3, PISO_ALEATORIO * 4)

    for i, (ax, (chave, rotulo, membros)) in enumerate(zip(axes, presentes)):
        ax.set_facecolor(p["surface"])
        ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0, zorder=1)
        if i == len(presentes) - 1:
            # rotulada uma vez, e no painel mais vazio: repetir a referência nos três
            # seria ruído, e à esquerda ela cai em cima do início das curvas
            ax.annotate(f"piso aleatório · {PISO_ALEATORIO:.2f}".replace(".", ","),
                        xy=(0.98, PISO_ALEATORIO), xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points",
                        color=p["muted"], fontsize=8, va="bottom", ha="right")

        # contexto: todo o resto, em cinza, atrás
        for (algo, _), ag in agregados.items():
            if algo not in membros:
                ax.plot(ag["x"], ag["mediana"], color=p["legado"], lw=1.2,
                        alpha=.45, zorder=2, solid_capstyle="round")

        rotulos = []
        for (algo, variante), ag in agregados.items():
            if algo not in membros:
                continue
            cor = cores[algo]
            nome = algo if variante in ("default", "") else f"{algo} · {variante}"
            ax.fill_between(ag["x"], ag["q1"], ag["q3"], color=cor, alpha=.16,
                            linewidth=0, zorder=3)
            ax.plot(ag["x"], ag["mediana"], color=cor, lw=2.0, zorder=4,
                    label=f"{nome}  (n={ag['n_sementes']})", solid_capstyle="round")
            rotulos.append((ag["x"][-1], ag["mediana"][-1], nome, cor))

        for x, y, nome, _ in _sem_colisao(rotulos):
            ax.annotate(nome, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                        color=p["ink2"], fontsize=8.5, va="center", ha="left", zorder=5)

        ax.set_title(rotulo, color=p["ink2"], fontsize=10.5, loc="left", pad=10)
        if x_log:
            ax.set_xscale("log")
        ax.set_ylim(0, topo)
        if not agregados:
            ax.set_xlim(1e4, 1e7)
        else:
            # espaço à direita para o rótulo direto de cada curva não sair do painel
            ax.margins(x=.30)
        ax.grid(True, which="major", color=p["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(p["axis"])
        ax.tick_params(colors=p["muted"], labelsize=9, length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(_formata_passos))
        ax.set_xlabel("passos de ambiente", color=p["ink2"], fontsize=9.5)
        if i:
            ax.tick_params(labelleft=False)
        if rotulos:
            leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                            labelcolor=p["ink2"], handlelength=1.6)
            for t in leg.get_texts():
                t.set_color(p["ink2"])

    if ax_leg is not None:
        _painel_legado(ax_leg, legado, p, ylim=(0, topo))
        ax_leg.tick_params(labelleft=False)

    axes[0].set_ylabel("score na avaliação (1.000 episódios, greedy)",
                       color=p["ink2"], fontsize=10)
    fig.suptitle(titulo or "snake-arena · por família de algoritmo",
                 color=p["ink"], fontsize=13, x=.006, ha="left", y=.985)
    fig.text(.006, .015,
             "cada painel colore uma família e mantém as demais curvas em cinza, na mesma "
             "escala; nove algoritmos não cabem numa paleta categórica. O painel do legado "
             "tem eixo x próprio, em episódios — ver docs/COMPARABILITY.md.",
             color=p["muted"], fontsize=8)
    fig.subplots_adjust(left=.058, right=.985, top=.83, bottom=.16)
    return fig, tuple(axes) + ((ax_leg,) if ax_leg is not None else ())


def arena_figure(registros, mode="light", figsize=(12.5, 6.2), titulo=None,
                 mostrar_legado=True, x_log=True, familias="auto"):
    """A figura principal do benchmark. Devolve `(fig, (ax, ax_legado))`.

    `familias="auto"` (o padrão) troca para *small multiples* assim que o número de
    algoritmos passa dos oito slots de cor. É automático de propósito: a alternativa
    seria a arena quebrar — ou, pior, ganhar uma nona cor — no dia em que o nono
    algoritmo termina de treinar.

    `registros` é uma lista de `snakeai.record.RunRecord` — tipicamente
    `record.load_all("runs")` mais as curvas legadas convertidas.

    O painel grande tem só as execuções `comparable=True`, no eixo oficial de passos de
    ambiente. As legadas, quando existem, vão para um painel estreito à direita com o
    **próprio eixo em episódios** — porque é isso que elas medem, e fingir o contrário
    seria exatamente o erro que este repositório foi criado para consertar.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    p = PALETA[mode]
    comparaveis = [r for r in registros if r.oficial]
    legado = [r for r in registros if not r.comparable] if mostrar_legado else []

    algos = {r.algo for r in comparaveis}
    if familias is True or (familias == "auto" and len(algos) > len(p["series"])):
        return arena_familias(registros, mode=mode, titulo=titulo, x_log=x_log,
                              mostrar_legado=mostrar_legado)

    cores = cores_por_algoritmo(algos, mode)

    fig = plt.figure(figsize=figsize, facecolor=p["plane"])
    if legado:
        gs = fig.add_gridspec(1, 2, width_ratios=(3.4, 1), wspace=.22)
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_leg = None
    ax.set_facecolor(p["surface"])

    # --- referências primeiro, para ficarem atrás dos dados
    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0, zorder=1)
    ax.annotate(f"piso aleatório com máscara · {PISO_ALEATORIO:.2f}".replace(".", ","),
                xy=(0.995, PISO_ALEATORIO), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points",
                color=p["muted"], fontsize=8.5, va="bottom", ha="right")

    # --- as curvas que competem
    rotulos = []
    for (algo, variante), rs in sorted(_agrupa(comparaveis).items()):
        ag = agrega_sementes(rs)
        if ag is None:
            continue
        cor = cores[algo]
        nome = algo if variante in ("default", "") else f"{algo} · {variante}"
        ax.fill_between(ag["x"], ag["q1"], ag["q3"], color=cor, alpha=.16,
                        linewidth=0, zorder=3)
        ax.plot(ag["x"], ag["mediana"], color=cor, lw=2.0, zorder=4,
                label=f"{nome}  (n={ag['n_sementes']})", solid_capstyle="round")
        rotulos.append((ag["x"][-1], ag["mediana"][-1], nome, cor))

    # --- rótulo direto no fim de cada curva (a "relief rule" do contraste)
    for x, y, nome, cor in _sem_colisao(rotulos):
        ax.annotate(nome, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    color=p["ink2"], fontsize=9, va="center", ha="left", zorder=5)

    # --- eixos e cromo
    if x_log:
        ax.set_xscale("log")
    ax.set_xlabel("passos de ambiente", color=p["ink2"], fontsize=10)
    ax.set_ylabel("score na avaliação (1.000 episódios, greedy)",
                  color=p["ink2"], fontsize=10)
    ax.set_title(titulo or "snake-arena · mesmo ambiente, mesmo orçamento, mesma régua",
                 color=p["ink"], fontsize=13, pad=14, loc="left")

    ax.grid(True, which="major", color=p["grid"], lw=0.8, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(p["axis"])
        ax.spines[lado].set_linewidth(1.0)
    ax.tick_params(colors=p["muted"], labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_formata_passos))

    # o teto do eixo y vem de TODOS os dados, inclusive os legados: os dois painéis
    # compartilham a escala de score, e calcular só a partir das curvas oficiais faz o
    # painel da direita ser cortado quando a arena ainda está vazia
    topo_oficial = max((y for _, y, _, _ in rotulos), default=0.0)
    topo_legado = max(
        (max(c["train_score_mean"] for c in r.curve) for r in legado), default=0.0
    ) if legado else 0.0
    topo = max(topo_oficial * 1.3, topo_legado * 1.15, PISO_ALEATORIO * 4)
    ax.set_ylim(0, topo)
    if rotulos:
        ax.margins(x=.18)
    else:
        # arena vazia: um eixo x de 1 a 10 e um retângulo em branco não comunicam nada
        ax.set_xlim(1e4, 1e7)
        ax.annotate(
            "nenhuma execução oficial ainda\n\n"
            "as curvas entram aqui quando forem treinadas no orçamento do contrato",
            xy=(0.5, 0.55), xycoords="axes fraction", ha="center", va="center",
            color=p["muted"], fontsize=11, linespacing=1.6)

    if len(rotulos) >= 2:
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9,
                        labelcolor=p["ink2"], handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(p["ink2"])

    # --- painel legado: eixo próprio, unidade própria, mesma escala de score
    if ax_leg is not None:
        _painel_legado(ax_leg, legado, p, ylim=ax.get_ylim())
        fig.text(0.012, 0.015,
                 "Os dois painéis não compartilham eixo x — e não podem. À esquerda, "
                 "passos de ambiente no jogo novo; à direita, episódios no jogo de 2019, "
                 "com outra recompensa e score de treino em vez de avaliação.",
                 color=p["muted"], fontsize=8)
        fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.135)
    else:
        fig.tight_layout()
    return fig, (ax, ax_leg)


def _painel_legado(ax, legado, p, ylim=None):
    """As curvas históricas, no eixo delas: episódios de treino.

    Compartilham a escala y com o painel principal — score é score, essa parte é
    conversível. O eixo x é que não é, e por isso está separado.
    """
    ax.set_facecolor(p["surface"])
    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0, zorder=1)

    melhor = (None, -1.0)
    for r in legado:
        x = np.array([c["episodes"] for c in r.curve], dtype=np.float64)
        y = np.array([c["train_score_mean"] for c in r.curve], dtype=np.float64)
        if y.size > 400:
            # 10 mil episódios num painel estreito viram um borrão cinza; a janela
            # larga mostra a tendência, que é o que o painel de contexto precisa dizer.
            k = max(1, y.size // 40)
            nucleo = np.ones(k) / k
            y = np.convolve(y, nucleo, mode="valid")
            x = x[k - 1:]
        ax.plot(x, y, color=p["legado"], lw=1.2, ls=(0, (4, 3)), alpha=.6, zorder=2)
        if y.max() > melhor[1]:
            melhor = (r.variant, float(y.max()), float(x[int(y.argmax())]))

    if melhor[0]:
        # rótulo ancorado no canto, não no ponto: no painel estreito um rótulo junto
        # ao máximo sai pela borda direita
        ax.annotate(f"melhor: {melhor[0]}\nmédia móvel {melhor[1]:.1f}".replace(".", ","),
                    xy=(0.04, 0.97), xycoords="axes fraction",
                    color=p["ink2"], fontsize=8.5, ha="left", va="top",
                    linespacing=1.5)

    ax.set_title("legado · 2019", color=p["ink2"], fontsize=10, loc="left", pad=14)
    ax.set_xlabel("episódios de treino", color=p["muted"], fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(p["axis"])
    ax.tick_params(colors=p["muted"], labelsize=8.5, length=0)
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(_formata_passos))


def _ate_o_limiar(registros, limiar):
    """Mediana dos passos até `limiar`, e quantas sementes chegaram lá.

    Quem não chegou **não** entra na mediana como um número grande inventado: fica de fora
    e o `n` denuncia. Uma mediana calculada sobre metade das sementes que chegaram, sem
    dizer que foi metade, seria a pior das duas opções.
    """
    passos = [r.passos_ate(limiar) for r in registros]
    chegaram = [p for p in passos if p is not None]
    return {"passos_ate": int(np.median(chegaram)) if chegaram else None,
            "sementes_ate": len(chegaram), "limiar": limiar}


def _sem_colisao(rotulos, minimo=0.045):
    """Empurra rótulos que ficariam sobrepostos, preservando a ordem vertical."""
    if not rotulos:
        return []
    ordenado = sorted(rotulos, key=lambda t: t[1])
    ys = [t[1] for t in ordenado]
    faixa = max(ys[-1] - ys[0], 1e-9)
    minimo = minimo * faixa
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < minimo:
            ys[i] = ys[i - 1] + minimo
    return [(x, ys[i], nome, cor) for i, (x, _, nome, cor) in enumerate(ordenado)]


def _formata_passos(v, _pos=None):
    if v >= 1e6:
        return f"{v / 1e6:g} M"
    if v >= 1e3:
        return f"{v / 1e3:g} mil"
    return f"{v:g}"


def plot_run(record, mode="light", figsize=(11, 3.4)):
    """Diagnóstico de uma execução: treino (com exploração) contra avaliação (honesta).

    As duas subindo juntas = aprendeu. A de treino subindo sozinha = está explorando com
    sorte, e o número honesto não acompanha.
    """
    import matplotlib.pyplot as plt

    p = PALETA[mode]
    fig, ax = plt.subplots(figsize=figsize, facecolor=p["plane"])
    ax.set_facecolor(p["surface"])

    treino = [(c["global_step"], c["train_score_mean"]) for c in record.curve
              if c.get("train_score_mean") is not None]
    if treino:
        x, y = zip(*treino)
        ax.plot(x, y, color=p["muted"], lw=1.4, label="treino (com exploração)")

    x, y = record.eval_curve()
    if x.size:
        ax.plot(x, y, color=p["series"][0], lw=2.0, label="avaliação (greedy)")

    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0)
    ax.set_xlabel("passos de ambiente", color=p["ink2"], fontsize=10)
    ax.set_ylabel("score", color=p["ink2"], fontsize=10)
    ax.set_title(record.run_id, color=p["ink"], fontsize=12, loc="left", pad=10)
    ax.grid(True, color=p["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(p["axis"])
    ax.tick_params(colors=p["muted"], labelsize=9, length=0)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(p["ink2"])
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------- tabela
def arena_tempo(registros, mode="light", figsize=(7.4, 5.0), titulo=None,
                limiar=LIMIAR_PADRAO):
    """A arena no eixo de **custo**: score contra horas de GPU. Devolve `(fig, ax)`.

    O eixo oficial — passos de ambiente — iguala os *dados vistos*. É o padrão da
    literatura e é o certo para "quem aprende mais com a mesma experiência". Mas ele
    esconde uma diferença enorme: o AlphaZero roda uma busca em árvore a cada passo e custa
    ordens de grandeza mais que o DQN para chegar ao mesmo ponto no eixo x. Comparar ali dá
    a ele computação de graça.

    Este painel mostra a outra metade da verdade, e **não substitui** o oficial: são duas
    perguntas diferentes, e a resposta de uma não vale para a outra.

    Quando as execuções vêm de hardwares diferentes o gráfico **diz isso na cara**, porque
    aí ele compara aceleradores e não algoritmos — e essa é a forma mais fácil de ler um
    número errado com confiança.
    """
    import matplotlib.pyplot as plt

    p = PALETA[mode]
    comparaveis = [r for r in registros if r.oficial]

    algos = {r.algo for r in comparaveis}
    if len(algos) > len(p["series"]):
        # A mesma regra do painel principal: acima de oito, a saída é mudar a forma do
        # gráfico, não gerar cor nova. Aqui isso vira um painel de tempo por família.
        return arena_tempo_familias(registros, mode=mode, titulo=titulo)
    # `cores_por_algoritmo`, e **não** `cores_por_familia`: num painel único a cor por
    # posição-dentro-da-família repete matiz entre famílias, e três curvas azuis no mesmo
    # eixo é exatamente a ambiguidade que a paleta existe para evitar.
    cores = cores_por_algoritmo(algos, mode)

    fig, ax = plt.subplots(figsize=figsize, facecolor=p["plane"])
    ax.set_facecolor(p["surface"])
    ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0, zorder=1)

    rotulos, topo = [], 0.0
    for (algo, variante), rs in sorted(_agrupa(comparaveis).items()):
        ag = agrega_tempo(rs)
        if ag is None:
            continue
        cor = cores[algo]
        nome = algo if variante in ("default", "") else f"{algo} · {variante}"
        ax.fill_between(ag["x"], ag["q1"], ag["q3"], color=cor, alpha=.16, linewidth=0,
                        zorder=3)
        ax.plot(ag["x"], ag["mediana"], color=cor, lw=2.0, zorder=4,
                label=f"{nome}  (n={ag['n_sementes']})", solid_capstyle="round")
        rotulos.append((ag["x"][-1], ag["mediana"][-1], nome, cor))
        topo = max(topo, float(ag["mediana"].max()))

    for x, y, nome, _ in _sem_colisao(rotulos):
        ax.annotate(nome, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    color=p["ink2"], fontsize=9, va="center", ha="left", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("horas de GPU (inclui as avaliações periódicas)",
                  color=p["ink2"], fontsize=10)
    ax.set_ylabel("score na avaliação (1.000 episódios, greedy)",
                  color=p["ink2"], fontsize=10)
    ax.set_title(titulo or "snake-arena · o mesmo resultado, no eixo do custo",
                 color=p["ink"], fontsize=13, pad=14, loc="left")
    ax.set_ylim(0, max(topo * 1.3, PISO_ALEATORIO * 4))
    ax.margins(x=.22)
    ax.grid(True, which="major", color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(p["axis"])
    ax.tick_params(colors=p["muted"], labelsize=9, length=0)
    if rotulos:
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(p["ink2"])

    igual, hw = mesmo_hardware(comparaveis)
    aviso = ("mesmo hardware em todas as execuções: " + (next(iter(hw)) if hw else "—")
             if igual else
             "⚠ HARDWARES DIFERENTES (" + " · ".join(sorted(hw)) +
             "): este eixo está comparando aceleradores, não algoritmos")
    fig.text(.012, .015, aviso, color=p["muted"] if igual else p["ink"], fontsize=8.5)
    fig.subplots_adjust(left=.115, right=.97, top=.9, bottom=.145)
    return fig, ax


def arena_tempo_familias(registros, mode="light", figsize=(14.5, 4.6), titulo=None):
    """`arena_tempo` acima de oito algoritmos: um painel por família, o resto em cinza."""
    import matplotlib.pyplot as plt

    p = PALETA[mode]
    cores = cores_por_familia(mode)
    comparaveis = [r for r in registros if r.oficial]

    agregados = {}
    for chave, rs in sorted(_agrupa(comparaveis).items()):
        ag = agrega_tempo(rs)
        if ag is not None:
            agregados[chave] = ag

    presentes = [f for f in FAMILIAS if any(a in f[2] for a, _ in agregados)] or FAMILIAS
    fig, axes = plt.subplots(1, len(presentes), figsize=figsize, sharex=True, sharey=True,
                             facecolor=p["plane"])
    axes = np.atleast_1d(axes)
    topo = max((float(a["mediana"].max()) for a in agregados.values()), default=0.0)

    for i, (ax, (_, rotulo, membros)) in enumerate(zip(axes, presentes)):
        ax.set_facecolor(p["surface"])
        ax.axhline(PISO_ALEATORIO, color=p["muted"], lw=1.0, zorder=1)
        for (algo, _), ag in agregados.items():
            if algo not in membros:
                ax.plot(ag["x"], ag["mediana"], color=p["legado"], lw=1.2, alpha=.45,
                        zorder=2)
        for (algo, variante), ag in agregados.items():
            if algo not in membros:
                continue
            nome = algo if variante in ("default", "") else f"{algo} · {variante}"
            ax.plot(ag["x"], ag["mediana"], color=cores[algo], lw=2.0, zorder=4,
                    label=f"{nome}  (n={ag['n_sementes']})", solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_ylim(0, max(topo * 1.3, PISO_ALEATORIO * 4))
        ax.set_title(rotulo, color=p["ink2"], fontsize=10.5, loc="left", pad=10)
        ax.set_xlabel("horas de GPU", color=p["ink2"], fontsize=9.5)
        ax.grid(True, color=p["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=p["muted"], labelsize=9, length=0, labelleft=(i == 0))
        if any(a in membros for a, _ in agregados):
            leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
            for t in leg.get_texts():
                t.set_color(p["ink2"])

    axes[0].set_ylabel("score na avaliação", color=p["ink2"], fontsize=10)
    fig.suptitle(titulo or "snake-arena · o mesmo resultado, no eixo do custo",
                 color=p["ink"], fontsize=13, x=.006, ha="left", y=.985)
    igual, hw = mesmo_hardware(comparaveis)
    fig.text(.006, .015,
             ("mesmo hardware: " + (next(iter(hw)) if hw else "—")) if igual else
             "⚠ HARDWARES DIFERENTES (" + " · ".join(sorted(hw)) +
             "): este eixo compara aceleradores, não algoritmos",
             color=p["muted"] if igual else p["ink"], fontsize=8.5)
    fig.subplots_adjust(left=.058, right=.99, top=.83, bottom=.17)
    return fig, tuple(axes)


def arena_table(registros, markdown=True, limiar=LIMIAR_PADRAO):
    """A tabela de resultados — a visão que o gráfico não dá.

    Existe também porque três cores da paleta clara ficam abaixo de 3:1 de contraste:
    a regra manda oferecer rótulos visíveis **ou** a visão em tabela. Aqui temos as duas.
    """
    linhas = []
    for (algo, variante), rs in sorted(_agrupa([r for r in registros if r.oficial]).items()):
        finais = [r.final for r in rs if r.final]
        if not finais:
            continue
        medias = np.array([f["score_mean"] for f in finais], dtype=np.float64)
        passos = max((r.curve[-1]["global_step"] for r in rs if r.curve), default=0)
        linhas.append({
            "algo": algo,
            "variante": variante,
            "rede": rs[0].net,
            "params": rs[0].params,
            "sementes": len(rs),
            "passos": int(passos),
            "score_mean": float(np.median(medias)),
            "score_spread": float(medias.max() - medias.min()) if len(medias) > 1 else 0.0,
            "score_median": float(np.median([f.get("score_median", np.nan) for f in finais])),
            "score_max": int(max(f.get("score_max", 0) for f in finais)),
            "win_rate": float(np.median([f.get("win_rate", 0.0) for f in finais])),
            # coluna à parte, como o filtro de flood-fill e a busca do AlphaZero: o
            # melhor checkpoint responde "o melhor que este algoritmo produziu", que não
            # é a mesma pergunta que "como ele terminou"
            "melhor_mean": float(np.median(
                [r.melhor["score_mean"] for r in rs if r.melhor])) if any(
                    r.melhor for r in rs) else None,
            # A leitura HORIZONTAL da curva: em vez de "quanto marcou no fim", "quantos
            # passos precisou para chegar a `limiar`". Sai dos mesmos dados e responde à
            # outra pergunta — eficiência amostral no sentido estrito.
            **_ate_o_limiar(rs, limiar),
            "horas": float(np.median([r.meta.get("wall_s_total", np.nan) / 3600
                                      for r in rs])),
        })
    linhas.sort(key=lambda d: -d["score_mean"])

    if not markdown:
        return linhas

    out = [
        "| algoritmo | rede | params | sementes | passos | score (last) | melhor ckpt | "
        + f"passos até {limiar:.0f} | horas | amplitude | mediana/ep | máx | cheio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| _piso aleatório_ | — | — | — | 0 | **{PISO_ALEATORIO:.2f}** | — | — | — | — | 1 | — | 0% |".replace(".", ","),
    ]
    for d in linhas:
        nome = d["algo"] if d["variante"] in ("default", "") else f"{d['algo']} · {d['variante']}"
        melhor = f"{d['melhor_mean']:.2f}" if d["melhor_mean"] is not None else "—"
        if d["passos_ate"] is None:
            ate = "não chegou"
        else:
            ate = f"{d['passos_ate']:,}"
            if d["sementes_ate"] < d["sementes"]:
                ate += f" ({d['sementes_ate']}/{d['sementes']})"
        horas = f"{d['horas']:.1f}" if np.isfinite(d["horas"]) else "—"
        out.append(
            f"| {nome} | `{d['rede']}` | {d['params']:,} | {d['sementes']} | "
            f"{d['passos']:,} | **{d['score_mean']:.2f}** | {melhor} | {ate} | {horas} | "
            f"±{d['score_spread']:.2f} | "
            f"{d['score_median']:.0f} | {d['score_max']} | {d['win_rate']:.1%} |"
        )
    out.append(f"\nScore perfeito no 10×10: **{SCORE_PERFEITO}**.")
    out.append(
        f"\n**passos até {limiar:.0f}** é a curva lida na horizontal em vez da vertical: "
        "em vez de *quanto marcou no fim*, *quantos passos precisou para chegar lá*. Sai "
        "dos mesmos dados e responde à outra pergunta — menor é melhor. A resolução é a "
        "cadência de avaliação, e não há interpolação: o passo mostrado é um em que a "
        "medição de fato aconteceu. `(k/n)` significa que só `k` das `n` sementes "
        "chegaram, e as que não chegaram ficam **fora** da mediana em vez de entrar como "
        "um número inventado."
    )
    out.append(
        "\n**horas** é tempo de parede da execução inteira, útil só entre execuções do "
        "mesmo hardware. O eixo de passos iguala os *dados vistos*; ele não iguala o "
        "*esforço*, e a diferença entre os dois é enorme para quem faz busca em árvore."
    )
    out.append(
        "\nA coluna **score (last)** é o número oficial: o modelo do último passo, que é o "
        "estado final do algoritmo. O valor é a **mediana entre as sementes** do score "
        "médio de cada uma — não a média entre elas. É a mesma estatística que o gráfico "
        "desenha como linha, com o intervalo entre sementes como faixa, e com três "
        "sementes ela é o que uma semente divergente não consegue arrastar. Os documentos "
        "de ablação (`ORCAMENTO_DE_GRADIENTE.md`, `CANAL_DE_FOME.md`) reportam **média e "
        "desvio**, porque lá a pergunta é o tamanho de um efeito, não a ordem de um "
        "ranking: os dois números convivem, e cada um diz qual é. **mediana/ep** é outra "
        "coisa ainda — a mediana entre *episódios*, não entre sementes. **melhor ckpt** é "
        "o melhor que aquela execução produziu em algum momento — fica à parte porque "
        "premia quem foi medido mais vezes, pela mesma razão que a busca do AlphaZero e o "
        "filtro de flood-fill ficam fora da curva."
    )
    return "\n".join(out)
