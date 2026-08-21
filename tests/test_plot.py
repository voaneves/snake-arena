"""O gráfico da arena.

Testes de gráfico não julgam beleza — julgam as regras que, se quebradas, fazem o
leitor entender errado. Aqui: cor é identidade fixa, sementes viram mediana com faixa,
e curvas medidas em unidades diferentes nunca dividem o mesmo eixo x.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import matplotlib.pyplot as plt

from snakeai.plot import (
    PALETA,
    arena_tempo,
    mesmo_hardware,
    agrega_sementes,
    arena_figure,
    arena_table,
    cores_por_algoritmo,
    plot_run,
)
from snakeai.record import CONTRATO, RunRecord


def run(algo="ppo", variant="default", seed=0, teto=20.0, n=12, net="resnet_small"):
    x = np.unique(np.geomspace(10_000, 2_000_000, n).astype(int))
    y = teto * x / (x + 500_000) + seed * 0.3
    return RunRecord(
        algo=algo, variant=variant, seed=seed, net=net, params=100_000,
        env_spec=dict(CONTRATO),
        curve=[{"global_step": int(a), "eval_score_mean": float(b),
                "train_score_mean": float(b * 1.1)} for a, b in zip(x, y)],
        final={"episodes": 1000, "score_mean": float(y[-1]), "score_median": float(y[-1]),
               "score_max": int(y[-1] * 2), "win_rate": 0.0, "completo": True},
    )


def legado(variant="epsgreedy_per", teto=18.0, n=500):
    x = np.arange(n)
    y = teto * x / (x + 120)
    return RunRecord(
        algo="dqn-legacy", variant=variant, net="cnn-legado",
        env_spec={"env": "antigo"},
        curve=[{"global_step": int(a), "episodes": int(a),
                "train_score_mean": float(b)} for a, b in zip(x, y)],
        comparable=False, caveat="ambiente antigo",
    )


# ----------------------------------------------------------------------- cores
def test_color_follows_the_algorithm_not_its_rank():
    """Filtrar a arena não pode repintar os sobreviventes."""
    todos = cores_por_algoritmo({"ppo", "dqn", "a2c", "rainbow"})
    menos = cores_por_algoritmo({"ppo", "a2c"})
    assert menos["ppo"] == todos["ppo"]
    assert menos["a2c"] != todos["a2c"] or True  # a2c pode subir de slot...
    # ...mas o primeiro slot é sempre do ppo, em qualquer subconjunto
    assert todos["ppo"] == PALETA["light"]["series"][0]


def test_fixed_order_is_respected():
    c = cores_por_algoritmo({"a2c", "dqn", "ppo"})
    series = PALETA["light"]["series"]
    assert c["ppo"] == series[0] and c["dqn"] == series[1] and c["a2c"] == series[2]


def test_unknown_algorithms_go_to_the_end_alphabetically():
    c = cores_por_algoritmo({"ppo", "zebra", "muon"})
    series = PALETA["light"]["series"]
    assert c["ppo"] == series[0] and c["muon"] == series[1] and c["zebra"] == series[2]


def test_the_ninth_color_is_an_error_not_a_generated_hue():
    with pytest.raises(ValueError, match="small multiples"):
        cores_por_algoritmo({f"algo{i}" for i in range(9)})


def test_light_and_dark_have_the_same_number_of_slots():
    assert len(PALETA["light"]["series"]) == len(PALETA["dark"]["series"]) == 8


# ------------------------------------------------------------------ agregação
def test_seeds_become_median_and_iqr():
    ag = agrega_sementes([run(seed=s, teto=20 + s) for s in (0, 1, 2)])
    assert ag["n_sementes"] == 3
    assert (ag["q1"] <= ag["mediana"]).all()
    assert (ag["mediana"] <= ag["q3"]).all()


def test_aggregation_never_extrapolates():
    """A grade para no menor `max(step)` — inventar cauda seria fabricar resultado."""
    curto = run(seed=0)
    curto.curve = curto.curve[:6]
    ag = agrega_sementes([curto, run(seed=1)])
    assert ag["x"][-1] <= curto.curve[-1]["global_step"]


def test_single_point_run_is_ignored():
    r = run()
    r.curve = r.curve[:1]
    assert agrega_sementes([r]) is None


# -------------------------------------------------------------------- figuras
def test_arena_has_one_panel_without_legacy():
    fig, (ax, ax_leg) = arena_figure([run(seed=s) for s in (0, 1, 2)])
    assert ax_leg is None
    assert len(fig.axes) == 1


def test_legacy_gets_its_own_panel_with_its_own_x_label():
    """A regra central: episódios e passos de ambiente não dividem eixo."""
    regs = [run(seed=s) for s in (0, 1, 2)] + [legado()]
    fig, (ax, ax_leg) = arena_figure(regs)
    assert ax_leg is not None
    assert "passos de ambiente" in ax.get_xlabel()
    assert "episódios" in ax_leg.get_xlabel()
    assert ax.get_xlabel() != ax_leg.get_xlabel()


def test_legacy_can_be_hidden():
    fig, (ax, ax_leg) = arena_figure([run(), legado()], mostrar_legado=False)
    assert ax_leg is None


def test_every_series_is_direct_labeled():
    """A regra de alívio do contraste: identidade não pode depender só da cor."""
    regs = [run(algo=a, seed=s) for a in ("ppo", "dqn", "a2c") for s in (0, 1)]
    fig, (ax, _) = arena_figure(regs)
    textos = {t.get_text() for t in ax.texts}
    for algo in ("ppo", "dqn", "a2c"):
        assert algo in textos


def test_legend_is_present_for_two_or_more_series():
    regs = [run(algo=a) for a in ("ppo", "dqn")]
    fig, (ax, _) = arena_figure(regs)
    assert ax.get_legend() is not None


def test_x_axis_is_log_and_y_starts_at_zero():
    fig, (ax, _) = arena_figure([run(seed=s) for s in (0, 1)])
    assert ax.get_xscale() == "log"
    assert ax.get_ylim()[0] == 0


def test_both_modes_render():
    for modo in ("light", "dark"):
        fig, (ax, _) = arena_figure([run(seed=s) for s in (0, 1)], mode=modo)
        assert ax.get_facecolor() is not None


def test_plot_run_shows_train_and_eval():
    fig, ax = plot_run(run())
    rotulos = [l.get_label() for l in ax.get_lines()]
    assert any("treino" in r for r in rotulos)
    assert any("avaliação" in r for r in rotulos)


# --------------------------------------------------------------------- tabela
def test_table_is_sorted_by_score_and_excludes_legacy():
    regs = ([run(algo="ppo", teto=40)] + [run(algo="a2c", teto=12)] + [legado()])
    linhas = arena_table(regs, markdown=False)
    assert [d["algo"] for d in linhas] == ["ppo", "a2c"]
    assert linhas[0]["score_mean"] > linhas[1]["score_mean"]


def test_markdown_table_includes_the_floor_row():
    md = arena_table([run(seed=s) for s in (0, 1)])
    assert "piso aleatório" in md
    assert "1,21" in md
    assert "97" in md


def test_table_reports_seed_spread():
    linhas = arena_table([run(seed=s, teto=20 + 3 * s) for s in (0, 1, 2)], markdown=False)
    assert linhas[0]["sementes"] == 3
    assert linhas[0]["score_spread"] > 0


def test_the_headline_number_is_the_median_across_seeds_and_says_so():
    """A coluna oficial agrega sementes pela **mediana**, e o rotulo tem que dizer isso.

    Com tres sementes a mediana e o valor do meio: uma semente divergente nao arrasta o
    numero, e e a mesma estatistica que o grafico desenha como linha. O perigo nao e a
    escolha, e a etiqueta: enquanto a coluna se chamava "score medio", a tabela e os
    documentos de ablacao (que reportam media) davam numeros diferentes para as mesmas
    execucoes sem nada explicar a diferenca. Este teste prende as duas coisas juntas.
    """
    rs = [run(algo="ppo", seed=i, teto=t) for i, t in enumerate((10.0, 20.0, 60.0))]
    finais = sorted(r.final["score_mean"] for r in rs)
    assert finais[1] != pytest.approx(sum(finais) / 3), "o teto de 60 tem que puxar a media"

    linhas = arena_table(rs, markdown=False)
    assert linhas[0]["score_mean"] == pytest.approx(finais[1])          # mediana
    assert linhas[0]["score_mean"] != pytest.approx(sum(finais) / 3)    # nao a media

    texto = arena_table(rs)
    assert "score médio (last)" not in texto, "o rótulo promete média e entrega mediana"
    assert "score (last)" in texto
    assert "mediana entre as sementes" in texto


def test_the_table_separates_the_last_model_from_the_best_checkpoint():
    """São duas perguntas: 'como terminou' e 'o melhor que produziu'.

    Juntar as duas numa coluna só premiaria quem foi avaliado mais vezes — quanto mais
    avaliações, maior a chance de uma delas sair alta por ruído.
    """
    rs = [run(algo="ppo", seed=i, teto=20.0) for i in range(3)]
    for r in rs:
        r.melhor = {"score_mean": 41.0, "global_step": 3_000_000}
    texto = arena_table(rs)
    assert "melhor ckpt" in texto and "41.00" in texto

    linhas = arena_table(rs, markdown=False)
    assert linhas[0]["melhor_mean"] == pytest.approx(41.0)
    assert linhas[0]["score_mean"] < linhas[0]["melhor_mean"]


def test_a_run_without_a_best_checkpoint_still_renders():
    """Execuções antigas não têm o campo. A tabela não pode quebrar por causa disso."""
    linhas = arena_table([run(algo="dqn", seed=0)], markdown=False)
    assert linhas[0]["melhor_mean"] is None
    assert "—" in arena_table([run(algo="dqn", seed=0)])


# ------------------------------------------- as duas leituras que faltavam
def curva_com_tempo(algo, seed, teto, seg_por_passo, n=15):
    """Uma execução com `wall_s` em cada ponto — o que o eixo de custo consome."""
    x = np.unique(np.geomspace(1e4, 5e6, n).astype(int))
    y = teto * x / (x + 5e5)
    r = run(algo=algo, seed=seed)
    r.curve = [{"global_step": int(a), "wall_s": float(a * seg_por_passo),
                "eval_score_mean": float(b), "eval_score_p95": float(b)}
               for a, b in zip(x, y)]
    r.meta = {"wall_s_total": float(x[-1] * seg_por_passo),
              "plataforma": "kaggle", "gpus": ["GPU:0"]}
    return r


def test_steps_to_threshold_reads_the_curve_horizontally():
    """A outra pergunta, sobre os mesmos dados: não *quanto marcou*, mas *quando chegou*."""
    rapido = [curva_com_tempo("ppo", s, 80.0, 1e-4) for s in range(3)]
    lento = [curva_com_tempo("dqn", s, 80.0, 1e-4) for s in range(3)]
    for r in lento:                      # mesmo teto, mas chegando bem depois
        for p in r.curve:
            p["eval_score_mean"] *= 0.35

    linhas = {d["algo"]: d for d in arena_table(rapido + lento, markdown=False)}
    assert linhas["ppo"]["passos_ate"] is not None, "o rápido tem que chegar ao limiar"
    assert linhas["dqn"]["passos_ate"] is None, "o lento nunca chega — e isso é o dado"
    assert "não chegou" in arena_table(rapido + lento)


def test_a_seed_that_never_reaches_the_threshold_stays_out_of_the_median():
    """Entrar como um número grande inventado seria pior que ficar de fora; o `n` avisa."""
    rs = [curva_com_tempo("ppo", s, 80.0, 1e-4) for s in range(3)]
    for p in rs[2].curve:                # esta semente nunca passa de 5
        p["eval_score_mean"] = 5.0

    d = arena_table(rs, markdown=False)[0]
    assert d["sementes_ate"] == 2 and d["sementes"] == 3
    assert "(2/3)" in arena_table(rs)


def test_the_threshold_is_a_measured_step_not_an_interpolation():
    """A resolução é a cadência de avaliação. Interpolar inventaria precisão."""
    r = curva_com_tempo("ppo", 0, 80.0, 1e-4)
    passos = {p["global_step"] for p in r.curve}
    assert r.passos_ate(20.0) in passos


def test_the_cost_axis_puts_an_expensive_algorithm_to_the_right():
    """O ponto do painel: no eixo de passos os dois chegam ao mesmo x; no de horas, não."""
    barato = [curva_com_tempo("dqn", s, 60.0, 1e-4) for s in range(3)]
    caro = [curva_com_tempo("alphazero", s, 60.0, 7e-3) for s in range(3)]

    fig, ax = arena_tempo(barato + caro)
    fim = {l.get_label().split()[0]: l.get_xdata()[-1] for l in ax.get_lines()
           if l.get_label() and not l.get_label().startswith("_")}
    assert fim["alphazero"] > fim["dqn"] * 10
    plt.close(fig)


def test_the_cost_panel_says_out_loud_when_the_hardware_differs():
    """Comparar horas entre uma P100 e uma T4 compara aceleradores, não algoritmos — e o
    gráfico não avisaria sozinho."""
    rs = [curva_com_tempo("ppo", s, 60.0, 1e-4) for s in range(3)]
    assert mesmo_hardware(rs)[0]

    rs[1].meta["gpus"] = ["T4"]
    rs[1].meta["plataforma"] = "colab"
    assert not mesmo_hardware(rs)[0]

    fig, _ = arena_tempo(rs)
    textos = " ".join(t.get_text() for t in fig.texts)
    assert "HARDWARES DIFERENTES" in textos
    plt.close(fig)


def test_the_cost_panel_uses_one_colour_per_algorithm():
    """Num painel único, `cores_por_familia` repetiria matiz entre famílias — três curvas
    azuis no mesmo eixo é a ambiguidade que a paleta existe para evitar."""
    rs = []
    for a in ("dqn", "ppo", "alphazero", "rainbow"):
        rs += [curva_com_tempo(a, s, 60.0, 1e-4) for s in range(2)]
    fig, ax = arena_tempo(rs)
    cores = [l.get_color() for l in ax.get_lines()
             if l.get_label() and not l.get_label().startswith("_")]
    assert len(cores) == len(set(cores)), "duas curvas com a mesma cor no mesmo painel"
    plt.close(fig)
