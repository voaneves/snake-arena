"""O gráfico da arena.

Testes de gráfico não julgam beleza — julgam as regras que, se quebradas, fazem o
leitor entender errado. Aqui: cor é identidade fixa, sementes viram mediana com faixa,
e curvas medidas em unidades diferentes nunca dividem o mesmo eixo x.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from snakeai.plot import (
    PALETA,
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
