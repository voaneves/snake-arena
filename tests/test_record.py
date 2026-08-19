"""O porteiro do benchmark.

Se estes testes passam, é impossível um resultado fora do contrato virar uma linha no
gráfico da arena sem que alguém tenha desligado a validação de propósito.
"""

import json
import os

import numpy as np
import pytest

from snakeai.record import (
    CONTRATO,
    ORCAMENTO_OFICIAL,
    SCHEMA_VERSION,
    ContractViolation,
    Recorder,
    RunRecord,
    assert_valid,
    from_legacy_csv,
    load,
    load_all,
    save,
    validate,
)


def registro_valido(**kw):
    base = dict(
        algo="ppo",
        variant="resnet_small",
        seed=0,
        net="resnet_small",
        params=135_000,
        # a curva vai até o orçamento porque `validate` confere o que foi **gasto**, e
        # não o que o `config` declara — ver `docs/REVISAO_ALGORITMOS.md` §1.3
        curve=[
            {"global_step": 0, "train_score_mean": 1.0},
            {"global_step": 50_000, "train_score_mean": 4.0, "eval_score_mean": 3.5},
            {"global_step": ORCAMENTO_OFICIAL, "train_score_mean": 9.0,
             "eval_score_mean": 8.1},
        ],
        final={"episodes": 1000, "score_mean": 8.1, "completo": True},
        config={"total_steps": ORCAMENTO_OFICIAL},
    )
    base.update(kw)
    return RunRecord(**base)


# ------------------------------------------------------------------- validação
def test_a_valid_record_passes():
    assert validate(registro_valido()) == []


def test_env_spec_must_match_the_contract():
    r = registro_valido()
    r.env_spec["board_size"] = 20
    problemas = validate(r)
    assert any("board_size" in p for p in problemas)


def test_missing_env_spec_key_is_caught():
    r = registro_valido()
    del r.env_spec["starve_base"]
    assert any("starve_base" in p for p in validate(r))


def test_metric_must_be_score_not_length():
    """O erro original do repositório antigo, agora impossível de repetir em silêncio."""
    r = registro_valido()
    r.env_spec["metric"] = "length"
    assert any("metric" in p for p in validate(r))


def test_eval_must_use_the_official_episode_count():
    r = registro_valido(final={"episodes": 50, "score_mean": 8.1, "completo": True})
    assert any("episódios" in p for p in validate(r))


def test_incomplete_eval_is_rejected():
    r = registro_valido(final={"episodes": 1000, "score_mean": 8.1, "completo": False})
    assert any("incompleta" in p for p in validate(r))


def test_impossible_score_is_rejected():
    r = registro_valido(final={"episodes": 1000, "score_mean": 120.0, "completo": True})
    assert any("faixa" in p for p in validate(r))


def test_empty_curve_is_rejected():
    assert any("curva vazia" in p for p in validate(registro_valido(curve=[])))


def test_non_monotonic_steps_are_rejected():
    r = registro_valido(curve=[{"global_step": 10}, {"global_step": 5}])
    assert any("monotônico" in p for p in validate(r))


def test_net_and_params_are_required():
    assert any("net" in p for p in validate(registro_valido(net="")))
    assert any("params" in p for p in validate(registro_valido(params=0)))


def test_assert_valid_raises_with_a_useful_message():
    r = registro_valido(net="")
    with pytest.raises(ContractViolation) as e:
        assert_valid(r)
    assert "ppo/resnet_small/seed0" in str(e.value)
    assert "net" in str(e.value)


# ------------------------------------------------------- curvas não comparáveis
def test_non_comparable_records_skip_the_contract_but_need_a_caveat():
    r = registro_valido(comparable=False, caveat="", env_spec={"env": "outro"},
                        final={}, net="", params=0)
    assert any("caveat" in p for p in validate(r))

    r.caveat = "medido no ambiente antigo"
    assert validate(r) == []


# --------------------------------------------------------------------- gravação
def test_recorder_roundtrip(tmp_path):
    rec = Recorder("ppo", variant="resnet_small", seed=1, net="resnet_small",
                   params=135_000,
                   config={"lr": 3e-4, "total_steps": ORCAMENTO_OFICIAL},
                   root=str(tmp_path))
    rec.log(0, train_score_mean=1.0)
    rec.log(50_000, train_score_mean=np.float32(4.0), eval_score_mean=3.5)
    rec.log(ORCAMENTO_OFICIAL, train_score_mean=9.0, eval_score_mean=8.1)
    rec.finish({"episodes": 1000, "score_mean": np.float64(8.1), "completo": True})
    caminho = rec.save()

    assert os.path.exists(caminho)
    lido = load(caminho)
    assert lido.algo == "ppo"
    assert lido.seed == 1
    assert lido.config["lr"] == pytest.approx(3e-4)
    assert lido.final["score_mean"] == pytest.approx(8.1)
    assert lido.schema_version == SCHEMA_VERSION
    assert validate(lido) == []


def test_recorder_refuses_to_save_an_invalid_run(tmp_path):
    rec = Recorder("dqn", seed=0, net="", params=0, root=str(tmp_path))
    rec.log(0, train_score_mean=1.0)
    rec.finish({"episodes": 10, "score_mean": 1.0})
    with pytest.raises(ContractViolation):
        rec.save()


def test_numpy_types_survive_serialization(tmp_path):
    rec = Recorder("a2c", seed=0, net="cnn3", params=1234,
                   config={"total_steps": ORCAMENTO_OFICIAL}, root=str(tmp_path))
    rec.log(np.int64(0), x=np.float32(1.5), y=np.int32(3), z=np.array([1, 2]))
    rec.log(ORCAMENTO_OFICIAL, x=np.float32(2.5))
    rec.finish({"episodes": 1000, "score_mean": np.float64(5.0), "completo": True})
    caminho = rec.save()
    bruto = json.load(open(caminho, encoding="utf-8"))
    ponto = bruto["curve"][0]
    assert isinstance(ponto["x"], float) and isinstance(ponto["y"], int)
    assert ponto["z"] == [1, 2]


def test_provenance_is_stamped(tmp_path):
    rec = Recorder("ppo", seed=0, net="resnet_small", params=10, root=str(tmp_path))
    assert "commit" in rec.record.meta
    assert "numpy" in rec.record.meta
    assert "created_at" in rec.record.meta


def test_load_all_finds_and_sorts(tmp_path):
    for algo, seed in (("ppo", 1), ("ppo", 0), ("dqn", 0)):
        r = registro_valido(algo=algo, seed=seed)
        save(r, os.path.join(str(tmp_path), algo, "v", f"seed{seed}", "history.json"))
    todos = load_all(str(tmp_path))
    assert [(r.algo, r.seed) for r in todos] == [("dqn", 0), ("ppo", 0), ("ppo", 1)]


def test_eval_curve_only_returns_evaluated_points():
    x, y = registro_valido().eval_curve()
    assert x.tolist() == [50_000, ORCAMENTO_OFICIAL]
    assert y.tolist() == [3.5, 8.1]


def test_rel_path_matches_the_contract_layout():
    assert registro_valido().rel_path.replace("\\", "/") == \
        "runs/ppo/resnet_small/seed0/history.json"


# ---------------------------------------------------------------------- legado
def test_legacy_csv_becomes_a_non_comparable_record(tmp_path):
    csv_path = tmp_path / "epsgreedy" / "keras_training_data.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        ",0,1,2,3\n"
        "0,3.0,16.0,0.0,-1.075\n"
        "1,5.0,40.0,0.1,-1.010\n"
        "2,7.0,90.0,0.2,0.500\n",
        encoding="utf-8",
    )
    r = from_legacy_csv(str(csv_path))
    assert r.comparable is False
    assert r.caveat
    assert r.variant == "epsgreedy"
    # comprimento 3 vira score 0 — a conversão que torna as curvas sobreponíveis
    assert r.curve[0]["train_score_mean"] == 0.0
    assert r.curve[2]["train_score_mean"] == 4.0
    assert validate(r) == [], "curva legada válida deve passar pela porta de contexto"


def test_legacy_csv_without_usable_rows_raises(tmp_path):
    ruim = tmp_path / "vazio.csv"
    ruim.write_text(",0,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        from_legacy_csv(str(ruim))


def test_budget_is_part_of_the_contract():
    """Comparar 5 M passos com 500 mil mede paciência, não algoritmo."""
    curto = registro_valido(config={"total_steps": 500_000})
    assert any("orçamento" in p for p in validate(curto))
    sem = registro_valido(config={})
    assert any("total_steps" in p for p in validate(sem))


def test_contract_constant_is_the_documented_one():
    """Se alguém mexer no contrato, este teste obriga a mexer conscientemente."""
    assert CONTRATO["board_size"] == 10
    assert CONTRATO["metric"] == "score"
    assert CONTRATO["eval_episodes"] == 1000
    assert CONTRATO["eval_seed"] == 123
    assert CONTRATO["eval_safety"] is False


def test_no_test_writes_run_artifacts_into_the_repository():
    """Nenhum teste pode chamar `train()` sem redirecionar `runs_dir` e `ckpt_dir`.

    Isto é um meta-teste porque a falha é invisível pelo caminho normal: um teste sem esses
    dois parâmetros usa os padrões e escreve em `runs/<algo>/<variante>/seed0/` **dentro do
    repositório**. Foi o que aconteceu com `runs/ppo/resnet_tiny/seed0/`, e o diff aparecia
    só em `wall_s` — parece ruído, e por isso ninguém olhava. Num diretório com resultado de
    verdade, seria uma execução de 5 M passos sobrescrita por um teste de 400.

    A varredura é textual de propósito: exercitar isto de verdade exigiria rodar a suíte
    inteira duas vezes e comparar o `git status`, o que custa vinte minutos para proteger
    algo que uma expressão regular pega em milissegundos.
    """
    import pathlib
    import re

    faltando = []
    for caminho in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
        texto = caminho.read_text(encoding="utf-8")
        for m in re.finditer(r"def (test_\w+)\([^)]*\)[^\n]*\n(.*?)(?=\ndef |\Z)",
                             texto, re.S):
            nome, corpo = m.groups()
            if ".train(" in corpo and "runs_dir" not in corpo:
                faltando.append(f"{caminho.name}::{nome}")

    assert not faltando, (
        "estes testes chamam train() sem isolar o disco e vão sujar o repositório:\n  "
        + "\n  ".join(faltando))
