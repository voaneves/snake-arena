"""Colab, Kaggle e local — o mesmo `.ipynb` nos três.

O que estes testes protegem é a retomada. Um treino de 5 M passos não cabe numa sessão
gratuita sem cair pelo menos uma vez, então "continuar de onde parou" não é conveniência,
é requisito. E cada serviço a implementa de um jeito diferente.
"""

import os

import pytest

from snakeai import plataforma
from snakeai.plataforma import (COLAB, KAGGLE, LOCAL, detecta, entregar_arquivo,
                                pasta_de_trabalho, resumo, semear_checkpoints)


@pytest.fixture
def finge_kaggle(tmp_path, monkeypatch):
    """Monta uma árvore igual à do Kaggle, com uma execução anterior anexada."""
    trabalho = tmp_path / "kaggle" / "working"
    entrada = tmp_path / "kaggle" / "input" / "execucao-anterior" / "checkpoints"
    trabalho.mkdir(parents=True)
    entrada.mkdir(parents=True)
    (entrada / "ppo_last.keras").write_text("modelo antigo")
    (entrada / "ppo_last.json").write_text('{"global_step": 2000000}')

    real_isdir, real_walk = os.path.isdir, os.walk
    monkeypatch.setattr(plataforma, "detecta", lambda: KAGGLE)
    monkeypatch.setattr(os.path, "isdir",
                        lambda p: True if p == "/kaggle/input" else real_isdir(p))
    monkeypatch.setattr(os, "walk",
                        lambda p, *a, **k: real_walk(str(entrada.parent), *a, **k)
                        if p == "/kaggle/input" else real_walk(p, *a, **k))
    return tmp_path


# ------------------------------------------------------------------- detecção
def test_detects_local_when_neither_service_is_present():
    """A suíte roda aqui: se a detecção não caísse em `local`, ela mesma não passaria."""
    assert detecta() == LOCAL


def test_detection_is_by_capability_not_by_a_variable():
    """Uma variável de ambiente qualquer não pode fazer o notebook achar que está no Kaggle.

    Se bastasse `KAGGLE_...` estar setada, um ambiente que a herdou por acidente escreveria
    em `/kaggle/working` — que não existe — e o treino morreria no primeiro checkpoint.
    """
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
    try:
        assert detecta() == LOCAL
    finally:
        os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)


# ---------------------------------------------------------------------- pasta
def test_local_folder_is_created_and_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = pasta_de_trabalho(nome="arena-teste", verbose=False)
    assert os.path.isabs(p) and os.path.isdir(p)


def test_usar_drive_is_ignored_outside_colab(tmp_path, monkeypatch):
    """O mesmo notebook roda nos três sem editar célula — então o parâmetro do Colab não
    pode virar erro no Kaggle."""
    monkeypatch.chdir(tmp_path)
    a = pasta_de_trabalho(usar_drive=True, nome="x", verbose=False)
    b = pasta_de_trabalho(usar_drive=False, nome="x", verbose=False)
    assert a == b


# ------------------------------------------------------------------- retomada
def test_kaggle_seeds_checkpoints_from_the_previous_run(finge_kaggle, tmp_path):
    """É isto que faz "retomar" existir no Kaggle: a sessão nova nasce vazia, e o que
    sobreviveu está montado somente-leitura em `/kaggle/input`."""
    ckpt = tmp_path / "ckpt"
    copiados = semear_checkpoints(str(ckpt), verbose=False)

    assert len(copiados) == 2
    assert (ckpt / "ppo_last.keras").read_text() == "modelo antigo"


def test_a_checkpoint_from_this_session_always_wins(finge_kaggle, tmp_path):
    """Sobrescrever o checkpoint atual com o de uma execução anterior faria o treino andar
    **para trás** — e sem erro nenhum, porque os dois arquivos são válidos."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "ppo_last.keras").write_text("modelo desta sessão")

    semear_checkpoints(str(ckpt), verbose=False)
    assert (ckpt / "ppo_last.keras").read_text() == "modelo desta sessão"


def test_seeding_is_a_no_op_outside_kaggle(tmp_path):
    assert semear_checkpoints(str(tmp_path / "ckpt"), verbose=False) == []


# -------------------------------------------------------------------- entrega
def test_delivering_a_file_never_raises_outside_colab(tmp_path, capsys):
    """A entrega é conveniência; o arquivo é o resultado. Uma exceção aqui derrubaria a
    última célula depois de horas de treino, por causa de um download."""
    alvo = tmp_path / "resultado.zip"
    alvo.write_text("conteúdo")
    assert entregar_arquivo(str(alvo)) is False
    assert str(alvo) in capsys.readouterr().out


def test_kaggle_explains_where_the_file_is(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(plataforma, "detecta", lambda: KAGGLE)
    alvo = tmp_path / "r.zip"
    alvo.write_text("x")
    entregar_arquivo(str(alvo))
    assert "Output" in capsys.readouterr().out


# --------------------------------------------------------------------- resumo
def test_resumo_records_the_platform_and_the_accelerators():
    r = resumo()
    assert r["plataforma"] in (COLAB, KAGGLE, LOCAL)
    assert r["n_gpus"] == len(r["gpus"])
