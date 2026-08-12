"""ACKTR — o A2C com gradiente natural.

O que estes testes protegem: que a única diferença entre A2C e ACKTR seja o
pré-condicionamento (senão a comparação na arena não mede o que diz medir), e que a região
de confiança seja de verdade — isto é, que a KL **medida depois do passo** respeite o alvo.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import A2C, A2CConfig, ACKTR, ACKTRConfig
from snakeai.plot import ORDEM_ALGORITMOS, cores_por_algoritmo


def cfg(**kw):
    base = dict(net="resnet_tiny", num_envs=32, rollout=8,
                eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return ACKTRConfig(**base)


# ---------------------------------------------------------------- o controle
def test_acktr_is_a2c_plus_kfac_and_nothing_else():
    """A herança é o que garante que a comparação A2C × ACKTR isole o gradiente natural.

    Se `collect` divergir, a diferença entre as curvas passa a incluir o rollout, e a
    afirmação "isto mede curvatura" deixa de ser verdadeira sem que nada quebre.
    """
    assert issubclass(ACKTR, A2C)
    assert ACKTR.collect is A2C.collect
    assert ACKTR.iterate is A2C.iterate
    assert ACKTR.update is not A2C.update, "o update é a única coisa que muda"


def test_acktr_is_its_own_algorithm_in_the_arena():
    ag = ACKTR(cfg())
    assert ag.algo == "acktr"
    assert "acktr" in ORDEM_ALGORITMOS
    cores = cores_por_algoritmo({"a2c", "acktr"})
    assert cores["acktr"] != cores["a2c"]


# ------------------------------------------------------------------- treino
def test_acktr_trains():
    ag = ACKTR(cfg())
    for _ in range(3):
        s = ag.iterate()
    for chave in ("pg", "vf", "ent", "lr", "kl"):
        assert np.isfinite(s[chave]), f"{chave} virou {s[chave]}"
    assert s["ent"] > 0


def test_kfac_covers_almost_the_whole_network():
    r = ACKTR(cfg()).resumo_kfac()
    assert r["fracao"] > 0.95, f"cobertura {r['fracao']:.1%} — K-FAC pela metade"
    assert "logits" in r["camadas"] and "value" in r["camadas"]


# --------------------------------------------------------- região de confiança
def test_measured_kl_respects_the_target():
    """A KL alvo é derivada de uma aproximação quadrática; esta é a KL que aconteceu.

    Aproximações de segunda ordem se degradam justamente quando o passo é grande — então
    um alvo que ninguém confere é um parâmetro decorativo.
    """
    ag = ACKTR(cfg(kl_max=1e-3))
    for _ in range(5):
        s = ag.iterate()
        assert s["kl"] <= 3 * s["kl_alvo"], f"KL {s['kl']:.5f} contra alvo {s['kl_alvo']}"


def test_a_tighter_kl_target_produces_a_smaller_step():
    """O tamanho do passo tem que sair da KL, não do learning rate."""
    passos = {}
    for alvo in (1e-4, 1e-2):
        ag = ACKTR(cfg(kl_max=alvo, seed=0))
        passos[alvo] = np.mean([ag.iterate()["lr"] for _ in range(3)])
    assert passos[1e-4] < passos[1e-2]


def test_the_learning_rate_is_only_a_ceiling():
    ag = ACKTR(cfg(kl_max=10.0, lr_start=0.05, lr_end=0.05))
    s = ag.iterate()
    assert s["lr"] == pytest.approx(0.05, rel=1e-4), \
        "com KL alvo enorme o passo tem que bater no teto do lr"


# ------------------------------------------------------------------- robustez
def test_reloading_the_model_rebuilds_the_preconditioner():
    """Sem isto, o K-FAC seguiria indexando as variáveis do modelo antigo — e
    pré-condicionaria camadas trocadas, sem erro nenhum."""
    ag = ACKTR(cfg())
    ag.iterate()
    antigo = ag.kfac
    ag.on_model_reloaded()
    assert ag.kfac is not antigo
    assert all(v1 is v2 for v1, v2 in
               zip(ag.kfac.model.trainable_variables, ag.model.trainable_variables))
    ag.iterate()


def test_capture_does_not_leak_into_evaluation():
    """Depois do update, o modelo tem que voltar ao `call` normal — a avaliação não pode
    ficar registrando ativações num escopo que já foi fechado."""
    from snakeai.kfac import _REGISTRO

    ag = ACKTR(cfg())
    ag.iterate()
    assert _REGISTRO == []
    for c in ag.kfac.camadas:
        assert "call" not in c.__dict__


def test_acktr_records_the_kfac_cost():
    """`kfac_ms` existe para que "segunda ordem compensa?" possa ser respondida em tempo
    de parede, e não só em passos de ambiente."""
    s = ACKTR(cfg()).iterate()
    assert s["kfac_ms"] > 0 and s["fwd_ms"] > 0
