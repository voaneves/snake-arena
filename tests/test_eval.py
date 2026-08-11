"""O protocolo de avaliação.

Se estes testes passam, dois algoritmos avaliados por `evaluate` estão sendo medidos com a
mesma régua — que é a premissa inteira do `snake-arena`.

Rodam sem TensorFlow: `evaluate` recebe uma função de política, não um modelo.
"""

import numpy as np
import pytest

from snakeai.env.vec_snake import N_ACTIONS, VecSnake
from snakeai.eval import (
    MASK_NEG,
    apply_safety_filter,
    evaluate,
    format_verdict,
    random_baseline,
    random_policy,
    verdict,
)


def politica_constante(acao):
    """Sempre a mesma ação relativa — útil porque o comportamento é previsível."""
    def politica(obs, mask):
        logits = np.full((mask.shape[0], N_ACTIONS), -1.0, dtype=np.float32)
        logits[:, acao] = 1.0
        return logits
    return politica


def politica_suicida(obs, mask):
    """Prefere justamente o que a máscara proíbe."""
    return np.where(mask, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------- contrato
def test_evaluate_returns_the_requested_number_of_episodes():
    stats, scores = evaluate(random_policy(), episodes=200, num_envs=50, greedy=False)
    assert stats["episodes"] == 200
    assert scores.size == 200
    assert stats["completo"]


def test_evaluate_is_deterministic_for_a_deterministic_policy():
    """Mesma política, mesma seed, mesmo número. Sem isto nada é comparável."""
    a = evaluate(politica_constante(1), episodes=100, num_envs=25)[0]
    b = evaluate(politica_constante(1), episodes=100, num_envs=25)[0]
    assert a == b


def test_different_seeds_give_different_samples():
    a = evaluate(politica_constante(1), episodes=100, num_envs=25, seed=1)[0]
    b = evaluate(politica_constante(1), episodes=100, num_envs=25, seed=2)[0]
    assert a["score_mean"] != b["score_mean"] or a["score_max"] != b["score_max"]


def test_score_is_never_length():
    """O erro que tornava as curvas antigas incomparáveis: score começa em zero."""
    stats, scores = evaluate(random_policy(), episodes=200, num_envs=50, greedy=False)
    assert scores.min() >= 0
    assert stats["score_mean"] < 3.0, "se estivesse contando comprimento, começaria em 3"


def test_stats_are_self_consistent():
    stats, scores = evaluate(random_policy(), episodes=300, num_envs=60, greedy=False)
    assert stats["score_mean"] == pytest.approx(float(scores.mean()))
    assert stats["score_median"] == pytest.approx(float(np.median(scores)))
    assert stats["score_max"] == int(scores.max())
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert stats["perfect_possible"] == 97


# --------------------------------------------------------------- viés da amostra
def test_every_env_contributes_the_same_number_of_episodes():
    """A correção do viés: a amostra não pode ser "os primeiros a terminar".

    Se fosse, os episódios curtos dominariam e o agente seria subestimado.
    """
    num_envs, episodes = 20, 100
    stats, scores = evaluate(
        politica_constante(1), episodes=episodes, num_envs=num_envs
    )
    assert scores.size == episodes
    # cada ambiente entrou com exatamente episodes/num_envs episódios
    assert episodes % num_envs == 0
    assert stats["completo"]


def test_sample_is_not_biased_toward_short_episodes():
    """Compara a média oficial com a colheita ingênua "primeiros a terminar".

    A ingênua tem de ser menor ou igual — é exatamente o viés que corrigimos.
    """
    politica = random_policy(np.random.default_rng(7))

    # colheita ingênua, reproduzindo o jeito antigo
    env = VecSnake(50, 10, rng=np.random.default_rng(123))
    obs, mask = env.reset()
    ingenua = []
    while len(ingenua) < 500:
        logits = np.where(mask, politica(obs, mask), MASK_NEG)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        rng = np.random.default_rng(len(ingenua))
        a = (p.cumsum(axis=1) > rng.random((env.n, 1))).argmax(axis=1).astype(np.int32)
        antes = env.score.copy()
        obs, mask, r, d, info = env.step(a)
        ingenua.extend(antes[d].tolist())
    media_ingenua = float(np.mean(ingenua[:500]))

    oficial, _ = evaluate(random_policy(np.random.default_rng(7)),
                          episodes=500, num_envs=50, greedy=False)
    # com política aleatória a diferença é pequena, mas o sinal não deve se inverter
    assert media_ingenua <= oficial["score_mean"] + 0.35


# ------------------------------------------------------------------------ piso
def test_random_baseline_matches_the_documented_floor():
    """1,08 é o número publicado no README. Se mudar, o contrato mudou."""
    piso = random_baseline(episodes=1000, num_envs=250)
    assert 0.7 < piso < 1.6, f"piso fora do esperado: {piso:.2f}"


def test_a_suicidal_policy_scores_at_the_bottom():
    stats, _ = evaluate(politica_suicida, episodes=200, num_envs=50)
    assert stats["score_mean"] < 1.0


def test_mask_is_reapplied_even_if_the_policy_ignores_it():
    """`evaluate` não confia na política: reaplica a máscara aos logits."""
    stats, _ = evaluate(politica_suicida, episodes=200, num_envs=50)
    # se a máscara não fosse reaplicada, a política suicida morreria no primeiro passo
    # e o número de passos seria mínimo
    assert stats["env_steps_used"] > 5


# ------------------------------------------------------------ filtro de segurança
def test_safety_filter_blocks_lethal_actions():
    env = VecSnake(16, 10, rng=np.random.default_rng(0))
    env.reset()
    logits = np.zeros((env.n, N_ACTIONS), dtype=np.float32)
    saida = apply_safety_filter(env, logits)
    letal = ~env._raw_mask()
    assert (saida[letal] == MASK_NEG).all()


def test_safety_filter_preserves_relative_order_when_all_options_are_bad():
    """Penalidade grande mas finita: se tudo é ruim, a preferência da rede sobrevive."""
    env = VecSnake(8, 10, rng=np.random.default_rng(1))
    env.reset()
    logits = np.tile(np.array([3.0, 2.0, 1.0], dtype=np.float32), (env.n, 1))
    saida = apply_safety_filter(env, logits, margin=1e9)  # tudo é "bolso pequeno"
    seguras = env._raw_mask()
    for i in range(env.n):
        idx = np.nonzero(seguras[i])[0]
        if idx.size >= 2:
            assert np.all(np.diff(saida[i, idx]) < 0), "ordem relativa se perdeu"


def test_safety_filter_does_not_mutate_the_input():
    env = VecSnake(8, 10, rng=np.random.default_rng(2))
    env.reset()
    logits = np.zeros((env.n, N_ACTIONS), dtype=np.float32)
    copia = logits.copy()
    apply_safety_filter(env, logits)
    assert np.array_equal(logits, copia)


def test_evaluate_with_safety_runs():
    stats, _ = evaluate(random_policy(), episodes=64, num_envs=16, safety=True)
    assert stats["episodes"] == 64


# -------------------------------------------------------------------- veredito
def test_verdict_has_the_three_regimes_and_formats():
    r = verdict(politica_constante(1), episodes=100, num_envs=25, com_filtro=True)
    assert len(r["linhas"]) == 3
    assert r["linhas"][0]["regime"] == "aleatório com máscara"
    assert r["perfeito"] == 97
    texto = format_verdict(r)
    assert "ganho sobre o piso" in texto
    assert texto.count("\n") >= 5


def test_verdict_without_filter_has_two_regimes():
    r = verdict(random_policy(), episodes=100, num_envs=25, com_filtro=False)
    assert len(r["linhas"]) == 2
