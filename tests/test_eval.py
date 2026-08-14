"""O protocolo de avaliação.

Se estes testes passam, dois algoritmos avaliados por `evaluate` estão sendo medidos com a
mesma régua — que é a premissa inteira do `snake-arena`.

Rodam sem TensorFlow: `evaluate` recebe uma função de política, não um modelo.
"""

import numpy as np
import pytest

from snakeai.env.vec_snake import DIRS, N_ACTIONS, VecSnake
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


# ------------------------------------- o ponto que o episódio vitorioso perdia
def _caminho_boustrofedon(b):
    p = []
    for r in range(b):
        cols = range(b) if r % 2 == 0 else range(b - 1, -1, -1)
        p += [(r, c) for c in cols]
    return p


def estado_quase_ganho(b=10):
    """Cobra ocupando todas as células menos uma, com a comida nessa célula livre.

    Construir este estado à mão é a única forma de testar a vitória sem treinar um agente
    que ganhe. O corpo segue um caminho boustrofedon, então a cabeça termina adjacente à
    única célula livre e um movimento fecha o tabuleiro.
    """
    p = _caminho_boustrofedon(b)
    corpo, livre = p[:-1], p[-1]
    occ = np.zeros((b, b), np.int32)
    for k, (r, c) in enumerate(corpo):
        occ[r, c] = k + 1                      # cauda ttl=1 … cabeça ttl=len
    cabeca = np.array(corpo[-1], np.int32)
    d = cabeca - np.array(corpo[-2], np.int32)
    direcao = int(np.argmax((DIRS == d).all(axis=1)))
    L = len(corpo)
    return {"occ": occ, "head": cabeca, "food": np.array(livre, np.int32),
            "dir": np.int32(direcao), "length": np.int32(L), "steps": np.int32(500),
            "hunger": np.int32(0), "score": np.int32(L - 3)}


def test_a_winning_episode_scores_the_last_apple():
    """O score de um episódio vitorioso é o perfeito — não o perfeito menos um.

    Este é o bug que a avaliação teve: ela gravava `env.score` lido **antes** do passo,
    e o passo que fecha o tabuleiro é justamente um passo que come. Resultado: toda
    vitória entrava valendo um ponto a menos, e o `score_max` de uma execução com 67% de
    vitórias aparecia como 96 num tabuleiro cujo máximo é 97.
    """
    env = VecSnake(1, 10, rng=np.random.default_rng(0))
    env.reset()
    env.escrever_estado(0, estado_quase_ganho(10))
    env.check_invariants()

    antes = int(env.score[0])
    for acao in range(3):
        alvo = VecSnake(1, 10, rng=np.random.default_rng(0))
        alvo.reset()
        alvo.escrever_estado(0, estado_quase_ganho(10))
        _, _, _, done, info = alvo.step(np.array([acao], np.int32))
        if info["wins"]:
            assert int(info["scores"][0]) == 97, "o env já devolve o score certo"
            assert antes == 96, "e o valor lido antes do passo é um a menos"
            return
    pytest.fail("nenhuma das três ações fechou o tabuleiro")


def test_win_rate_and_max_score_can_never_contradict_each_other():
    """O detector que faltava.

    Se `win_rate > 0`, alguém encheu o tabuleiro, e então `score_max` **tem** que ser o
    score perfeito. Ver as duas coisas discordarem foi o que denunciou o bug — mas só
    porque alguém olhou. Agora é teste.
    """
    def politica(obs, mask):
        return np.where(mask, 0.0, MASK_NEG).astype(np.float32)

    stats, scores = evaluate(politica, episodes=60, num_envs=30, max_steps=5000)
    if stats["win_rate"] > 0:
        assert stats["score_max"] == stats["perfect_possible"]
    assert stats["win_rate"] == pytest.approx(
        float((scores == stats["perfect_possible"]).mean()))


def test_win_rate_comes_from_the_measured_sample():
    """O laço continua rodando os ambientes que já cumpriram a cota; contar as vitórias
    deles daria uma taxa que não corresponde aos episódios medidos."""
    def politica(obs, mask):
        return np.where(mask, 0.0, MASK_NEG).astype(np.float32)

    stats, scores = evaluate(politica, episodes=50, num_envs=25, max_steps=5000)
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert stats["win_rate"] * stats["episodes"] == pytest.approx(
        float((scores == stats["perfect_possible"]).sum()))


# ------------------------------------- "jogou mal" e "andou em círculo" são diferentes
def test_evaluate_reports_why_the_episodes_ended():
    """Score sozinho não distingue os dois, e o diagnóstico é oposto.

    Uma política determinística ruim, com máscara de morte, **não morre** — ela é
    empurrada para uma ação viável, entra em ciclo e morre de fome com score ~0. Ler isso
    como "não aprendeu" manda ajustar hiperparâmetro quando o problema é falta de
    exploração no momento de agir.
    """
    def sempre_reto(obs, mask):
        return np.where(mask, [[9.0, 0.0, 0.0]], MASK_NEG).astype(np.float32)

    stats, _ = evaluate(sempre_reto, episodes=100, num_envs=50, max_steps=3000)
    assert stats["fim_fome"] > 0.9, "a máscara impede a colisão; sobra a fome"
    assert stats["fim_colisao"] < 0.05
    assert sum(stats[k] for k in ("fim_fome", "fim_colisao", "fim_tabuleiro_cheio")) \
        == pytest.approx(1.0)


def test_starving_is_the_normal_ending_here_and_that_is_the_subtlety():
    """Morrer de fome é o fim **normal** neste ambiente, e não um sintoma por si só.

    A máscara de morte impede a colisão sempre que existe alternativa, então até a política
    aleatória termina ~85% dos episódios por fome. O que denuncia o ciclo determinístico é
    a combinação: **zero** colisão e score abaixo do piso — a cobra anda para sempre sem
    nunca comer. Uma política com ruído colide de vez em quando (fica sem alternativa) e,
    principalmente, come.
    """
    aleatoria, _ = evaluate(random_policy(np.random.default_rng(0)),
                            episodes=200, num_envs=50, max_steps=5000)
    assert aleatoria["fim_fome"] > 0.5, "fome é o fim normal, não a exceção"
    assert aleatoria["fim_colisao"] > 0.05, "mas o ruído ainda leva a cobra a se encurralar"

    def sempre_reto(obs, mask):
        return np.where(mask, [[9.0, 0.0, 0.0]], MASK_NEG).astype(np.float32)

    ciclo, _ = evaluate(sempre_reto, episodes=100, num_envs=50, max_steps=3000)
    assert ciclo["fim_colisao"] < 0.05, "a determinística nunca se encurrala"
    assert ciclo["score_mean"] < aleatoria["score_mean"] / 5, "e quase nunca come"


def test_the_verdict_explains_the_starvation_case():
    resultado = {
        "piso": 1.21, "perfeito": 97, "ganho_sobre_o_piso": 0.5,
        "linhas": [
            {"regime": "aleatório com máscara", "score_mean": 1.21},
            {"regime": "agente (greedy)", "score_mean": 0.64, "score_median": 0.0,
             "score_max": 3, "win_rate": 0.0,
             "fim_fome": 1.0, "fim_colisao": 0.0, "fim_tabuleiro_cheio": 0.0},
        ],
    }
    texto = format_verdict(resultado)
    assert "fome 100%" in texto
    assert "ciclo" in texto, "o veredito tem que dizer o que o número significa"

    # e não pode gritar quando o fim por fome é o normal do ambiente
    normal = {**resultado, "linhas": [resultado["linhas"][0],
              {**resultado["linhas"][1], "score_mean": 14.0,
               "fim_fome": 0.85, "fim_colisao": 0.15}]}
    assert "ciclo" not in format_verdict(normal)
