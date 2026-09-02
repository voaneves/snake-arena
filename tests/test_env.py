"""Invariantes do `VecSnake`.

Estes testes são o contrato do ambiente em forma executável. Se algum deles quebrar,
**nenhum número do benchmark vale**, porque a comparabilidade entre algoritmos depende
de o jogo ser exatamente o mesmo para todos.

Rodam em segundos, sem GPU e sem TensorFlow.
"""

import numpy as np
import pytest

from snakeai.env.vec_snake import (
    DIRS,
    N_ACTIONS,
    N_CHANNELS,
    VecSnake,
)


def make(n=64, b=10, seed=0, **kw):
    return VecSnake(n, b, rng=np.random.default_rng(seed), **kw)


def random_masked_actions(mask, rng):
    """Escolhe uniformemente entre as ações permitidas pela máscara."""
    p = mask.astype(np.float64)
    p /= p.sum(axis=1, keepdims=True)
    return (p.cumsum(axis=1) > rng.random((mask.shape[0], 1))).argmax(axis=1).astype(np.int32)


# --------------------------------------------------------------------- formato
def test_shapes_and_dtypes():
    env = make()
    obs, mask = env.reset()
    assert obs.shape == (64, 10, 10, N_CHANNELS)
    assert obs.dtype == np.float32
    assert mask.shape == (64, N_ACTIONS)
    assert mask.dtype == np.bool_


def test_board_too_small_is_rejected():
    with pytest.raises(ValueError):
        VecSnake(4, board_size=5)


def test_initial_state():
    env = make()
    env.check_invariants()
    assert (env.length == 3).all()
    assert (env.score == 0).all(), "score começa em zero, não em 3 — este é o ponto"
    assert (env.steps == 0).all()


# ----------------------------------------------------------------- invariantes
def test_invariants_hold_over_a_long_rollout():
    env = make(n=64)
    rng = np.random.default_rng(1)
    obs, mask = env.reset()
    for _ in range(2000):
        a = random_masked_actions(mask, rng)
        obs, mask, r, d, info = env.step(a, shaping_coef=0.1)
        env.check_invariants()
        assert np.isfinite(obs).all()
        assert np.isfinite(r).all()


def test_score_is_always_length_minus_three():
    """A regra que torna as curvas antigas e novas conversíveis."""
    env = make(n=32)
    rng = np.random.default_rng(2)
    obs, mask = env.reset()
    for _ in range(1500):
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        assert (env.score == env.length - 3).all()


# --------------------------------------------------------------------- máscara
def test_mask_never_allows_immediate_death():
    """Se a máscara diz que dá, não pode morrer — salvo beco sem saída."""
    env = make(n=128)
    rng = np.random.default_rng(3)
    obs, mask = env.reset()
    for _ in range(1500):
        dead_end = env.dead_ends()
        allowed = mask.copy()
        a = random_masked_actions(mask, rng)
        was_allowed = allowed[np.arange(env.n), a]
        obs, mask, r, d, info = env.step(a)
        # morte por colisão vale exatamente -1; fome vale -0,5 e não depende da máscara
        morreu = np.isclose(r, -1.0)
        # onde a ação era permitida e não era beco sem saída, não pode ter morrido
        assert not (morreu & was_allowed & ~dead_end).any()


def test_dead_ends_are_reported_and_masked_open():
    """Num beco sem saída, `action_mask` libera tudo e `dead_ends` acusa."""
    env = make(n=128)
    rng = np.random.default_rng(11)
    obs, mask = env.reset()
    viu = False
    for _ in range(1500):
        de = env.dead_ends()
        if de.any():
            viu = True
            assert mask[de].all(), "beco sem saída deve liberar as três ações"
            assert not env._raw_mask()[de].any(), "nenhuma delas é de fato segura"
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
    assert viu, "nenhum beco sem saída no rollout — ajuste o teste"


def test_mask_is_never_all_false():
    env = make(n=128)
    rng = np.random.default_rng(4)
    obs, mask = env.reset()
    for _ in range(800):
        assert mask.any(axis=1).all(), "máscara sem suporte quebra o log-prob"
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))


# ------------------------------------------------------------------ recompensa
def test_reward_is_plus_one_for_food_minus_one_for_death():
    env = make(n=256)
    rng = np.random.default_rng(5)
    obs, mask = env.reset()
    viu_comida = viu_morte = False
    for _ in range(1200):
        length_before = env.length.copy()
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        comeu = (env.length == length_before + 1) & ~d
        if comeu.any():
            assert np.allclose(r[comeu], 1.0)
            viu_comida = True
        morreu = d & (info["deaths"] > 0) & (r == -1.0)
        if morreu.any():
            viu_morte = True
    assert viu_comida and viu_morte, "o teste não exercitou os dois casos"


def test_shaping_is_zero_sum_ish_and_bounded():
    """O shaping potencial não pode dominar a recompensa real."""
    env = make(n=128)
    rng = np.random.default_rng(6)
    obs, mask = env.reset()
    for _ in range(500):
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng), shaping_coef=0.5)
        assert (np.abs(r) < 3.0).all()


# ------------------------------------------------------------------------ fome
def test_starvation_truncates_and_reports_final_obs():
    """Fome é truncamento, não terminação — precisa devolver o estado final."""
    env = make(n=16, starve_base=12)
    rng = np.random.default_rng(7)
    obs, mask = env.reset()
    viu = False
    for _ in range(400):
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        if info["starved"] > 0:
            viu = True
            k = info["trunc_idx"].size
            assert k == info["starved"]
            assert info["final_obs"].shape == (k, env.b, env.b, N_CHANNELS)
            assert info["final_mask"].shape == (k, N_ACTIONS)
    assert viu, "nenhum episódio morreu de fome — ajuste o teste"


def test_hunger_resets_when_eating():
    env = make(n=64)
    rng = np.random.default_rng(8)
    obs, mask = env.reset()
    for _ in range(600):
        length_before = env.length.copy()
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        comeu = (env.length == length_before + 1) & ~d
        if comeu.any():
            assert (env.hunger[comeu] == 0).all()


# ------------------------------------------------------- reprodutibilidade
def test_same_seed_same_trajectory():
    """Sem isto, nenhuma comparação entre algoritmos significa nada."""
    def roda(seed):
        env = make(n=32, seed=seed)
        rng = np.random.default_rng(99)
        obs, mask = env.reset()
        rs = []
        for _ in range(300):
            obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
            rs.append(r.copy())
        return np.array(rs)

    assert np.array_equal(roda(0), roda(0))
    assert not np.array_equal(roda(0), roda(1))


def test_reset_is_deterministic_given_the_generator():
    a = make(n=16, seed=123)
    b = make(n=16, seed=123)
    assert np.array_equal(a.occ, b.occ)
    assert np.array_equal(a.food, b.food)
    assert np.array_equal(a.dir, b.dir)


# ---------------------------------------------------------------- observação
def test_observation_is_egocentric():
    """A cabeça sempre olha para cima: a célula à frente é sempre a mesma no obs."""
    env = make(n=256)
    obs, mask = env.reset()
    canal_cabeca = obs[..., 1]
    # existe exatamente uma cabeça por tabuleiro
    assert np.allclose(canal_cabeca.sum(axis=(1, 2)), 1.0)
    pos = np.argwhere(canal_cabeca > 0.5)
    # após a rotação egocêntrica, a direção é constante — testamos que girar o
    # tabuleiro de volta recupera a posição real da cabeça
    for i in range(env.n):
        y, x = pos[pos[:, 0] == i][0][1:]
        raw = np.rot90(obs[i], k=-env.dir[i], axes=(0, 1))
        yy, xx = np.argwhere(raw[..., 1] > 0.5)[0]
        assert (yy, xx) == (env.head[i, 0], env.head[i, 1])


def test_head_is_not_in_the_body_channel():
    """O bug do jogo antigo: a cabeça era apagada pelo laço que desenha o corpo."""
    env = make(n=64)
    obs, mask = env.reset()
    corpo, cabeca = obs[..., 0], obs[..., 1]
    assert not ((corpo > 0.5) & (cabeca > 0.5)).any()
    assert np.allclose(cabeca.sum(axis=(1, 2)), 1.0)
    assert np.allclose(corpo.sum(axis=(1, 2)), 2.0), "corpo inicial = 3 - 1 cabeça"


def test_food_channel_has_exactly_one_cell():
    env = make(n=64)
    obs, mask = env.reset()
    assert np.allclose(obs[..., 3].sum(axis=(1, 2)), 1.0)


def test_food_never_spawns_inside_the_body():
    env = make(n=64)
    rng = np.random.default_rng(10)
    obs, mask = env.reset()
    rows = np.arange(env.n)
    for _ in range(800):
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        vivos = env.length < env.cells
        ocupada = env.occ[rows, env.food[:, 0], env.food[:, 1]] > 0
        assert not (ocupada & vivos).any()


# -------------------------------------------------------------- flood fill
def test_free_space_on_an_empty_board():
    env = make(n=1, b=10)
    env.occ[0] = 0
    # tabuleiro vazio: alcança as 100 células
    assert env.free_space_from(0, np.array([5, 5])) == 100


def test_free_space_respects_the_body():
    env = make(n=1, b=10)
    env.occ[0] = 0
    env.occ[0, 3, :] = 5      # parede horizontal de corpo "fresco"
    alcancavel = env.free_space_from(0, np.array([0, 0]))
    assert alcancavel == 30, "só as 3 linhas acima da parede"


# ------------------------------------------------------------------ piso
def test_masked_random_baseline_is_around_one():
    """O piso documentado no README: 1,08. Se mudar, o contrato mudou."""
    env = make(n=250, b=10, seed=123)
    rng = np.random.default_rng(123)
    obs, mask = env.reset()
    scores = []
    while len(scores) < 1000:
        obs, mask, r, d, info = env.step(random_masked_actions(mask, rng))
        scores.extend(info["scores"].tolist())
    media = float(np.mean(scores[:1000]))
    assert 0.7 < media < 1.6, f"piso aleatório fora do esperado: {media:.2f}"


def test_the_shaping_potential_is_exposed_and_recomposes_exactly():
    """`phi_old`, `phi_new` e `shaping_valido` no `info` são **informação**, não
    comportamento: a recompensa devolvida é a mesma com ou sem eles.

    O contrato que este teste guarda é o que permite ao LBC dar um coeficiente de shaping
    **por política** sem reimplementar o potencial (`docs/LBC.md` §2.2): com
    `shaping_coef=0` a recompensa é a esparsa pura, e

        r = esparsa + coef · shaping_valido · (γ·phi_new − phi_old)

    tem que reproduzir, bit a bit, o que o ambiente calcula sozinho. Se alguém mudar a
    fórmula do shaping dentro do `step()` e esquecer do `info`, é aqui que quebra.
    """
    coef, gamma = 0.5, 0.995
    a = VecSnake(64, 10, rng=np.random.default_rng(7))
    b = VecSnake(64, 10, rng=np.random.default_rng(7))
    a.reset(); b.reset()
    acoes = np.random.default_rng(0)
    for _ in range(300):
        act = acoes.integers(0, N_ACTIONS, 64).astype(np.int32)
        _, _, r_env, d_env, _ = a.step(act, coef, gamma)
        _, _, r_esparsa, d_man, info = b.step(act, 0.0, gamma)

        for k in ("phi_old", "phi_new", "shaping_valido"):
            assert k in info, f"o `info` perdeu {k!r}"
        r_manual = r_esparsa + coef * info["shaping_valido"] * (
            gamma * info["phi_new"] - info["phi_old"])

        assert r_env == pytest.approx(r_manual, abs=1e-6)
        assert (d_env == d_man).all()


def test_the_potential_is_invalid_where_the_food_moved_and_only_there():
    """`shaping_valido` é `~(morreu | venceu | comeu)` — e **não** `~done`.

    A diferença é a fome, e ela não é um detalhe: fome é `done`, mas é *truncamento*, não
    terminação. A comida não mudou de lugar, o episódio continuaria, e o delta de potencial
    daquele passo é tão legítimo quanto o de qualquer outro. Zerá-lo ali seria descartar
    informação boa; deixá-lo valer em morte, vitória ou comida seria comparar distâncias
    para **comidas diferentes**, que é um número sem significado.

    É a mesma convenção que o `bootstrap_truncados` do agente segue, e pelo mesmo motivo.
    """
    env = VecSnake(128, 10, rng=np.random.default_rng(3))
    env.reset()
    acoes = np.random.default_rng(1)
    viu_fome_valida = viu_invalido = False
    for _ in range(400):
        act = acoes.integers(0, N_ACTIONS, 128).astype(np.int32)
        _, _, _, done, info = env.step(act, 0.5, 0.995)
        vale = info["shaping_valido"]

        # quem terminou por morte ou vitória tem que estar zerado
        terminou = done.copy()
        terminou[info["trunc_idx"]] = False
        assert not vale[terminou].any(), "morte ou vitória com potencial válido"

        # quem foi truncado por fome **pode** continuar válido — e é o ponto do teste
        if info["trunc_idx"].size and vale[info["trunc_idx"]].any():
            viu_fome_valida = True
        viu_invalido |= bool((~vale).any())

    assert viu_invalido, "400 passos sem um passo inválido: o teste não mediu nada"
    assert viu_fome_valida, (
        "nenhum truncamento por fome manteve o potencial válido — ou o teste não "
        "alcançou a fome, ou `shaping_valido` virou `~done` e passou a descartar "
        "informação boa")
