"""MCTS e AlphaZero.

O teste mais importante deste arquivo é `test_search_beats_random_with_an_informative_value`:
uma busca que não joga melhor que o acaso, quando lhe dão um valor informativo, está
quebrada — e quebra de forma silenciosa, porque o código roda, os números saem, e só o
score denuncia. Foi assim que dois bugs desta implementação apareceram.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import AlphaZero, AlphaZeroConfig
from snakeai.env.vec_snake import N_ACTIONS, VecSnake
from snakeai.eval import MASK_NEG
from snakeai.search import MCTS, No


def uniforme(obs, mask):
    p = np.asarray(mask, dtype=np.float64)
    p /= p.sum(1, keepdims=True)
    return p, np.zeros(len(obs), dtype=np.float32)


def heuristica(obs, mask):
    """Valor = −distância de Manhattan até a comida, lida da própria observação."""
    n, b = obs.shape[0], obs.shape[1]
    cab = obs[..., 1].reshape(n, -1).argmax(1)
    com = obs[..., 3].reshape(n, -1).argmax(1)
    d = np.abs(cab // b - com // b) + np.abs(cab % b - com % b)
    p = np.asarray(mask, dtype=np.float64)
    p /= p.sum(1, keepdims=True)
    return p, (-d / (2.0 * b)).astype(np.float32)


def joga(avaliador, sims, n=32, alvo=32, seed=123):
    env = VecSnake(n, 10, rng=np.random.default_rng(seed))
    obs, mask = env.reset()
    mcts = MCTS(avaliador, num_simulations=sims, gamma=0.997,
                rng=np.random.default_rng(0))
    scores = []
    while len(scores) < alvo:
        visitas, _ = mcts.run(env.get_state(), mask, obs)
        obs, mask, r, d, info = env.step(visitas.argmax(1).astype(np.int32))
        scores.extend(info["scores"].tolist())
    return float(np.mean(scores[:alvo]))


# ------------------------------------------------------- estado do ambiente
def test_state_roundtrip_is_exact():
    """A busca inteira depende disto: restaurar um nó tem que devolver o jogo idêntico."""
    env = VecSnake(8, 10, rng=np.random.default_rng(0))
    env.reset()
    for _ in range(30):
        env.step(np.ones(8, np.int32))
    estado = env.get_state()
    obs_a, mask_a = env.obs(), env.action_mask()

    for _ in range(50):
        env.step(np.zeros(8, np.int32))
    env.set_state(estado)
    env.check_invariants()

    assert np.array_equal(env.obs(), obs_a)
    assert np.array_equal(env.action_mask(), mask_a)


def test_restored_state_produces_the_same_transition():
    env = VecSnake(4, 10, rng=np.random.default_rng(1))
    env.reset()
    estado = env.get_state()
    a = np.array([0, 1, 2, 1], np.int32)
    _, _, r1, d1, _ = env.step(a)
    env.set_state(estado)
    _, _, r2, d2, _ = env.step(a)
    assert np.array_equal(r1, r2) and np.array_equal(d1, d2)


# ----------------------------------------------------------------- a busca
def test_run_returns_visit_counts_and_values():
    env = VecSnake(6, 10, rng=np.random.default_rng(0))
    obs, mask = env.reset()
    mcts = MCTS(uniforme, num_simulations=8)
    visitas, valores = mcts.run(env.get_state(), mask, obs)
    assert visitas.shape == (6, N_ACTIONS)
    assert valores.shape == (6,)
    assert (visitas.sum(1) > 0).all()


def test_search_never_visits_a_masked_action():
    """A máscara vale dentro da árvore também — senão a busca planeja morrer."""
    env = VecSnake(32, 10, rng=np.random.default_rng(2))
    obs, mask = env.reset()
    mcts = MCTS(uniforme, num_simulations=16)
    for _ in range(15):
        visitas, _ = mcts.run(env.get_state(), mask, obs)
        assert (visitas[~mask] == 0).all(), "a busca visitou uma ação proibida"
        obs, mask, *_ = env.step(visitas.argmax(1).astype(np.int32))


def test_search_beats_random_with_an_informative_value():
    """O teste que pega bug silencioso na busca.

    Com um valor que sabe onde está a comida, o MCTS tem que jogar MUITO melhor que o
    acaso. A primeira versão deste módulo jogava *pior*, porque o PUCT usava só o valor do
    filho e ignorava a recompensa de chegar até ele — um filho alcançado morrendo tinha
    valor 0 e parecia tão bom quanto um seguro.
    """
    env = VecSnake(32, 10, rng=np.random.default_rng(123))
    obs, mask = env.reset()
    rng = np.random.default_rng(0)
    scores = []
    while len(scores) < 32:
        p = mask.astype(np.float64); p /= p.sum(1, keepdims=True)
        a = (p.cumsum(1) > rng.random((32, 1))).argmax(1).astype(np.int32)
        obs, mask, r, d, info = env.step(a)
        scores.extend(info["scores"].tolist())
    piso = float(np.mean(scores[:32]))

    com_busca = joga(heuristica, sims=12)
    assert com_busca > 5 * max(piso, 0.5), \
        f"busca ({com_busca:.2f}) não superou o acaso ({piso:.2f}) com valor informativo"


def test_more_simulations_do_not_hurt():
    """Mais computação não pode piorar o jogo. Se piorar, o backup está errado."""
    poucas = joga(heuristica, sims=6)
    muitas = joga(heuristica, sims=24)
    assert muitas > poucas * 0.75, f"{muitas:.2f} contra {poucas:.2f} com 4x mais busca"


def test_terminal_children_are_never_expanded():
    """O `VecSnake` reseta sozinho ao terminar.

    Se o nó terminal guardasse o estado devolvido pelo `step`, a árvore teria uma partida
    NOVA e aleatória enxertada dentro dela — e a busca planejaria sobre um jogo que não
    existe. Nó terminal não tem estado nem filhos: vale 0 e acabou.

    Para forçar o caso, colocamos a fome a um passo do limite: qualquer ação termina o
    episódio. Num tabuleiro novo a máscara evita morte por vários níveis de profundidade, e
    o caso nunca apareceria.
    """
    env = VecSnake(8, 10, starve_base=10, rng=np.random.default_rng(5))
    obs, mask = env.reset()
    env.hunger[:] = env.starve_base + 2 * env.length - 1     # o próximo passo é fome
    mcts = MCTS(uniforme, num_simulations=12, starve_base=env.starve_base,
                rng=np.random.default_rng(0))
    mcts.run(env.get_state(), mask, obs)

    def varrer(no, achados):
        for filho in no.filhos.values():
            achados.append(filho)
            varrer(filho, achados)
        return achados

    terminais = 0
    for raiz in mcts._ultimas_raizes:
        for no in varrer(raiz, []):
            if no.terminal:
                terminais += 1
                assert no.estado is None, "nó terminal guardou o estado pós-reset"
                assert not no.filhos, "nó terminal foi expandido"
    assert terminais > 0, "o cenário deveria produzir nós terminais"


def test_search_env_inherits_the_training_env_config():
    """A árvore tem que simular o MESMO jogo. Regra de fome diferente = mundo diferente."""
    mcts = MCTS(uniforme, board_size=10, starve_base=7)
    assert mcts._ambiente(4).starve_base == 7
    padrao = MCTS(uniforme, board_size=10)
    assert padrao._ambiente(4).starve_base == 100


def test_visit_policy_temperature():
    visitas = np.array([[1.0, 9.0, 0.0]])
    quente = MCTS.politica_das_visitas(visitas, 1.0)
    fria = MCTS.politica_das_visitas(visitas, 1e-9)
    assert quente[0, 1] == pytest.approx(0.9)
    assert fria[0, 1] == pytest.approx(1.0)
    assert np.allclose(quente.sum(1), 1.0) and np.allclose(fria.sum(1), 1.0)


def test_dirichlet_noise_changes_the_root_priors():
    env = VecSnake(16, 10, rng=np.random.default_rng(0))
    obs, mask = env.reset()
    limpo = MCTS(uniforme, num_simulations=16, rng=np.random.default_rng(0))
    ruid = MCTS(uniforme, num_simulations=16, rng=np.random.default_rng(0))
    va, _ = limpo.run(env.get_state(), mask, obs, adicionar_ruido=False)
    vb, _ = ruid.run(env.get_state(), mask, obs, adicionar_ruido=True)
    assert not np.array_equal(va, vb)


# ------------------------------------------------------------------ AlphaZero
def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=3, num_simulations=6,
                batch_size=16, memory_size=2000, total_steps=1000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return AlphaZeroConfig(**base)


def test_collect_fills_the_buffer_with_search_targets():
    ag = AlphaZero(cfg_min())
    stats = ag.collect()
    n = ag.cfg.rollout * ag.cfg.num_envs
    assert ag._cheio == n
    assert np.allclose(ag._buf_pi[:n].sum(1), 1.0, atol=1e-5), \
        "o alvo de política tem que ser distribuição"
    assert np.isfinite(ag._buf_z[:n]).all()
    assert "valor_raiz" in stats


def test_policy_target_never_puts_mass_on_masked_actions():
    ag = AlphaZero(cfg_min())
    ag.collect()
    n = ag.cfg.rollout * ag.cfg.num_envs
    assert (ag._buf_pi[:n][~ag._buf_mask[:n]] == 0).all()


def test_official_policy_has_no_search():
    """A curva do contrato mede a rede pura — busca é coluna separada, como o flood-fill."""
    ag = AlphaZero(cfg_min())
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a, b = fn(obs, mask), fn(obs, mask)
    assert np.array_equal(a, b)
    assert (a[~mask] == MASK_NEG).all()


def test_search_evaluation_follows_the_same_protocol():
    ag = AlphaZero(cfg_min())
    st = ag.avaliar_com_busca(episodes=32, num_simulations=4)
    assert st["episodes"] == 32
    assert st["num_simulations"] == 4
    assert 0.0 <= st["score_mean"] <= 97


def test_temperature_decays():
    ag = AlphaZero(cfg_min(total_steps=1000, temp_inicio=1.0, temp_fim=0.25,
                           temp_frac=0.5))
    assert ag.temperatura() == pytest.approx(1.0)
    ag.global_step = 500
    assert ag.temperatura() == pytest.approx(0.25)


def test_iterate_trains_and_reports():
    ag = AlphaZero(cfg_min(batch_size=16))
    antes = [w.numpy().copy() for w in ag.model.trainable_variables]
    ag.iterate()
    stats = ag.iterate()
    depois = [w.numpy() for w in ag.model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))
    assert np.isfinite(stats["perda_pi"]) and np.isfinite(stats["perda_v"])


def test_checkpoint_roundtrip(tmp_path):
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = AlphaZero(cfg); ag.iterate(); ag.salvar("last")
    outro = AlphaZero(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    outro.iterate()
