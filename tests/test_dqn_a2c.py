"""DQN (a família toda) e A2C.

O DQN aqui substitui seis notebooks quase idênticos. Os testes cobrem cada componente do
Rainbow isolado, mais o que o repositório antigo nunca teve: prova de que o alvo de TD está
certo, conferido à mão.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from snakeai.agents import A2C, DQN, A2CConfig, DQNConfig
from snakeai.eval import MASK_NEG

VARIANTES = [
    ("base", {}),
    ("double", {"double": True}),
    ("dueling", {"dueling": True}),
    ("per", {"per": True}),
    ("noisy", {"noisy": True}),
    ("3steps", {"n_steps": 3}),
    ("c51", {"n_atoms": 51}),
    ("rainbow", {"double": True, "dueling": True, "per": True, "noisy": True,
                 "n_steps": 3, "n_atoms": 51}),
]


def dqn_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, batch_size=16, memory_size=2000,
                warmup_steps=0, learn_every=2, total_steps=2000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return DQNConfig(**base)


def a2c_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=4, total_steps=2000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return A2CConfig(**base)


# --------------------------------------------------------------- DQN: família
@pytest.mark.parametrize("nome,kw", VARIANTES, ids=[v[0] for v in VARIANTES])
def test_every_rainbow_variant_trains(nome, kw):
    """Seis notagens quase idênticas do repositório antigo, agora uma linha cada."""
    ag = DQN(dqn_min(**kw))
    for _ in range(4):
        stats = ag.iterate()
    assert ag.global_step > 0
    assert stats["loss"] is not None and np.isfinite(stats["loss"])


def test_variant_name_describes_the_configuration():
    assert DQN(dqn_min()).variant == "base"
    ag = DQN(dqn_min(double=True, per=True, n_steps=3))
    assert ag.variant == "double+per+3steps"


def test_target_network_starts_synchronized_and_lags():
    ag = DQN(dqn_min(target_update=10**9))
    for a, b in zip(ag.model.get_weights(), ag.target.get_weights()):
        assert np.allclose(a, b)
    for _ in range(4):
        ag.iterate()
    difs = [not np.allclose(a, b)
            for a, b in zip(ag.model.get_weights(), ag.target.get_weights())]
    assert any(difs), "a rede alvo deveria estar defasada — é para isso que ela existe"


def test_target_network_syncs_on_schedule():
    ag = DQN(dqn_min(target_update=1))
    for _ in range(3):
        ag.iterate()
    for a, b in zip(ag.model.get_weights(), ag.target.get_weights()):
        assert np.allclose(a, b)


# ------------------------------------------------------------ DQN: alvo de TD
def test_td_target_matches_a_hand_computation():
    """`r + γ^n · max_a Q_alvo(s', a) · (1 − done)`, conferido termo a termo."""
    ag = DQN(dqn_min(n_steps=2, gamma=0.5))
    n = 4
    lote = {
        "rew": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "done": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "next_obs": np.zeros((n, 10, 10, 5), dtype=np.float32),
        "next_mask": np.ones((n, 3), dtype=bool),
    }
    alvo = ag._alvo(lote)
    q_prox = np.asarray(ag._q_valores(ag.target, tf.convert_to_tensor(lote["next_obs"])))
    v = q_prox.max(axis=1)
    g = 0.5 ** 2
    esperado = lote["rew"] + g * v * (1 - lote["done"])
    assert np.allclose(alvo, esperado, atol=1e-5)
    # terminais não carregam bootstrap
    assert alvo[1] == pytest.approx(2.0)
    assert alvo[3] == pytest.approx(4.0)


@pytest.mark.parametrize("double", [False, True])
def test_double_dqn_uses_online_argmax_and_target_value(double):
    """O ponto do Double DQN: quem escolhe não é quem avalia.

    Testado com Q controlado, não com pesos perturbados — perturbar pesos não garante que
    os argmax discordem, e um teste que passa por sorte não prova nada. Aqui a rede online
    prefere a ação 0 e a alvo prefere a 2, com valores escolhidos para que as duas
    fórmulas deem números diferentes e conhecidos.
    """
    ag = DQN(dqn_min(double=double, gamma=1.0, n_steps=1))
    q_online = np.array([[9.0, 1.0, 2.0]], dtype=np.float32)   # argmax = ação 0
    q_alvo = np.array([[0.0, 5.0, 8.0]], dtype=np.float32)     # argmax = ação 2

    def falso(modelo, obs, training=False):
        return q_online if modelo is ag.model else q_alvo

    ag._q_valores = falso
    lote = {
        "rew": np.zeros(1, np.float32), "done": np.zeros(1, np.float32),
        "next_obs": np.zeros((1, 10, 10, 5), np.float32),
        "next_mask": np.ones((1, 3), bool),
    }
    alvo = ag._alvo(lote)
    if double:
        # online escolhe a ação 0; o valor vem da rede alvo -> 0.0
        assert alvo[0] == pytest.approx(0.0)
    else:
        # max da própria rede alvo -> 8.0, o viés otimista que o Double corta
        assert alvo[0] == pytest.approx(8.0)


def test_masked_actions_never_enter_the_target():
    ag = DQN(dqn_min())
    mascara = np.zeros((4, 3), bool)
    mascara[:, 1] = True                      # só a ação 1 é legal
    lote = {
        "rew": np.zeros(4, np.float32), "done": np.zeros(4, np.float32),
        "next_obs": np.random.default_rng(1).normal(size=(4, 10, 10, 5)).astype(np.float32),
        "next_mask": mascara,
    }
    alvo = ag._alvo(lote)
    q = np.asarray(ag._q_valores(ag.target, tf.convert_to_tensor(lote["next_obs"])))
    assert np.allclose(alvo, ag.cfg.gamma * q[:, 1], atol=1e-5)


def test_c51_projection_keeps_the_distribution_normalized():
    """A projeção categórica não pode perder massa — se perder, o alvo está errado."""
    ag = DQN(dqn_min(n_atoms=51))
    lote = {
        "rew": np.array([0.0, 1.0, -1.0, 5.0], np.float32),
        "done": np.array([0.0, 0.0, 1.0, 0.0], np.float32),
        "next_obs": np.zeros((4, 10, 10, 5), np.float32),
        "next_mask": np.ones((4, 3), bool),
    }
    alvo = ag._alvo(lote)
    assert alvo.shape == (4, 51)
    assert np.allclose(alvo.sum(axis=1), 1.0, atol=1e-4), \
        f"massa perdida na projeção: {alvo.sum(axis=1)}"
    assert (alvo >= 0).all()


# ------------------------------------------------------------ DQN: exploração
def test_epsilon_decays_and_is_zero_when_noisy():
    ag = DQN(dqn_min(eps_start=1.0, eps_end=0.02, eps_frac=0.5, total_steps=1000))
    assert ag.epsilon() == pytest.approx(1.0)
    ag.global_step = 500
    assert ag.epsilon() == pytest.approx(0.02)
    ag.global_step = 900
    assert ag.epsilon() == pytest.approx(0.02), "epsilon não pode passar do piso"

    ruidoso = DQN(dqn_min(noisy=True))
    assert ruidoso.epsilon() == 0.0, "com noisy nets a exploração é da rede"


def test_per_beta_rises_to_one():
    ag = DQN(dqn_min(per=True, per_beta0=0.4, total_steps=1000))
    assert ag.beta() == pytest.approx(0.4)
    ag.global_step = 1000
    assert ag.beta() == pytest.approx(1.0)


def test_greedy_policy_has_no_exploration():
    """A política que o benchmark mede é greedy — sem ε, sem ruído, determinística."""
    ag = DQN(dqn_min(noisy=True, eps_start=1.0))
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a = fn(obs, mask)
    b = fn(obs, mask)
    assert np.allclose(a, b)
    assert (np.asarray(a)[~mask] == MASK_NEG).all()


def test_actions_respect_the_mask():
    ag = DQN(dqn_min(eps_start=1.0))     # exploração máxima, o caso mais arriscado
    obs, mask = ag.env.reset()
    for _ in range(20):
        acoes = ag._escolher(obs, mask)
        assert mask[np.arange(len(acoes)), acoes].all()
        obs, mask, *_ = ag.env.step(acoes)


# -------------------------------------------------------------------- A2C
def test_a2c_shares_the_ppo_rollout():
    """Herança direta, não cópia: uma correção no rollout vale para os dois na hora."""
    from snakeai.agents.ppo import PPO
    assert issubclass(A2C, PPO)
    assert A2C.collect is PPO.collect


def test_a2c_trains_and_reports():
    ag = A2C(a2c_min())
    for _ in range(3):
        stats = ag.iterate()
    for chave in ("pg", "vf", "ent", "lr"):
        assert chave in stats and np.isfinite(stats[chave])
    assert stats["epochs_done"] == 1


def test_a2c_does_a_single_pass_over_the_rollout():
    """Sem clipping, reaproveitar o rollout afastaria a política dos dados que a geraram."""
    ag = A2C(a2c_min(epochs=5))          # mesmo pedindo 5, tem que fazer 1
    lote, _ = ag.collect()
    assert ag.update(lote)["epochs_done"] == 1


def test_a2c_has_no_ratio_clipping():
    """O que separa A2C de PPO: gradiente de política puro."""
    import inspect
    fonte = inspect.getsource(A2C._train_step_a2c)
    assert "clip_by_value" not in fonte
    assert "logp * adv" in fonte


def test_a2c_and_ppo_use_the_same_algo_slot_but_different_names():
    assert A2C.algo == "a2c"
    from snakeai.agents.ppo import PPO
    assert PPO.algo == "ppo"


# ------------------------------------------------------------------ integração
def test_dqn_full_train_writes_a_record(tmp_path):
    cfg = dqn_min(total_steps=400, ckpt_dir=str(tmp_path / "ck"),
                  runs_dir=str(tmp_path / "runs"))
    rec = DQN(cfg).train(verbose=False)
    caminho = tmp_path / "runs" / "dqn" / "base" / "seed0" / "history.json"
    assert caminho.exists(), "a curva tem que chegar ao disco"
    assert rec.record.curve
    assert rec.record.meta["contract_violations"], \
        "execução de fumaça não pode passar como oficial"
    assert rec.record.oficial is False


def test_a2c_checkpoint_roundtrip(tmp_path):
    cfg = a2c_min(ckpt_dir=str(tmp_path))
    ag = A2C(cfg); ag.iterate(); ag.salvar("last")
    outro = A2C(a2c_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    assert outro.global_step == ag.global_step
    outro.iterate()


def test_dqn_checkpoint_rebuilds_the_target_network(tmp_path):
    """Sem reconstruir o alvo, o `retomar` devolveria um agente sem rede alvo."""
    cfg = dqn_min(ckpt_dir=str(tmp_path))
    ag = DQN(cfg); ag.iterate(); ag.salvar("last")
    outro = DQN(dqn_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    assert outro.target is not None
    outro.iterate()
