"""MuZero.

O teste central deste arquivo é `test_muzero_and_alphazero_share_the_same_search`: se a
busca não for literalmente o mesmo objeto, a comparação entre os dois deixa de medir "o que
custa não ter o simulador" e passa a medir duas implementações diferentes de MCTS.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from snakeai.agents import AlphaZero, AlphaZeroConfig, MuZero, MuZeroConfig
from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS, VecSnake
from snakeai.eval import MASK_NEG
from snakeai.nets.muzero import (
    build_dinamica,
    build_predicao,
    build_representacao,
    escala_gradiente,
    normaliza_oculto,
)
from snakeai.search import MCTS, DinamicaAprendida, DinamicaReal

LARGURA_TINY = 32


def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=6, unroll=2, num_simulations=4,
                batch_size=16, memory_size=2000, total_steps=1000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return MuZeroConfig(**base)


# ------------------------------------------------------------------- as redes
def test_the_three_networks_have_the_right_shapes():
    h = build_representacao(10, "resnet_tiny")
    g = build_dinamica(10, "resnet_tiny")
    f = build_predicao(10, "resnet_tiny")

    obs = np.zeros((4, 10, 10, N_CHANNELS), np.float32)
    s = np.asarray(h(obs, training=False))
    assert s.shape == (4, 10, 10, LARGURA_TINY)

    # os dois tensores da dinâmica têm que ser do MESMO tipo: o Keras 3 recusa uma lista
    # que mistura tf.Tensor e ndarray com "you cannot mix tensors and non-tensors"
    planos = np.zeros((4, 10, 10, N_ACTIONS), np.float32)
    s2, r = g([s, planos], training=False)
    assert tuple(s2.shape) == tuple(s.shape)
    assert tuple(r.shape) == (4, 1)

    logits, v = f(s, training=False)
    assert tuple(logits.shape) == (4, N_ACTIONS) and tuple(v.shape) == (4, 1)


def test_hidden_state_is_normalized_to_zero_one():
    """Sem isso a escala do estado cresce a cada `g` e o desenrolar de K passos explode."""
    x = np.random.default_rng(0).normal(0, 50, size=(4, 10, 10, 8)).astype(np.float32)
    y = np.asarray(normaliza_oculto(ops.convert_to_tensor(x)))
    assert y.min() >= -1e-5 and y.max() <= 1 + 1e-5
    for i in range(4):
        assert y[i].min() == pytest.approx(0.0, abs=1e-4)
        assert y[i].max() == pytest.approx(1.0, abs=1e-4)


def test_representation_and_dynamics_stay_bounded_over_a_long_unroll():
    h = build_representacao(10, "resnet_tiny")
    g = build_dinamica(10, "resnet_tiny")
    s = np.asarray(h(np.zeros((2, 10, 10, N_CHANNELS), np.float32), training=False))
    planos = np.zeros((2, 10, 10, N_ACTIONS), np.float32)
    for _ in range(20):
        s, _ = g([s, planos], training=False)
        s = np.asarray(s)
    assert np.isfinite(s).all() and s.max() <= 1 + 1e-4


def test_gradient_scale_keeps_the_value_and_shrinks_the_gradient():
    x = tf.Variable([2.0])
    with tf.GradientTape() as t:
        y = escala_gradiente(x * 1.0, 0.5)
    assert float(y[0]) == pytest.approx(2.0)
    assert float(t.gradient(y, x)[0]) == pytest.approx(0.5)


# ------------------------------------------------------------------ a dinâmica
def test_muzero_and_alphazero_share_the_same_search():
    """A afirmação arquitetural do repositório, verificada.

    Se a busca não for o MESMO objeto, a diferença entre os dois algoritmos deixa de medir
    "quanto custa não ter o simulador" e passa a medir duas implementações de MCTS.
    """
    az = AlphaZero(AlphaZeroConfig(net="resnet_tiny", num_envs=4, rollout=2,
                                   num_simulations=3, batch_size=8,
                                   salvar_gif=False, salvar_grafico=False))
    mz = MuZero(cfg_min(num_envs=4, rollout=2, num_simulations=3))
    assert type(az.mcts) is type(mz.mcts) is MCTS
    assert isinstance(az.mcts.dinamica, DinamicaReal)
    assert isinstance(mz.mcts.dinamica, DinamicaAprendida)
    # e o código da busca é o mesmo método, não uma cópia
    assert az.mcts.run.__func__ is mz.mcts.run.__func__


def test_real_dynamics_matches_the_environment():
    din = DinamicaReal(10)
    env = VecSnake(4, 10, rng=np.random.default_rng(0))
    env.reset()
    estado = env.get_state()
    a = np.array([0, 1, 2, 1], np.int32)
    _, obs, mask, r, d = din.passo(estado, a)
    env.set_state(estado)
    obs2, mask2, r2, d2, _ = env.step(a)
    assert np.array_equal(r, r2) and np.array_equal(d, d2)


def test_learned_dynamics_reports_no_terminal_and_no_mask():
    """Documenta a limitação: o modelo não prevê fim de episódio nem ação ilegal."""
    mz = MuZero(cfg_min())
    din = mz.mcts.dinamica
    assert din.usa_mascara is False
    s = mz.h(np.zeros((3, 10, 10, N_CHANNELS), np.float32), training=False).numpy()
    novo, obs, mask, r, d = din.passo(s, np.array([0, 1, 2], np.int32))
    assert novo.shape == s.shape
    assert mask.all() and not d.any()


# ------------------------------------------------------------------ o agente
def test_collect_stores_the_unroll_targets_aligned():
    """Cada amostra guarda os K passos que vêm DEPOIS dela — desalinhar isso ensina errado."""
    cfg = cfg_min(rollout=6, unroll=2, num_envs=4)
    ag = MuZero(cfg)
    ag.collect()
    n = ag._cheio
    assert n == (cfg.rollout - cfg.unroll) * cfg.num_envs
    assert ag._buf_pi[:n].shape[1] == cfg.unroll + 1
    assert ag._buf_act[:n].shape[1] == cfg.unroll
    assert ag._buf_r[:n].shape[1] == cfg.unroll
    assert np.allclose(ag._buf_pi[:n].sum(-1), 1.0, atol=1e-4)


def test_training_reports_the_three_losses():
    ag = MuZero(cfg_min(batch_size=16))
    ag.iterate()
    stats = ag.iterate()
    for chave in ("perda_pi", "perda_v", "perda_r"):
        assert chave in stats and np.isfinite(stats[chave])


def test_reward_loss_anchors_the_model_to_the_world():
    """A perda de recompensa é a única âncora: sem ela a dinâmica inventa a física."""
    ag = MuZero(cfg_min(batch_size=16))
    for _ in range(4):
        stats = ag.iterate()
    assert stats["perda_r"] >= 0
    # a rede de dinâmica recebe gradiente
    antes = [w.numpy().copy() for w in ag.g.trainable_variables]
    ag.iterate()
    depois = [w.numpy() for w in ag.g.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))


def test_representation_receives_gradient_through_the_unroll():
    ag = MuZero(cfg_min(batch_size=16, unroll=3))
    ag.iterate()
    antes = [w.numpy().copy() for w in ag.h.trainable_variables]
    ag.iterate()
    depois = [w.numpy() for w in ag.h.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))


def test_official_policy_has_no_search():
    ag = MuZero(cfg_min())
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a, b = fn(obs, mask), fn(obs, mask)
    assert np.array_equal(a, b)
    assert (a[~mask] == MASK_NEG).all()


def test_search_runs_over_hidden_states():
    ag = MuZero(cfg_min(num_simulations=6))
    obs, mask = ag.env.reset()
    visitas, valores = ag._busca(obs, mask)
    assert visitas.shape == (ag.cfg.num_envs, N_ACTIONS)
    assert (visitas.sum(1) > 0).all()
    assert (visitas[~mask] == 0).all(), "a máscara vale na raiz, onde o estado é real"


def test_temperature_decays():
    ag = MuZero(cfg_min(total_steps=1000, temp_inicio=1.0, temp_fim=0.25, temp_frac=0.5))
    assert ag.temperatura() == pytest.approx(1.0)
    ag.global_step = 500
    assert ag.temperatura() == pytest.approx(0.25)


def test_checkpoint_saves_all_three_networks(tmp_path):
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = MuZero(cfg)
    ag.iterate()
    ag.salvar("last")
    for nome in ("h", "g", "f"):
        assert (tmp_path / f"muzero_last_{nome}.keras").exists()

    outro = MuZero(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    x = np.zeros((2, 10, 10, N_CHANNELS), np.float32)
    assert np.allclose(np.asarray(ag.h(x, training=False)),
                       np.asarray(outro.h(x, training=False)), atol=1e-5)
    outro.iterate()
