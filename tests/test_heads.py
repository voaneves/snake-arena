"""As cabeças: dueling, noisy e C51.

São os componentes que separam o DQN base do Rainbow. Os testes verificam as propriedades
matemáticas que, se quebradas, fazem o algoritmo treinar sem erro e aprender errado — o
pior tipo de bug em RL.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
from keras import layers, ops

from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS
from snakeai.nets import (
    NoisyDense,
    build_q_network,
    distributional_head,
    dueling_head,
    q_de_distribuicao,
    suporte_c51,
)

X = np.zeros((6, 10, 10, N_CHANNELS), dtype=np.float32)


# ------------------------------------------------------------------- NoisyDense
def test_noisy_dense_shapes_and_weights():
    camada = NoisyDense(7)
    y = camada(np.zeros((4, 5), dtype=np.float32))
    assert tuple(y.shape) == (4, 7)
    nomes = {w.name for w in camada.weights}
    assert nomes == {"w_mu", "w_sigma", "b_mu", "b_sigma"}


def test_noisy_dense_is_deterministic_at_inference():
    """A regra que protege o benchmark: sem ruído quando `training=False`.

    Se a rede sorteasse ruído na avaliação, o mesmo modelo daria números diferentes a cada
    execução e a comparação entre algoritmos perderia o sentido.
    """
    camada = NoisyDense(4, seed=0)
    x = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    a = np.asarray(camada(x, training=False))
    b = np.asarray(camada(x, training=False))
    assert np.array_equal(a, b)


def test_noisy_dense_actually_injects_noise_during_training():
    camada = NoisyDense(4, seed=0)
    x = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    a = np.asarray(camada(x, training=True))
    b = np.asarray(camada(x, training=True))
    assert not np.allclose(a, b), "sem ruído, a NoisyDense é só uma Dense cara"


def test_noisy_dense_training_output_is_centered_on_the_inference_output():
    """O ruído perturba em torno de μ; a média de muitas amostras volta para μ."""
    camada = NoisyDense(4, seed=1)
    x = np.random.default_rng(2).normal(size=(16, 6)).astype(np.float32)
    limpo = np.asarray(camada(x, training=False))
    amostras = np.stack([np.asarray(camada(x, training=True)) for _ in range(120)])
    assert np.allclose(amostras.mean(0), limpo, atol=0.12)


def test_noisy_dense_sigma_scales_with_input_size():
    pequena = NoisyDense(4); pequena(np.zeros((1, 4), dtype=np.float32))
    grande = NoisyDense(4); grande(np.zeros((1, 400), dtype=np.float32))
    assert pequena.ruido_medio() > grande.ruido_medio()


def test_noisy_dense_receives_gradient():
    import tensorflow as tf

    camada = NoisyDense(4, seed=0)
    x = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    with tf.GradientTape() as tape:
        perda = tf.reduce_mean(camada(x, training=True) ** 2)
    grads = tape.gradient(perda, camada.trainable_variables)
    assert all(g is not None for g in grads)
    # o sigma também precisa aprender, senão a exploração nunca diminui
    por_nome = {w.name: g for w, g in zip(camada.trainable_variables, grads)}
    assert float(tf.reduce_max(tf.abs(por_nome["w_sigma"]))) > 0


def test_noisy_dense_survives_save_and_load(tmp_path):
    inp = keras.Input(shape=(6,))
    m = keras.Model(inp, NoisyDense(4, seed=0)(inp))
    caminho = tmp_path / "noisy.keras"
    m.save(caminho)
    lido = keras.models.load_model(caminho)
    x = np.random.default_rng(0).normal(size=(3, 6)).astype(np.float32)
    assert np.allclose(np.asarray(m(x, training=False)),
                       np.asarray(lido(x, training=False)))


# ---------------------------------------------------------------------- dueling
def test_dueling_advantage_is_centered():
    """`A − média(A)` é o que torna a decomposição identificável.

    Sem a centralização, somar uma constante a V e subtraí-la de A daria o mesmo Q, e as
    duas correntes poderiam derivar sem que a perda percebesse.
    """
    inp = keras.Input(shape=(16,))
    modelo = keras.Model(inp, dueling_head(inp, N_ACTIONS, largura=8))
    camada_a = modelo.get_layer("dueling_center")
    x = np.random.default_rng(0).normal(size=(32, 16)).astype(np.float32)
    saida_centrada = np.asarray(
        keras.Model(modelo.input, camada_a.output)(x, training=False)
    )
    assert np.allclose(saida_centrada.mean(axis=-1), 0.0, atol=1e-5)


def test_dueling_and_plain_q_have_the_same_output_shape():
    plano = build_q_network(net="resnet_tiny")
    duel = build_q_network(net="resnet_tiny", dueling=True)
    assert tuple(plano(X, training=False).shape) == tuple(duel(X, training=False).shape)


def test_dueling_costs_parameters():
    plano = build_q_network(net="resnet_tiny").count_params()
    duel = build_q_network(net="resnet_tiny", dueling=True).count_params()
    assert duel > plano, "duas correntes deveriam custar mais que uma"


# ------------------------------------------------------------------------- C51
@pytest.mark.parametrize("dueling", [False, True])
def test_distributional_output_shape(dueling):
    m = build_q_network(net="resnet_tiny", n_atoms=51, dueling=dueling)
    assert tuple(m(X, training=False).shape) == (6, N_ACTIONS, 51)


def test_softmax_over_atoms_sums_to_one():
    m = build_q_network(net="resnet_tiny", n_atoms=31)
    logits = np.asarray(m(X, training=False))
    p = np.asarray(ops.softmax(logits, axis=-1))
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-5)


def test_q_from_distribution_is_the_expected_value():
    suporte = suporte_c51(-5, 5, 11)
    # distribuição concentrada no último átomo -> Q deve ser ~ v_max
    logits = np.full((2, 3, 11), -20.0, dtype=np.float32)
    logits[..., -1] = 20.0
    q = np.asarray(q_de_distribuicao(logits, suporte))
    assert np.allclose(q, 5.0, atol=1e-3)


def test_q_from_distribution_matches_manual_expectation():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 3, 9)).astype(np.float32)
    suporte = suporte_c51(-2, 8, 9)
    p = np.asarray(ops.softmax(logits, axis=-1))
    esperado = (p * suporte).sum(axis=-1)
    assert np.allclose(np.asarray(q_de_distribuicao(logits, suporte)), esperado, atol=1e-5)


def test_support_is_evenly_spaced_and_bounded():
    s = suporte_c51(-2.0, 60.0, 51)
    assert s.shape == (51,)
    assert s[0] == pytest.approx(-2.0) and s[-1] == pytest.approx(60.0)
    assert np.allclose(np.diff(s), np.diff(s)[0])


# ------------------------------------------------------------------ combinações
@pytest.mark.parametrize("kw", [
    {},
    {"dueling": True},
    {"noisy": True},
    {"dueling": True, "noisy": True},
    {"n_atoms": 51},
    {"n_atoms": 51, "dueling": True, "noisy": True},
])
def test_every_dqn_variant_builds_and_trains(kw):
    """Os seis notebooks quase idênticos do repositório antigo, agora seis chamadas."""
    import tensorflow as tf

    m = build_q_network(net="resnet_tiny", **kw)
    with tf.GradientTape() as tape:
        perda = tf.reduce_mean(m(X, training=True) ** 2)
    grads = tape.gradient(perda, m.trainable_variables)
    assert all(g is not None for g in grads)


def test_variant_name_records_what_is_on():
    m = build_q_network(net="resnet_tiny", dueling=True, noisy=True, n_atoms=51)
    for parte in ("dueling", "noisy", "c51x51"):
        assert parte in m.name


def test_noisy_network_is_deterministic_end_to_end_at_inference():
    """A propriedade que o protocolo de avaliação depende, testada no modelo inteiro."""
    m = build_q_network(net="resnet_tiny", noisy=True, dueling=True)
    a = np.asarray(m(X, training=False))
    b = np.asarray(m(X, training=False))
    assert np.array_equal(a, b)


# ------------------------------------------------- o checkpoint volta do disco?
@pytest.mark.parametrize("kw", [
    dict(dueling=True, noisy=True, n_atoms=121),    # Rainbow
    dict(dueling=True, noisy=False, n_atoms=0),     # DQN + dueling
    dict(dueling=True, noisy=True, n_atoms=0),      # dueling + noisy
    dict(dueling=False, noisy=True, n_atoms=121),   # C51 sem dueling
    dict(dueling=False, noisy=False, n_atoms=0),    # DQN base
], ids=["rainbow", "dqn+dueling", "dueling+noisy", "c51", "dqn_base"])
def test_the_checkpoint_survives_a_round_trip_through_disk(tmp_path, kw):
    """`load_model` sem `safe_mode=False` — exatamente como `AgentBase.modelo_melhor()`.

    Este teste existe por causa de uma execução perdida. As cabeças `dueling` e C51 usavam
    `layers.Lambda(lambda t: t - mean(t))`, e o Keras 3 recusa desserializar um lambda
    Python: `ValueError: Requested the deserialization of a Lambda layer...`. Como o
    recarregamento só acontece em `avaliar_melhor()`, **no fim do treino**, o erro chegava
    depois do orçamento inteiro gasto — 8.931 s de GPU numa execução do Rainbow.

    Duas coisas escondiam o defeito: o DQN base não liga `dueling`, e a cabeça C51 só usa
    `Lambda` no ramo dueling. O Rainbow é o primeiro agente com os dois ligados, e por isso
    foi o primeiro a bater. Qualquer ablação de DQN com `dueling=True` teria batido também.

    A correção é `CentraNaMedia`, camada registrada — a mesma solução que
    `snakeai/nets/muzero.py` já usava, com o comentário certo, no arquivo errado.
    """
    m = build_q_network(net="resnet_tiny", **kw)
    antes = np.asarray(m(X, training=False))
    caminho = str(tmp_path / "best.keras")
    m.save(caminho)
    recarregado = keras.models.load_model(caminho)      # sem safe_mode=False, de propósito
    depois = np.asarray(recarregado(X, training=False))
    np.testing.assert_allclose(antes, depois, atol=1e-5)


def test_no_lambda_layers_in_the_heads():
    """A regra, não só o sintoma: `Lambda` com função anônima não é serializável.

    Falhar aqui é mais barato que falhar no fim de um treino de 5 M passos.
    """
    for kw in (dict(dueling=True, noisy=True, n_atoms=121), dict(dueling=True)):
        m = build_q_network(net="resnet_tiny", **kw)
        culpadas = [c.name for c in m.layers if isinstance(c, layers.Lambda)]
        assert not culpadas, f"camadas Lambda em {m.name}: {culpadas}"
