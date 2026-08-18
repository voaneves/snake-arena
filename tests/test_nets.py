"""As redes.

Testam o contrato de forma (entrada, saída, número de ações) e as propriedades que fazem
a comparação entre arquiteturas ser honesta: mesmo tronco, mesmo formato de entrada, saída
compatível com o mesmo agente.

Alguns testes documentam defeitos do repositório antigo em vez de proibi-los — os troncos
com pooling continuam colapsando o tabuleiro porque é isso que eles faziam, e a curva
histórica só faz sentido se a rede for a mesma. O teste existe para que o defeito seja
visível e intencional, nunca uma surpresa.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest

from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS, VecSnake
from snakeai.nets import (
    APELIDOS_LEGADOS,
    TRONCOS,
    build_actor_critic,
    build_q_network,
    listar_troncos,
    resumo,
)
from snakeai.nets.registry import LARGURA_DENSA_LEGADA, _resolve

TODOS = sorted(TRONCOS)


# ------------------------------------------------------------------- registro
def test_every_trunk_is_reachable_by_name():
    for nome in TODOS:
        fn, canonico = _resolve(nome)
        assert canonico == nome


def test_legacy_aliases_point_to_the_notebook_definitions():
    """O apelido tem que apontar para a rede que de fato rodou, não para a homônima."""
    assert APELIDOS_LEGADOS["cnn1"] == "cnn_rainbow"
    assert APELIDOS_LEGADOS["cnn2"] == "cnn_alphazero"
    assert APELIDOS_LEGADOS["cnn3"] == "cnn_vgg"
    assert APELIDOS_LEGADOS["cnn4"] == "cnn_vgg_dropout"
    for apelido, canonico in APELIDOS_LEGADOS.items():
        assert _resolve(apelido)[1] == canonico


def test_unknown_trunk_raises_with_the_list():
    with pytest.raises(ValueError, match="resnet_small"):
        build_actor_critic(net="cnn_que_nao_existe")


def test_listar_troncos_includes_aliases():
    nomes = listar_troncos()
    assert "resnet_small" in nomes and "cnn2" in nomes


# ------------------------------------------------------------- forma e contrato
@pytest.mark.parametrize("net", TODOS)
def test_actor_critic_shapes(net):
    m = build_actor_critic(board_size=10, net=net)
    x = np.zeros((4, 10, 10, N_CHANNELS), dtype=np.float32)
    logits, valor = m(x, training=False)
    assert tuple(logits.shape) == (4, N_ACTIONS)
    assert tuple(valor.shape) == (4, 1)
    assert np.isfinite(np.asarray(logits)).all()


@pytest.mark.parametrize("net", TODOS)
def test_q_network_shapes(net):
    m = build_q_network(board_size=10, net=net)
    x = np.zeros((4, 10, 10, N_CHANNELS), dtype=np.float32)
    q = m(x, training=False)
    assert tuple(q.shape) == (4, N_ACTIONS)


def test_input_matches_the_environment_observation():
    """A rede tem que aceitar exatamente o que o `VecSnake` produz, sem adaptador."""
    env = VecSnake(8, 10, rng=np.random.default_rng(0))
    obs, _ = env.reset()
    m = build_actor_critic(net="resnet_small")
    logits, valor = m(obs, training=False)
    assert tuple(logits.shape) == (8, N_ACTIONS)


def test_board_size_is_honored():
    m = build_actor_critic(board_size=12, net="resnet_tiny")
    assert tuple(m.input.shape[1:]) == (12, 12, N_CHANNELS)


# ------------------------------------------------------------ o defeito herdado
def test_pooling_trunks_collapse_the_board():
    """Documenta o defeito, não o esconde: 10 → 5 → 2 → 1 com três poolings.

    O tronco entrega 64 números sem nenhuma informação de posição. Se este teste um dia
    falhar, alguém "consertou" a rede — e aí as curvas históricas deixaram de ser
    comparáveis com ela.
    """
    for net in ("cnn_vgg", "cnn_vgg_dropout"):
        inp = keras.Input(shape=(10, 10, N_CHANNELS))
        from snakeai.nets.registry import build_backbone
        saida, _ = build_backbone(inp, net)
        assert tuple(saida.shape[1:]) == (64,), f"{net} deveria colapsar para 64"


def test_the_no_pool_ablation_keeps_the_board():
    from snakeai.nets.registry import build_backbone
    inp = keras.Input(shape=(10, 10, N_CHANNELS))
    saida, _ = build_backbone(inp, "cnn_vgg_sem_pool")
    assert tuple(saida.shape[1:]) == (6400,), "10×10×64 achatado"


def test_resnets_keep_the_spatial_map():
    linhas = {d["tronco"]: d for d in resumo()}
    for nome in ("resnet_tiny", "resnet_small", "resnet_base"):
        assert linhas[nome]["espacial"] is True
        assert linhas[nome]["saida_tronco"].startswith("10×10×")


def test_legacy_dense_width_is_expensive_and_optional():
    """3136 unidades era o padrão antigo. Medir o custo é melhor que argumentar."""
    barato = build_actor_critic(net="cnn_rainbow").count_params()
    caro = build_actor_critic(net="cnn_rainbow",
                             largura_densa=LARGURA_DENSA_LEGADA).count_params()
    assert caro > 5 * barato


# --------------------------------------------------------------- treinabilidade
@pytest.mark.parametrize("net", ["resnet_tiny", "cnn_rainbow", "cnn_vgg"])
def test_gradients_flow_through_every_trunk(net):
    """Uma rede que não recebe gradiente é um experimento perdido silenciosamente."""
    import tensorflow as tf

    m = build_actor_critic(net=net)
    x = np.random.default_rng(0).normal(size=(8, 10, 10, N_CHANNELS)).astype(np.float32)
    with tf.GradientTape() as tape:
        logits, valor = m(x, training=True)
        perda = tf.reduce_mean(logits ** 2) + tf.reduce_mean(valor ** 2)
    grads = tape.gradient(perda, m.trainable_variables)
    assert all(g is not None for g in grads), "alguma variável ficou desconectada"
    assert any(float(tf.reduce_max(tf.abs(g))) > 0 for g in grads)


def test_policy_head_starts_near_uniform():
    """Ganho pequeno no inicializador: o PPO não deve gastar iterações desfazendo viés."""
    m = build_actor_critic(net="resnet_small")
    x = np.random.default_rng(1).normal(size=(64, 10, 10, N_CHANNELS)).astype(np.float32)
    logits = np.asarray(m(x, training=False)[0])
    espalhamento = float(np.abs(logits - logits.mean(axis=1, keepdims=True)).max())
    assert espalhamento < 0.5, f"política inicial já opinativa demais: {espalhamento:.3f}"


def test_models_are_saveable_in_the_keras_3_format(tmp_path):
    """`.keras`, não `.h5` — a nota de porte do README, verificada."""
    m = build_actor_critic(net="resnet_tiny")
    caminho = tmp_path / "m.keras"
    m.save(caminho)
    lido = keras.models.load_model(caminho)
    x = np.zeros((2, 10, 10, N_CHANNELS), dtype=np.float32)
    a = np.asarray(m(x, training=False)[0])
    b = np.asarray(lido(x, training=False)[0])
    assert np.allclose(a, b)


def test_channels_last_everywhere():
    """O porte para Keras 3: nada de `set_image_dim_ordering('th')`."""
    assert keras.backend.image_data_format() == "channels_last"


# ---------------------------------------------------------------------- resumo
def test_resumo_covers_every_trunk():
    linhas = resumo()
    assert {d["tronco"] for d in linhas} == set(TODOS)
    for d in linhas:
        assert d["params_tronco"] > 0
        assert d["params_actor_critic"] > d["params_tronco"]


# ---------------------------------------------------------------- exportação
def test_export_reads_the_channel_count_from_the_network():
    """O exportador não pode assumir os 5 canais do contrato: uma execução com
    `canal_fome=True` treina uma rede de 6, e a constante quebrava a medição de latência
    — na última célula do notebook, depois do treino inteiro."""
    from snakeai.export import canais_do_modelo, medir_latencia

    m5 = build_actor_critic(net="resnet_tiny")
    m6 = build_actor_critic(net="resnet_tiny", canais=6)
    assert canais_do_modelo(m5) == N_CHANNELS == 5
    assert canais_do_modelo(m6) == 6

    # a medição alimenta a rede de verdade: com o canal errado, isto levantaria ValueError
    medir_latencia(lambda x: m6(x, training=False), repeticoes=2, aquecimento=1,
                   canais=canais_do_modelo(m6))
