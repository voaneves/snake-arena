"""As três redes do MuZero.

O MuZero substitui o simulador por três funções aprendidas:

======================  ==============================================================
rede                    o que faz
======================  ==============================================================
**representação** `h`   observação → estado oculto `s₀`
**dinâmica** `g`        `(s, a)` → `(s', recompensa)`
**predição** `f`        `s` → `(política, valor)`
======================  ==============================================================

O detalhe que define o algoritmo: **`s` não precisa significar nada**. Não há perda
pedindo que o estado oculto reconstrua a observação. As três redes são treinadas só para
que a busca produza boas jogadas — o modelo aprende o que é útil para planejar, não o que
é fiel ao mundo. É a diferença entre o MuZero e um model-based clássico, e é por isso que
ele funciona em domínios onde reconstruir pixels seria impossível.

Duas peças pequenas que decidem se treina
------------------------------------------
* **Normalização do estado oculto para [0, 1]** (min-max por amostra). Sem isso a escala do
  estado cresce a cada aplicação de `g` e o desenrolar de `K` passos explode.
* **Escala de gradiente de ½ na dinâmica**, aplicada a cada passo do desenrolar. Sem ela o
  gradiente que chega em `h` cresce com `K` e o treino fica instável.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers, ops

from ..env.vec_snake import N_ACTIONS, N_CHANNELS
from .resnet import PRESETS, residual_block

__all__ = ["normaliza_oculto", "escala_gradiente",
           "build_representacao", "build_dinamica", "build_predicao"]


def normaliza_oculto(x):
    """Min-max por amostra, para [0, 1]. Segura a escala ao longo do desenrolar."""
    minimo = ops.min(x, axis=(1, 2, 3), keepdims=True)
    maximo = ops.max(x, axis=(1, 2, 3), keepdims=True)
    return (x - minimo) / ops.maximum(maximo - minimo, 1e-5)


def escala_gradiente(x, escala):
    """Deixa o valor intacto e multiplica o gradiente por `escala`.

    `x·s + stop_gradient(x·(1−s))` — o truque padrão do MuZero para que o gradiente que
    atravessa `K` aplicações da dinâmica não cresça com `K`.
    """
    return x * escala + ops.stop_gradient(x) * (1.0 - escala)


@keras.saving.register_keras_serializable(package="snakeai")
class NormalizaOculto(layers.Layer):
    """Camada em vez de `Lambda`: `Lambda` não sobrevive a `save`/`load` sem gambiarra."""

    def call(self, x):
        return normaliza_oculto(x)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_representacao(board_size=10, preset="resnet_small", nome="h"):
    """`observação → estado oculto`. O estado oculto tem a forma espacial do tabuleiro."""
    largura, blocos = PRESETS[preset]
    inp = keras.Input(shape=(board_size, board_size, N_CHANNELS), name="board")
    x = layers.Conv2D(largura, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{nome}_c")(inp)
    x = layers.GroupNormalization(groups=8, name=f"{nome}_n")(x)
    x = layers.Activation("relu", name=f"{nome}_a")(x)
    for i in range(blocos):
        x = residual_block(x, largura, f"{nome}_res{i}")
    x = NormalizaOculto(name=f"{nome}_norm")(x)
    return keras.Model(inp, x, name="representacao")


def build_dinamica(board_size=10, preset="resnet_small", n_actions=N_ACTIONS,
                   nome="g", n_suporte=0):
    """`(estado, ação) → (estado', recompensa)`.

    A ação entra como **planos constantes** concatenados ao estado — um plano de uns no
    canal da ação escolhida, zeros nos outros. É o encoding do MuZero: mantém a estrutura
    convolucional e não obriga a rede a aprender um embedding.
    """
    largura, blocos = PRESETS[preset]
    s = keras.Input(shape=(board_size, board_size, largura), name="estado")
    a = keras.Input(shape=(board_size, board_size, n_actions), name="acao_planos")

    x = layers.Concatenate(name=f"{nome}_cat")([s, a])
    x = layers.Conv2D(largura, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{nome}_c")(x)
    x = layers.GroupNormalization(groups=8, name=f"{nome}_n")(x)
    x = layers.Activation("relu", name=f"{nome}_a")(x)
    for i in range(max(1, blocos - 1)):
        x = residual_block(x, largura, f"{nome}_res{i}")
    novo = NormalizaOculto(name=f"{nome}_norm")(x)

    r = layers.Conv2D(2, 1, use_bias=False, name=f"{nome}_rc")(x)
    r = layers.GroupNormalization(groups=2, name=f"{nome}_rn")(r)
    r = layers.Activation("relu", name=f"{nome}_ra")(r)
    r = layers.Flatten(name=f"{nome}_rf")(r)
    r = layers.Dense(64, activation="relu", name=f"{nome}_rd")(r)
    # `n_suporte > 0` troca a cabeça escalar por logits sobre um suporte discreto —
    # o Apêndice F do MuZero. A saída deixa de ser um número e passa a ser uma
    # distribuição; quem lê converte por esperança. Ver §2.33.
    recompensa = layers.Dense(max(1, n_suporte), name="recompensa")(r)

    return keras.Model([s, a], [novo, recompensa], name="dinamica")


def build_predicao(board_size=10, preset="resnet_small", n_actions=N_ACTIONS,
                   nome="f", n_suporte=0):
    """`estado oculto → (logits de política, valor)`."""
    largura, _ = PRESETS[preset]
    s = keras.Input(shape=(board_size, board_size, largura), name="estado")

    p = layers.Conv2D(4, 1, use_bias=False, name=f"{nome}_pc")(s)
    p = layers.GroupNormalization(groups=2, name=f"{nome}_pn")(p)
    p = layers.Activation("relu", name=f"{nome}_pa")(p)
    p = layers.Flatten(name=f"{nome}_pf")(p)
    logits = layers.Dense(
        n_actions, name="logits",
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01))(p)

    v = layers.Conv2D(2, 1, use_bias=False, name=f"{nome}_vc")(s)
    v = layers.GroupNormalization(groups=2, name=f"{nome}_vn")(v)
    v = layers.Activation("relu", name=f"{nome}_va")(v)
    v = layers.Flatten(name=f"{nome}_vf")(v)
    v = layers.Dense(128, activation="relu", name=f"{nome}_vd")(v)
    valor = layers.Dense(max(1, n_suporte), name="valor")(v)

    return keras.Model(s, [logits, valor], name="predicao")
