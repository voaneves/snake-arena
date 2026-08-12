"""As peças do DreamerV3 — RSSM, cabeças em two-hot, e as transformações que o fazem
funcionar sem ajuste de hiperparâmetro por ambiente.

O DreamerV3 (Hafner et al., 2023) aprende um **modelo do mundo** e treina o ator dentro
dele: o ambiente real só é usado para coletar dados e para medir. Isso o coloca numa
família diferente do AlphaZero e do MuZero — os dois planejam com busca no momento de agir,
o Dreamer não busca nada em tempo de inferência, ele **treina** dentro de um sonho.

O modelo, em quatro linhas
--------------------------
O estado latente é um par `(h, z)`: `h` é determinístico e recorrente, `z` é estocástico e
**categórico**::

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])          recorrência
    ẑ_t ~ p(z | h_t)                                 prior — o que o modelo prevê sozinho
    z_t ~ q(z | h_t, enc(o_t))                       posterior — corrigido pela observação
    ô_t, r̂_t, ĉ_t = decode(h_t, z_t)                reconstrução, recompensa, continuação

Treinar o modelo é aproximar o prior do posterior (para que ele saiba prever sem ver) e o
posterior do prior (para que não invente detalhe irrelevante) — com pesos diferentes, que é
o *KL balancing*.

Por que `z` é categórico e não gaussiano
----------------------------------------
Num Snake, o que o próximo quadro tem de imprevisível é **discreto**: onde a comida vai
reaparecer. Um latente gaussiano tem que espalhar densidade entre as possibilidades e
acaba prevendo a média — comida borrada no meio do tabuleiro. Um latente categórico
representa "uma das K opções" nativamente, e o gradiente passa por *straight-through*.

As três transformações que substituem ajuste manual
---------------------------------------------------
**symlog.** `symlog(x) = sign(x)·log(1+|x|)`. Recompensas e valores em Snake vivem em
escalas muito diferentes ao longo do treino: no começo o retorno é ~0, no fim é ~50. Prever
`symlog(v)` em vez de `v` faz a mesma rede e o mesmo learning rate servirem nos dois
regimes. É invertível, então nada de informação se perde.

**two-hot.** Em vez de regressão escalar, a recompensa e o valor são previstos como
distribuição sobre uma grade fixa de *bins*, com o alvo distribuído entre os dois bins
vizinhos ao valor real. Regressão de valor em RL sofre com alvos que mudam de escala; a
classificação, não. É o mesmo motivo do C51 no Rainbow — e aqui aparece de novo, o que é
um argumento a favor da ideia e não uma coincidência.

**unimix.** Toda categórica recebe 1% de uniforme misturada. Sem isso uma classe pode
chegar a probabilidade zero, o log vira `-inf` e a KL explode. Um por cento é barato e
elimina a classe inteira de NaN.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from ..env.vec_snake import N_ACTIONS, N_CHANNELS

__all__ = [
    "symlog", "symexp", "two_hot", "de_two_hot", "bins_simetricos",
    "unimix", "amostra_straight_through",
    "build_encoder", "build_decoder", "build_rssm_prior", "build_rssm_post",
    "build_cabecas", "build_ator", "build_critico", "CelulaRecorrente",
    "PRESETS_DREAMER",
]

#: Tamanhos, do menor ao maior. `deter` é a largura de `h`; `grupos × classes` é `z`.
PRESETS_DREAMER = {
    "dreamer_tiny": dict(deter=128, grupos=8, classes=8, largura=128, canais=24),
    "dreamer_small": dict(deter=256, grupos=16, classes=16, largura=256, canais=32),
    "dreamer_base": dict(deter=512, grupos=32, classes=32, largura=384, canais=48),
}


# ------------------------------------------------------------------- symlog
def symlog(x):
    return tf.sign(x) * tf.math.log1p(tf.abs(x))


def symexp(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1.0)


def bins_simetricos(n=41, limite=20.0):
    """Grade de bins no espaço de `symlog`. `symexp(20) ≈ 4,9×10⁸` — teto de sobra."""
    return tf.linspace(-limite, limite, n)


def two_hot(x, bins):
    """Distribui `x` entre os dois bins vizinhos, com peso proporcional à proximidade.

    O alvo de um valor exatamente sobre um bin é one-hot naquele bin; entre dois bins, a
    massa se reparte linearmente. É isso que permite representar valores contínuos numa
    saída de classificação **sem perder resolução** — e sem o alvo de regressão, cuja
    escala muda ao longo do treino.
    """
    x = tf.clip_by_value(x, bins[0], bins[-1])
    n = tf.shape(bins)[0]
    abaixo = tf.reduce_sum(tf.cast(bins[None, :] <= x[:, None], tf.int32), axis=-1) - 1
    abaixo = tf.clip_by_value(abaixo, 0, n - 2)
    acima = abaixo + 1
    b_lo = tf.gather(bins, abaixo)
    b_hi = tf.gather(bins, acima)
    peso_hi = (x - b_lo) / tf.maximum(b_hi - b_lo, 1e-8)
    return (tf.one_hot(abaixo, n) * (1.0 - peso_hi)[:, None]
            + tf.one_hot(acima, n) * peso_hi[:, None])


def de_two_hot(logits, bins):
    """Volta da distribuição ao escalar: média sob a distribuição prevista."""
    return tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * bins, axis=-1)


# --------------------------------------------------------------- categóricas
def unimix(logits, grupos, classes, mistura=0.01):
    """Mistura `mistura` de uniforme. Devolve **logits**, para seguir compondo."""
    cabeca = tf.shape(logits)[:-1]
    p = tf.nn.softmax(tf.reshape(logits, tf.concat([cabeca, [grupos, classes]], 0)),
                      axis=-1)
    p = (1.0 - mistura) * p + mistura / tf.cast(classes, p.dtype)
    # devolve achatado: `unimix` é um filtro sobre logits, então tem que devolver a mesma
    # forma que recebeu — senão quem compõe com ele reshape duas vezes
    return tf.reshape(tf.math.log(p), tf.concat([cabeca, [grupos * classes]], 0))


def amostra_straight_through(logits, grupos, classes, seed=None):
    """Amostra one-hot com gradiente reto: `z = onehot + (p - stop_grad(p))`.

    A amostragem é discreta — não tem derivada. O truque devolve, no forward, exatamente
    a amostra; e, no backward, a derivada de `p`. Sem ele o gradiente não atravessa o
    latente e o encoder nunca aprende.
    """
    forma = tf.concat([tf.shape(logits)[:-1], [grupos, classes]], axis=0)
    lg = tf.reshape(logits, forma)
    p = tf.nn.softmax(lg, axis=-1)
    plano = tf.reshape(lg, [-1, classes])
    idx = tf.random.categorical(plano, 1, seed=seed)[:, 0]
    amostra = tf.reshape(tf.one_hot(idx, classes), forma)
    z = tf.stop_gradient(amostra - p) + p
    return tf.reshape(z, tf.concat([tf.shape(logits)[:-1], [grupos * classes]], axis=0)), lg


# ------------------------------------------------------------------- módulos
def _mlp(x, largura, camadas=2, nome="mlp"):
    for i in range(camadas):
        x = layers.Dense(largura, use_bias=False, name=f"{nome}_d{i}")(x)
        x = layers.LayerNormalization(name=f"{nome}_n{i}")(x)
        x = layers.Activation("silu", name=f"{nome}_a{i}")(x)
    return x


def build_encoder(board_size=10, canais=32, nome="enc"):
    """`(B, B, 5)` → vetor. Convoluções sem pooling: num 10×10, pooling apaga o tabuleiro."""
    inp = keras.Input(shape=(board_size, board_size, N_CHANNELS), name="obs")
    x = inp
    for i, m in enumerate((1, 2)):
        x = layers.Conv2D(canais * m, 3, padding="same", use_bias=False,
                          name=f"{nome}_c{i}")(x)
        x = layers.GroupNormalization(groups=4, name=f"{nome}_n{i}")(x)
        x = layers.Activation("silu", name=f"{nome}_a{i}")(x)
    x = layers.Flatten(name=f"{nome}_f")(x)
    return keras.Model(inp, x, name=nome)


def build_decoder(dim_estado, board_size=10, canais=32, nome="dec"):
    """`(h, z)` → logits de reconstrução, um por canal da observação.

    A reconstrução é o que ancora o latente na realidade. Sem ela, o modelo pode achar
    qualquer representação que preveja recompensa — e recompensa em Snake é esparsa, então
    ele acharia a representação trivial.
    """
    inp = keras.Input(shape=(dim_estado,), name="estado")
    x = _mlp(inp, canais * board_size * board_size // 4, camadas=1, nome=f"{nome}_p")
    x = layers.Reshape((board_size // 2, board_size // 2, canais), name=f"{nome}_r")(x)
    x = layers.Conv2DTranspose(canais, 3, strides=2, padding="same", use_bias=False,
                               name=f"{nome}_t0")(x)
    x = layers.GroupNormalization(groups=4, name=f"{nome}_n0")(x)
    x = layers.Activation("silu", name=f"{nome}_a0")(x)
    out = layers.Conv2D(N_CHANNELS, 3, padding="same", name=f"{nome}_out")(x)
    return keras.Model(inp, out, name=nome)


class CelulaRecorrente(layers.Layer):
    """GRU com LayerNorm nas portas — a recorrência do RSSM.

    A `GRUCell` do Keras não normaliza, e sem normalização a recorrência do Dreamer
    diverge em treinos longos: `h` cresce sem limite porque nada o segura.
    """

    def __init__(self, unidades, **kw):
        super().__init__(**kw)
        self.unidades = int(unidades)

    def build(self, forma):
        self.projeta = layers.Dense(3 * self.unidades, use_bias=False, name="proj")
        self.norma = layers.LayerNormalization(name="norm")
        super().build(forma)

    def call(self, entrada, h):
        partes = self.norma(self.projeta(tf.concat([entrada, h], axis=-1)))
        reset, cand, atualiza = tf.split(partes, 3, axis=-1)
        reset = tf.sigmoid(reset)
        cand = tf.tanh(reset * cand)
        atualiza = tf.sigmoid(atualiza - 1.0)  # viés para lembrar, como em Hafner et al.
        return atualiza * cand + (1 - atualiza) * h

    def get_config(self):
        return {**super().get_config(), "unidades": self.unidades}


def build_rssm_prior(deter, grupos, classes, largura, nome="prior"):
    """`h` → logits de `z`. É o que o modelo prevê **sem** ver a observação."""
    inp = keras.Input(shape=(deter,), name="h")
    x = _mlp(inp, largura, camadas=1, nome=f"{nome}_m")
    out = layers.Dense(grupos * classes, name=f"{nome}_out")(x)
    return keras.Model(inp, out, name=nome)


def build_rssm_post(deter, dim_emb, grupos, classes, largura, nome="post"):
    """`(h, enc(o))` → logits de `z`. O posterior, corrigido pela observação."""
    h = keras.Input(shape=(deter,), name="h")
    e = keras.Input(shape=(dim_emb,), name="emb")
    x = _mlp(layers.Concatenate()([h, e]), largura, camadas=1, nome=f"{nome}_m")
    out = layers.Dense(grupos * classes, name=f"{nome}_out")(x)
    return keras.Model([h, e], out, name=nome)


def build_cabecas(dim_estado, largura, n_bins, n_actions=N_ACTIONS, nome="cab"):
    """`(h, z)` → `[recompensa (two-hot), continuação, máscara de ações]`.

    A cabeça de máscara não está no DreamerV3 original — está aqui porque este ambiente
    tem máscara de morte imediata, e sem prevê-la o ator treinaria num sonho onde
    movimentos suicidas continuam disponíveis e depois agiria num mundo onde não estão.
    Essa diferença entre o sonho e o jogo é exatamente o que faz um agente baseado em
    modelo aprender uma política que não transfere.
    """
    inp = keras.Input(shape=(dim_estado,), name="estado")
    x = _mlp(inp, largura, camadas=2, nome=f"{nome}_m")
    r = layers.Dense(n_bins, name=f"{nome}_rec")(x)
    c = layers.Dense(1, name=f"{nome}_cont")(x)
    m = layers.Dense(n_actions, name=f"{nome}_mask")(x)
    return keras.Model(inp, [r, c, m], name=nome)


def build_ator(dim_estado, largura, n_actions=N_ACTIONS, nome="ator"):
    inp = keras.Input(shape=(dim_estado,), name="estado")
    x = _mlp(inp, largura, camadas=2, nome=f"{nome}_m")
    out = layers.Dense(n_actions, name=f"{nome}_out",
                       kernel_initializer=keras.initializers.Orthogonal(gain=0.01))(x)
    return keras.Model(inp, out, name=nome)


def build_critico(dim_estado, largura, n_bins, nome="critico"):
    """Crítico em two-hot, como a recompensa. Regressão escalar de valor é o que faz o
    crítico do Dreamer instabilizar quando o retorno cresce de 1 para 50."""
    inp = keras.Input(shape=(dim_estado,), name="estado")
    x = _mlp(inp, largura, camadas=2, nome=f"{nome}_m")
    out = layers.Dense(n_bins, name=f"{nome}_out",
                       kernel_initializer="zeros")(x)
    return keras.Model(inp, out, name=nome)
