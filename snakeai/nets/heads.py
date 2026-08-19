"""Cabeças de rede — dueling, noisy e distribucional (C51).

São os componentes que separam um DQN simples de um Rainbow. Ficam separados dos troncos
de propósito: qualquer cabeça encaixa em qualquer tronco, e é isso que permite perguntar
"quanto o dueling vale?" com o resto do experimento congelado.

Todas foram reescritas para Keras 3. A `NoisyDense` do repositório antigo herdava de
`Dense` e mexia nos internals dela (`self.kernel`, `build` reimplementado), o que quebra
em qualquer versão moderna; esta é uma `Layer` própria, com `add_weight` e `keras.random`.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers, ops

__all__ = ["NoisyDense", "dueling_head", "distributional_head", "q_de_distribuicao",
           "ruido_ligado"]


@keras.saving.register_keras_serializable(package="snakeai")
class NoisyDense(layers.Layer):
    """Camada densa com ruído fatorado nos pesos (Fortunato et al., 2017).

    Substitui a exploração ε-greedy por ruído aprendido: a rede começa barulhenta e vai
    reduzindo o próprio σ conforme fica confiante. A vantagem sobre o ε-greedy é que a
    exploração passa a ser **dependente do estado** — o agente explora onde ainda não sabe,
    não uniformemente.

    Uma decisão importante: **o ruído é desligado quando `training=False`**. O protocolo de
    avaliação do contrato é greedy e determinístico; se a rede sorteasse ruído durante o
    benchmark, o mesmo modelo daria números diferentes a cada execução e a comparação entre
    algoritmos perderia o sentido. Alguns trabalhos mantêm o ruído na avaliação — aqui não,
    e a escolha está registrada porque muda o número publicado.

    Só que **a coleta não é a avaliação**, e amarrar o ruído a `training` juntava as duas:
    a política de comportamento saía determinística, e um Rainbow com `eps=0` (porque "a
    exploração é responsabilidade da rede") passava o treino inteiro sem explorar nada. O
    atributo `ruido` desempata: `None` segue o `training`, `True` força ruído, `False`
    força determinismo. Use o gerenciador `ruido_ligado` — ele é para uso **eager**, na
    coleta; dentro de uma `tf.function` o valor vira constante no traçado.

    Parâmetros
    ----------
    units : int
        Dimensão de saída.
    sigma0 : float
        Escala inicial do ruído, dividida por `sqrt(entrada)`. 0,5 é o valor do paper.
    """

    def __init__(self, units, activation=None, sigma0=0.5, seed=None, **kw):
        super().__init__(**kw)
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.sigma0 = float(sigma0)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)
        #: `None` = segue `training`; `True`/`False` forçam. Ver `ruido_ligado`.
        self.ruido = None

    def build(self, input_shape):
        entrada = int(input_shape[-1])
        limite = 1.0 / (entrada ** 0.5)
        sigma_ini = self.sigma0 / (entrada ** 0.5)

        self.w_mu = self.add_weight(
            shape=(entrada, self.units), name="w_mu",
            initializer=keras.initializers.RandomUniform(-limite, limite))
        self.w_sigma = self.add_weight(
            shape=(entrada, self.units), name="w_sigma",
            initializer=keras.initializers.Constant(sigma_ini))
        self.b_mu = self.add_weight(
            shape=(self.units,), name="b_mu",
            initializer=keras.initializers.RandomUniform(-limite, limite))
        self.b_sigma = self.add_weight(
            shape=(self.units,), name="b_sigma",
            initializer=keras.initializers.Constant(sigma_ini))
        self._entrada = entrada

    @staticmethod
    def _f(x):
        """`sign(x) * sqrt(|x|)` — a transformação que fatora o ruído no paper."""
        return ops.sign(x) * ops.sqrt(ops.abs(x))

    def call(self, inputs, training=False):
        if self.ruido if self.ruido is not None else training:
            eps_in = self._f(keras.random.normal((self._entrada,),
                                                 seed=self.seed_generator))
            eps_out = self._f(keras.random.normal((self.units,),
                                                  seed=self.seed_generator))
            w = self.w_mu + self.w_sigma * ops.outer(eps_in, eps_out)
            b = self.b_mu + self.b_sigma * eps_out
        else:
            w, b = self.w_mu, self.b_mu

        y = ops.matmul(inputs, w) + b
        return self.activation(y) if self.activation is not None else y

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def ruido_medio(self):
        """σ médio dos pesos — cai conforme a rede fica confiante. Bom de registrar."""
        return float(ops.convert_to_numpy(ops.mean(ops.abs(self.w_sigma))))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "sigma0": self.sigma0,
            "seed": self.seed,
        })
        return cfg


def _densa(tipo, unidades, ativacao=None, nome=None):
    if tipo == "noisy":
        return NoisyDense(unidades, activation=ativacao, name=nome)
    return layers.Dense(unidades, activation=ativacao, name=nome)


def dueling_head(x, n_actions, largura=256, densa="dense", nome="dueling"):
    """`Q(s,a) = V(s) + A(s,a) − média_a A(s,a)`.

    A subtração da média é o que torna a decomposição identificável: sem ela, somar uma
    constante a `V` e subtraí-la de `A` daria o mesmo `Q`, e as duas correntes poderiam
    derivar sem que a perda percebesse.

    O original usava a **média**; o paper também oferece o **máximo**. Ficamos na média,
    que é o padrão do Rainbow.
    """
    a = _densa(densa, largura, "relu", f"{nome}_a_h")(x)
    a = _densa(densa, n_actions, None, f"{nome}_a")(a)
    v = _densa(densa, largura, "relu", f"{nome}_v_h")(x)
    v = _densa(densa, 1, None, f"{nome}_v")(v)

    a_centrada = layers.Lambda(
        lambda t: t - ops.mean(t, axis=-1, keepdims=True),
        output_shape=lambda s: s, name=f"{nome}_center",
    )(a)
    return layers.Add(name=f"{nome}_q")([v, a_centrada])


def distributional_head(x, n_actions, n_atoms=51, largura=256, densa="dense",
                        dueling=False, nome="c51"):
    """Cabeça categórica do C51: distribuição sobre `n_atoms` valores por ação.

    Em vez de estimar `Q(s,a)` — a média do retorno — o C51 estima a distribuição inteira.
    O ganho não é só estatístico: aprender uma distribuição dá um sinal de treino mais
    rico por transição, e é a peça que mais contribui no Rainbow.

    Devolve **logits** de forma `(lote, n_ações, n_átomos)`. A softmax e a projeção sobre
    o suporte ficam no agente, onde o `v_min`/`v_max` é conhecido.
    """
    if dueling:
        a = _densa(densa, largura, "relu", f"{nome}_a_h")(x)
        a = _densa(densa, n_actions * n_atoms, None, f"{nome}_a")(a)
        a = layers.Reshape((n_actions, n_atoms), name=f"{nome}_a_r")(a)

        v = _densa(densa, largura, "relu", f"{nome}_v_h")(x)
        v = _densa(densa, n_atoms, None, f"{nome}_v")(v)
        v = layers.Reshape((1, n_atoms), name=f"{nome}_v_r")(v)

        a_centrada = layers.Lambda(
            lambda t: t - ops.mean(t, axis=1, keepdims=True),
            output_shape=lambda s: s, name=f"{nome}_center",
        )(a)
        return layers.Add(name=f"{nome}_logits")([v, a_centrada])

    h = _densa(densa, largura, "relu", f"{nome}_h")(x)
    h = _densa(densa, n_actions * n_atoms, None, f"{nome}_d")(h)
    return layers.Reshape((n_actions, n_atoms), name=f"{nome}_logits")(h)


def q_de_distribuicao(logits, suporte):
    """Colapsa a distribuição categórica em `Q(s,a)` — só para escolher a ação.

    `logits`: `(lote, n_ações, n_átomos)`. `suporte`: `(n_átomos,)`.
    """
    p = ops.softmax(logits, axis=-1)
    return ops.sum(p * ops.reshape(suporte, (1, 1, -1)), axis=-1)


def suporte_c51(v_min=-10.0, v_max=10.0, n_atoms=51):
    """Os `n_atoms` valores igualmente espaçados em `[v_min, v_max]`.

    Com recompensa `+1`/`−1` e γ = 0,995, o retorno de um episódio de Snake fica bem
    dentro de `[−2, 60]` — a faixa padrão de `[−10, 10]` do Atari é estreita demais aqui.
    O agente escolhe a sua; este é só o utilitário.
    """
    import numpy as np

    return np.linspace(v_min, v_max, n_atoms, dtype=np.float32)


@contextlib.contextmanager
def ruido_ligado(modelo, ativo=True):
    """Liga o ruído das `NoisyDense` de `modelo` dentro do bloco, e devolve como estava.

    Existe porque **coletar não é avaliar**. `NoisyDense.call` amarra o ruído a
    `training`, e ligar `training=True` na coleta traria junto tudo o que esse sinalizador
    significa nos outros troncos — o `Dropout` do `cnn_classic`, por exemplo. Este
    gerenciador mexe só nas camadas ruidosas.

    Uso **eager**, na escolha da ação. Dentro de uma `tf.function` o atributo é lido no
    traçado e vira constante no grafo, que não é o que se quer.
    """
    camadas = [c for c in _camadas(modelo) if isinstance(c, NoisyDense)]
    antes = [c.ruido for c in camadas]
    for c in camadas:
        c.ruido = ativo
    try:
        yield camadas
    finally:
        for c, valor in zip(camadas, antes):
            c.ruido = valor


def _camadas(modelo):
    """Todas as camadas de `modelo`, inclusive as aninhadas em submodelos."""
    vistas, pilha = [], list(getattr(modelo, "layers", []))
    while pilha:
        c = pilha.pop()
        vistas.append(c)
        pilha.extend(getattr(c, "layers", []))
    return vistas
