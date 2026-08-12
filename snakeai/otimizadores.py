"""Otimizadores — o eixo que substitui o K-FAC.

Por que o K-FAC saiu
--------------------
Dois notebooks do `colab-rl` tentaram K-FAC e nenhum roda: ele dependia de
`tensorflow.contrib.kfac`, que não existe desde o TensorFlow 2. Reimplementar aproximação
de Fisher por blocos Kronecker do zero é um projeto próprio, não uma correção de porte.

Mas a **pergunta** por trás daqueles notebooks era boa: *o otimizador importa?* Ela merece
resposta, e merece uma resposta medida. Este módulo é essa resposta — um eixo de ablação
com otimizadores que existem, funcionam em Keras 3 e cobrem escolhas de projeto diferentes:

===========  ==============================================================
nome         o que muda
===========  ==============================================================
``rmsprop``  o que o repositório antigo usava na maioria dos experimentos
``adam``     momento + escala adaptativa; o padrão de fato em RL
``adamw``    Adam com decaimento de peso desacoplado — regulariza sem mexer
             na escala adaptativa, ao contrário do `weight_decay` clássico
``lion``     só o **sinal** do momento; usa muito menos memória de estado e
             costuma preferir LR ~10x menor
``sgd``      o controle: momento e nada mais. Se o eixo não separar nada,
             este aqui denuncia
===========  ==============================================================

Todos entram pelo mesmo lugar: `cfg.optimizer = "adamw"`. O resto do experimento não muda,
que é o que torna a comparação uma ablação e não uma anedota.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

__all__ = ["OTIMIZADORES", "cria_otimizador", "LR_SUGERIDO"]

OTIMIZADORES = ("adam", "adamw", "rmsprop", "lion", "sgd")

#: Multiplicador de learning rate típico de cada otimizador, relativo ao Adam. O Lion usa
#: só o sinal do momento, então o passo tem magnitude constante e o LR precisa ser bem
#: menor; o SGD, sem escala adaptativa, precisa de bem maior. Comparar otimizadores com o
#: mesmo LR não mede otimizador — mede quem tolera aquele LR específico.
LR_SUGERIDO = {"adam": 1.0, "adamw": 1.0, "rmsprop": 1.0, "lion": 0.1, "sgd": 30.0}


def cria_otimizador(nome, learning_rate, clipnorm=None, weight_decay=1e-4, **kw):
    """Devolve um `keras.optimizers.Optimizer` pelo nome.

    `learning_rate` é o valor **base**; aplique `LR_SUGERIDO[nome]` por fora se quiser a
    escala típica de cada um. Deixar isso explícito é de propósito: um experimento que
    ajusta o LR junto com o otimizador está medindo os dois ao mesmo tempo, e precisa
    dizer isso.
    """
    nome = nome.lower()
    comum = {"learning_rate": learning_rate}
    if clipnorm is not None:
        comum["clipnorm"] = clipnorm

    if nome == "adam":
        return keras.optimizers.Adam(epsilon=1e-5, **comum, **kw)
    if nome == "adamw":
        return keras.optimizers.AdamW(epsilon=1e-5, weight_decay=weight_decay,
                                      **comum, **kw)
    if nome == "rmsprop":
        return keras.optimizers.RMSprop(rho=0.95, epsilon=1e-5, **comum, **kw)
    if nome == "lion":
        return keras.optimizers.Lion(beta_1=0.9, beta_2=0.99, **comum, **kw)
    if nome == "sgd":
        return keras.optimizers.SGD(momentum=0.9, nesterov=True, **comum, **kw)
    raise ValueError(f"otimizador desconhecido: {nome!r}. Use um de {OTIMIZADORES}")
