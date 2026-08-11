"""Os troncos convolucionais do `colab-rl`, portados para Keras 3 e corrigidos.

Estes são os corpos de rede que produziram as curvas históricas. Estão aqui para que a
pergunta "quanto do ganho é o algoritmo e quanto é a arquitetura?" tenha resposta medida
em vez de opinião.

Duas coisas que a portabilidade revelou, e que valem mais que o código
------------------------------------------------------------------------

**1. "CNN2" significava duas coisas diferentes no mesmo repositório.**

O `colab-rl` tinha as CNNs definidas em dois lugares, com a mesma numeração e conteúdo
diferente:

===========  ==============================  ==================================
nome         em `models/utilities/networks.py`  nos notebooks
===========  ==============================  ==================================
``CNN1``     16→32, **quebrada** (`return model`)  32→64→64 (Rainbow)
``CNN2``     16→32→32, **quebrada**             32→64→64 com regularização L2
``CNN3``     32→64→64 (Rainbow)                 3 blocos VGG com max-pooling
``CNN4``     não existia                        idem CNN3, com dropout
===========  ==============================  ==================================

Ou seja: o notebook chamado *"DQN (RMSprop - CNN2 - KL-Divergence)"* usava um tronco que
**não é** a `CNN2` do pacote. Um leitor que fosse ao `networks.py` entender o experimento
leria a rede errada. Aqui as redes têm nome descritivo (`cnn_rainbow`, `cnn_alphazero`,
`cnn_vgg`, `cnn_vgg_dropout`) e os apelidos numéricos apontam para as definições **dos
notebooks**, que são as que de fato rodaram.

**2. As redes com pooling destroem o tabuleiro.**

`cnn_vgg` e `cnn_vgg_dropout` aplicam três `MaxPooling2D(2, 2)` seguidos. Num tabuleiro
10×10 isso é ``10 → 5 → 2 → 1``: a saída do tronco tem **uma única célula**. Toda a
informação de *onde* as coisas estão no tabuleiro é jogada fora antes da cabeça densa —
sobra só "existe corpo em algum lugar", "existe comida em algum lugar".

Essas arquiteturas foram desenhadas para imagens 224×224, onde três poolings deixam 28×28.
Copiadas para 10×10, elas colapsam. É uma explicação forte para o platô dos notebooks que
as usavam, e por isso `cnn_vgg_sem_pool` existe: mesma rede, sem os poolings, para medir
exatamente quanto custou.

Os troncos são fiéis ao original de propósito. A correção fica na cabeça (`heads.py`) e
nas variantes explicitamente marcadas.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers, regularizers

__all__ = [
    "cnn_rainbow",
    "cnn_alphazero",
    "cnn_vgg",
    "cnn_vgg_dropout",
    "cnn_vgg_sem_pool",
    "TRONCOS_CLASSICOS",
    "APELIDOS_LEGADOS",
]


def cnn_rainbow(x, nome="cnn_rainbow"):
    """32→64→64 com kernels 3×3, 2×2, 1×1, sem padding.

    Da implementação do Rainbow do @Kaixhin. É a `CNN1` dos notebooks e a `CNN3` do
    `networks.py` — a mesma rede com dois nomes.
    """
    x = layers.Conv2D(32, 3, activation="relu", name=f"{nome}_c1")(x)
    x = layers.Conv2D(64, 2, activation="relu", name=f"{nome}_c2")(x)
    x = layers.Conv2D(64, 1, activation="relu", name=f"{nome}_c3")(x)
    return layers.Flatten(name=f"{nome}_flat")(x)


def cnn_alphazero(x, l2const=1e-4, nome="cnn_alphazero"):
    """A mesma pilha da `cnn_rainbow`, com regularização L2 e ativação separada.

    É a `CNN2` **dos notebooks** — a que rodou no experimento "CNN2 - KL-Divergence".
    """
    reg = regularizers.l2(l2const)
    for i, (filtros, k) in enumerate(((32, 3), (64, 2), (64, 1)), start=1):
        x = layers.Conv2D(filtros, k, kernel_regularizer=reg, name=f"{nome}_c{i}")(x)
        x = layers.Activation("relu", name=f"{nome}_a{i}")(x)
    return layers.Flatten(name=f"{nome}_flat")(x)


def _blocos_vgg(x, nome, dropout=0.0, pooling=True):
    plano = ((16, 2), (32, 2), (64, 3))
    for b, (filtros, convs) in enumerate(plano, start=1):
        for c in range(1, convs + 1):
            x = layers.Conv2D(filtros, 3, activation="relu", padding="same",
                              name=f"{nome}_b{b}_c{c}")(x)
            if dropout:
                x = layers.Dropout(dropout, name=f"{nome}_b{b}_d{c}")(x)
        if pooling:
            x = layers.MaxPooling2D(2, strides=2, name=f"{nome}_b{b}_pool")(x)
    return layers.Flatten(name=f"{nome}_flat")(x)


def cnn_vgg(x, nome="cnn_vgg"):
    """Três blocos no estilo VGG com max-pooling. É a `CNN3` dos notebooks.

    **Atenção:** os três poolings reduzem um tabuleiro 10×10 a 1×1. Ver o cabeçalho do
    módulo. Mantida fiel ao original porque é o que produziu as curvas históricas.
    """
    return _blocos_vgg(x, nome, dropout=0.0, pooling=True)


def cnn_vgg_dropout(x, nome="cnn_vgg_dropout", taxa=0.1):
    """`cnn_vgg` com dropout de 0,1 após cada convolução. É a `CNN4` dos notebooks."""
    return _blocos_vgg(x, nome, dropout=taxa, pooling=True)


def cnn_vgg_sem_pool(x, nome="cnn_vgg_sem_pool"):
    """`cnn_vgg` sem os max-poolings — a variante de ablação.

    Não existia no repositório antigo. Existe aqui para responder, com número, quanto do
    platô daquelas execuções veio de colapsar o tabuleiro a uma célula.
    """
    return _blocos_vgg(x, nome, dropout=0.0, pooling=False)


#: Nome descritivo -> função de tronco.
TRONCOS_CLASSICOS = {
    "cnn_rainbow": cnn_rainbow,
    "cnn_alphazero": cnn_alphazero,
    "cnn_vgg": cnn_vgg,
    "cnn_vgg_dropout": cnn_vgg_dropout,
    "cnn_vgg_sem_pool": cnn_vgg_sem_pool,
}

#: Apelidos numéricos do repositório antigo. Apontam para as definições **dos
#: notebooks**, que são as que realmente rodaram (ver o cabeçalho do módulo).
APELIDOS_LEGADOS = {
    "cnn1": "cnn_rainbow",
    "cnn2": "cnn_alphazero",
    "cnn3": "cnn_vgg",
    "cnn4": "cnn_vgg_dropout",
}
