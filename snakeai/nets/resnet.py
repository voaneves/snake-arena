"""Tronco residual totalmente convolucional — a rede do PPO.

No espírito do AlphaZero, mas minúsculo. Convoluções 3×3 com `padding="same"` num
tabuleiro 10×10 dão campo receptivo global depois de ~5 camadas, então 3 blocos residuais
já enxergam o tabuleiro inteiro — **sem jogar fora a posição**, que é onde as redes com
pooling do repositório antigo se perdiam.

Sobre normalização: PPO e BatchNorm se dão mal. As estatísticas do rollout não batem com
as do minibatch de update, e o valor aprendido fica dependente do tamanho do lote. Usamos
**GroupNorm**, que normaliza por amostra e não tem esse problema — e que funciona igual
para DQN, o que mantém a comparação limpa.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras import layers

__all__ = ["PRESETS", "residual_block", "resnet", "TRONCOS_RESIDUAIS"]

#: nome -> (largura, número de blocos residuais)
PRESETS = {
    "resnet_tiny": (32, 2),     # ~40k params com a cabeça
    "resnet_small": (48, 3),    # ~135k — o ponto doce
    "resnet_base": (64, 4),     # ~320k
}


def residual_block(x, largura, nome):
    y = layers.Conv2D(largura, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{nome}_c1")(x)
    y = layers.GroupNormalization(groups=8, name=f"{nome}_n1")(y)
    y = layers.Activation("relu", name=f"{nome}_a1")(y)
    y = layers.Conv2D(largura, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{nome}_c2")(y)
    y = layers.GroupNormalization(groups=8, name=f"{nome}_n2")(y)
    out = layers.Add(name=f"{nome}_add")([x, y])
    return layers.Activation("relu", name=f"{nome}_a2")(out)


def resnet(x, preset="resnet_small", nome=None):
    """Tronco residual. Devolve o mapa de features `(B, B, largura)`, **sem achatar**.

    Não achatar é de propósito: as cabeças convolucionais 1×1 de `heads.py` aproveitam a
    estrutura espacial, e achatar cedo seria desperdiçá-la.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset desconhecido: {preset!r}. Use um de {list(PRESETS)}")
    largura, blocos = PRESETS[preset]
    nome = nome or preset

    x = layers.Conv2D(largura, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{nome}_stem_c")(x)
    x = layers.GroupNormalization(groups=8, name=f"{nome}_stem_n")(x)
    x = layers.Activation("relu", name=f"{nome}_stem_a")(x)
    for i in range(blocos):
        x = residual_block(x, largura, f"{nome}_res{i}")
    return x


TRONCOS_RESIDUAIS = {
    nome: (lambda x, _p=nome: resnet(x, preset=_p)) for nome in PRESETS
}
