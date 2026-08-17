"""O registro de redes — qualquer tronco, para qualquer algoritmo, por string.

É isto que transforma "qual arquitetura é melhor?" numa ablação medida: o agente recebe
`net="cnn_vgg"` ou `net="resnet_small"` e o resto do experimento não muda. Sem isso, cada
comparação de rede viraria um notebook novo, que é como o repositório antigo acabou com
seis DQNs que ninguém conseguia comparar.

Todo modelo construído aqui obedece ao contrato: entrada `(B, B, 5)` egocêntrica,
saída de política com 3 ações relativas.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers

from ..env.vec_snake import N_ACTIONS, N_CHANNELS
from .classic import APELIDOS_LEGADOS, TRONCOS_CLASSICOS
from .heads import distributional_head, dueling_head
from .resnet import TRONCOS_RESIDUAIS

__all__ = [
    "TRONCOS",
    "listar_troncos",
    "build_backbone",
    "build_actor_critic",
    "build_q_network",
    "build_policy_q",
    "resumo",
]

TRONCOS = {**TRONCOS_RESIDUAIS, **TRONCOS_CLASSICOS}

#: A cabeça densa do repositório antigo tinha **3136** unidades. O número não é arbitrário
#: — é exatamente o achatamento da `cnn_rainbow` num tabuleiro 10×10 (7×7×64), o mesmo
#: valor do DQN do Atari por coincidência de kernels. Só que uma camada de 3136 sobre uma
#: entrada de 3136 são **9,8 milhões de parâmetros**, e ela era replicada nas duas
#: correntes do dueling. Sobre a `cnn_vgg`, que entrega 64 features, a mesma camada liga
#: 64 entradas a 3136 unidades: quase toda a capacidade do modelo depois de o tronco já
#: ter descartado a informação espacial.
#: O padrão aqui é 256. `LARGURA_DENSA_LEGADA` continua disponível para reproduzir o
#: original quando a fidelidade importar mais que o bom senso.
LARGURA_DENSA_LEGADA = 3136
LARGURA_DENSA_PADRAO = 256


def listar_troncos():
    """Nomes aceitos, incluindo os apelidos numéricos do repositório antigo."""
    return sorted(TRONCOS) + sorted(APELIDOS_LEGADOS)


def _resolve(nome):
    if nome in TRONCOS:
        return TRONCOS[nome], nome
    if nome in APELIDOS_LEGADOS:
        canonico = APELIDOS_LEGADOS[nome]
        return TRONCOS[canonico], canonico
    raise ValueError(
        f"tronco desconhecido: {nome!r}. Disponíveis: {listar_troncos()}"
    )


def build_backbone(entrada, net="resnet_small"):
    """Aplica o tronco `net` a um tensor de entrada. Devolve `(saida, nome_canonico)`."""
    fn, canonico = _resolve(net)
    return fn(entrada), canonico


def _entrada(board_size, canais=N_CHANNELS):
    """A entrada do tronco. `canais` só sai de 5 numa ablação declarada.

    O contrato fixa 5 canais, e mudar isso muda a **entrada da rede** — nenhuma curva de 5
    canais é comparável a uma de 6. O parâmetro existe para `VecSnake(canal_fome=True)`,
    que é uma ablação `comparable=False`, e não para configuração casual.
    """
    return keras.Input(shape=(board_size, board_size, canais), name="board")


def _e_espacial(t):
    """True se o tronco devolveu um mapa `(H, W, C)` em vez de um vetor achatado."""
    return len(t.shape) == 4


def build_actor_critic(board_size=10, net="resnet_small", largura_densa=None,
                       n_actions=N_ACTIONS, nome=None, canais=N_CHANNELS):
    """Modelo de duas saídas `[logits, valor]` — o que PPO, A2C e ACER consomem.

    Em troncos que preservam a estrutura espacial (as ResNets), as cabeças são
    convoluções 1×1 seguidas de achatamento, como no AlphaZero: mais barato e mais
    informativo que jogar um `Dense` gigante em cima de um mapa achatado. Em troncos
    clássicos, que já achatam, usa-se a cabeça densa mesmo.

    O `Dense` final da política nasce com `kernel_initializer` de ganho pequeno: no início
    do treino a política precisa ser quase uniforme, senão o PPO gasta as primeiras
    iterações desfazendo uma preferência aleatória.
    """
    inp = _entrada(board_size, canais)
    x, canonico = build_backbone(inp, net)
    largura = LARGURA_DENSA_PADRAO if largura_densa is None else int(largura_densa)

    if _e_espacial(x):
        p = layers.Conv2D(4, 1, use_bias=False, name="pi_c")(x)
        p = layers.GroupNormalization(groups=2, name="pi_n")(p)
        p = layers.Activation("relu", name="pi_a")(p)
        p = layers.Flatten(name="pi_f")(p)

        v = layers.Conv2D(2, 1, use_bias=False, name="v_c")(x)
        v = layers.GroupNormalization(groups=2, name="v_n")(v)
        v = layers.Activation("relu", name="v_a")(v)
        v = layers.Flatten(name="v_f")(v)
        v = layers.Dense(largura, activation="relu", name="v_d")(v)
    else:
        p = layers.Dense(largura, activation="relu", name="pi_d")(x)
        v = layers.Dense(largura, activation="relu", name="v_d")(x)

    logits = layers.Dense(
        n_actions, name="logits",
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
        bias_initializer="zeros",
    )(p)
    valor = layers.Dense(
        1, name="value",
        kernel_initializer=keras.initializers.Orthogonal(gain=1.0),
        bias_initializer="zeros",
    )(v)

    return keras.Model(inp, [logits, valor], name=nome or f"ac_{canonico}")


def build_q_network(board_size=10, net="cnn_rainbow", largura_densa=None,
                    n_actions=N_ACTIONS, dueling=False, noisy=False, n_atoms=0,
                    nome=None, canais=N_CHANNELS):
    """A família DQN inteira num construtor só.

    `dueling`, `noisy` e `n_atoms` são os eixos que separam o DQN base do Rainbow — e são
    ortogonais de propósito, para que cada um possa ser medido isolado. Essa é a resposta
    aos seis notebooks quase idênticos do repositório antigo: uma função, seis chamadas.

    Saída
    -----
    `(lote, n_ações)` no modo normal; `(lote, n_ações, n_atoms)` de **logits** quando
    `n_atoms > 0` (C51).
    """
    inp = _entrada(board_size, canais)
    x, canonico = build_backbone(inp, net)
    largura = LARGURA_DENSA_PADRAO if largura_densa is None else int(largura_densa)
    densa = "noisy" if noisy else "dense"

    if _e_espacial(x):
        x = layers.Conv2D(8, 1, use_bias=False, name="q_c")(x)
        x = layers.GroupNormalization(groups=2, name="q_n")(x)
        x = layers.Activation("relu", name="q_a")(x)
        x = layers.Flatten(name="q_f")(x)

    if n_atoms:
        saida = distributional_head(x, n_actions, n_atoms=n_atoms, largura=largura,
                                    densa=densa, dueling=dueling)
    elif dueling:
        saida = dueling_head(x, n_actions, largura=largura, densa=densa)
    else:
        from .heads import _densa
        h = _densa(densa, largura, "relu", "q_d")(x)
        saida = _densa(densa, n_actions, None, "q")(h)

    partes = [p for p, on in (("dueling", dueling), ("noisy", noisy),
                              (f"c51x{n_atoms}", bool(n_atoms))) if on]
    sufixo = ("_" + "_".join(partes)) if partes else ""
    return keras.Model(inp, saida, name=nome or f"q_{canonico}{sufixo}")


def build_policy_q(board_size=10, net="resnet_small", largura_densa=None,
                   n_actions=N_ACTIONS, nome=None, canais=N_CHANNELS):
    """Modelo de duas saídas `[logits, Q(s,·)]` — o que o ACER consome.

    Diferente do actor-critic comum: aqui o crítico devolve **um valor por ação**, não um
    escalar. É disso que o Retrace precisa, e `V(s) = Σ_a π(a|s) Q(s,a)` sai de graça —
    sem uma terceira cabeça e sem inconsistência entre V e Q, que é uma fonte clássica de
    bug silencioso em ACER.
    """
    inp = _entrada(board_size, canais)
    x, canonico = build_backbone(inp, net)
    largura = LARGURA_DENSA_PADRAO if largura_densa is None else int(largura_densa)

    if _e_espacial(x):
        p = layers.Conv2D(4, 1, use_bias=False, name="pi_c")(x)
        p = layers.GroupNormalization(groups=2, name="pi_n")(p)
        p = layers.Activation("relu", name="pi_a")(p)
        p = layers.Flatten(name="pi_f")(p)

        q = layers.Conv2D(8, 1, use_bias=False, name="q_c")(x)
        q = layers.GroupNormalization(groups=2, name="q_n")(q)
        q = layers.Activation("relu", name="q_a")(q)
        q = layers.Flatten(name="q_f")(q)
        q = layers.Dense(largura, activation="relu", name="q_d")(q)
    else:
        p = layers.Dense(largura, activation="relu", name="pi_d")(x)
        q = layers.Dense(largura, activation="relu", name="q_d")(x)

    logits = layers.Dense(
        n_actions, name="logits",
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
        bias_initializer="zeros",
    )(p)
    q_saida = layers.Dense(n_actions, name="q", bias_initializer="zeros")(q)
    return keras.Model(inp, [logits, q_saida], name=nome or f"acer_{canonico}")


def resumo(board_size=10, largura_densa=None):
    """Tabela comparativa dos troncos: parâmetros e formato de saída.

    Usada no notebook de ablação e no README. Revela, de graça, quais troncos colapsam o
    tabuleiro — a coluna `saída do tronco` mostra `1×1` para os que usam pooling.
    """
    linhas = []
    for nome in sorted(TRONCOS):
        inp = _entrada(board_size)
        saida, _ = build_backbone(inp, nome)
        forma = tuple(saida.shape[1:])
        modelo = build_actor_critic(board_size, nome, largura_densa)
        tronco = keras.Model(inp, saida)
        linhas.append({
            "tronco": nome,
            "saida_tronco": "×".join(str(d) for d in forma),
            "espacial": _e_espacial(saida),
            "params_tronco": tronco.count_params(),
            "params_actor_critic": modelo.count_params(),
        })
    return linhas
