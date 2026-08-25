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
    "build_actor_critic_populacao",
    "build_option_actor_critic",
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


def build_actor_critic_populacao(board_size=10, net="resnet_small", n_politicas=3,
                                 largura_densa=None, n_actions=N_ACTIONS, nome=None,
                                 canais=N_CHANNELS):
    """`N` pares (política, valor) sobre um tronco **compartilhado** — o que o LBC consome.

    Saída `[logits, valor]` com formas `(lote, N, ações)` e `(lote, N)`: a população
    inteira num forward só. É essa forma que permite ao comportamento do LBC ser uma
    mistura sobre as `N` políticas sem `N` passadas pela rede.

    O tronco compartilhado é um **desvio declarado** do paper, que trata cada política como
    um modelo inteiro e independente (Assumption 1). A razão é o orçamento: o contrato deste
    repositório dá 5 M passos de ambiente a todos os algoritmos, e três ResNets separadas
    triplicariam o custo por passo — o LBC entraria na arena competindo com o mesmo
    orçamento de ambiente e três vezes mais computação, que é a comparação que este
    repositório existe para não fazer. Ver `docs/LBC.md`.

    O que se perde é diversidade de **representação**: as três políticas veem as mesmas
    features. O que se mantém é diversidade de **objetivo** — cada cabeça é treinada com o
    seu próprio γ e o seu próprio alvo V-trace — e é ela que constrói o espaço de
    comportamento não-degenerado do §4.1.
    """
    if int(n_politicas) < 1:
        raise ValueError("a população precisa de pelo menos uma política")
    n_politicas = int(n_politicas)

    inp = _entrada(board_size, canais)
    x, canonico = build_backbone(inp, net)
    largura = LARGURA_DENSA_PADRAO if largura_densa is None else int(largura_densa)
    espacial = _e_espacial(x)

    logits_por_politica, valores_por_politica = [], []
    for i in range(n_politicas):
        if espacial:
            p = layers.Conv2D(4, 1, use_bias=False, name=f"pi{i}_c")(x)
            p = layers.GroupNormalization(groups=2, name=f"pi{i}_n")(p)
            p = layers.Activation("relu", name=f"pi{i}_a")(p)
            p = layers.Flatten(name=f"pi{i}_f")(p)

            v = layers.Conv2D(2, 1, use_bias=False, name=f"v{i}_c")(x)
            v = layers.GroupNormalization(groups=2, name=f"v{i}_n")(v)
            v = layers.Activation("relu", name=f"v{i}_a")(v)
            v = layers.Flatten(name=f"v{i}_f")(v)
            v = layers.Dense(largura, activation="relu", name=f"v{i}_d")(v)
        else:
            p = layers.Dense(largura, activation="relu", name=f"pi{i}_d")(x)
            v = layers.Dense(largura, activation="relu", name=f"v{i}_d")(x)

        li = layers.Dense(
            n_actions, name=f"logits_{i}",
            kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
            bias_initializer="zeros",
        )(p)
        vi = layers.Dense(
            1, name=f"value_{i}",
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0),
            bias_initializer="zeros",
        )(v)
        logits_por_politica.append(
            layers.Reshape((1, n_actions), name=f"logits_{i}_r")(li))
        valores_por_politica.append(vi)

    # `Concatenate` recusa uma entrada só — e uma população de tamanho 1 é justamente a
    # ablação "reduzir H" da Fig. 5 do paper, então este caminho tem que existir.
    if n_politicas == 1:
        logits = logits_por_politica[0]
        valor = valores_por_politica[0]
    else:
        logits = layers.Concatenate(axis=1, name="logits")(logits_por_politica)
        valor = layers.Concatenate(axis=-1, name="value")(valores_por_politica)

    return keras.Model(inp, [logits, valor],
                       name=nome or f"lbc{n_politicas}_{canonico}")


def build_option_actor_critic(board_size=10, net="resnet_small", n_opcoes=4,
                              largura_densa=None, n_actions=N_ACTIONS, nome=None,
                              canais=N_CHANNELS):
    """Política com opções — o que o SOAP consome. Três saídas:

    * `logits_a` `(lote, Z, ações)` — a sub-política `π_θ(a|s,z)`, uma por opção;
    * `logits_z` `(lote, Z, ações, Z)` — a transição `π_ψ(z'|s,a,z)`;
    * `valor` `(lote, Z)` — o crítico condicionado à opção corrente.

    A transição depende de `(s, a, z)`, e não só de `s`: é a fatoração que o paper do SOAP
    propõe contra a do Option-Critic, e é ela que permite a uma opção **persistir** por
    conta própria em vez de ser re-sorteada a cada passo. O custo é um tensor de saída
    `Z × A × Z` — com `Z = 4` e `A = 3`, 48 números por estado, que é barato.

    Os logits da sub-política nascem com ganho pequeno, como no `build_actor_critic`: no
    começo do treino toda opção precisa ser quase uniforme. Os da transição também, e por
    um motivo mais forte — uma preferência inicial de troca de opção é um viés que o agente
    gasta as primeiras iterações desfazendo, e enquanto isso a crença `ζ` já colapsou.
    """
    if int(n_opcoes) < 1:
        raise ValueError("é preciso pelo menos uma opção")
    n_opcoes = int(n_opcoes)

    inp = _entrada(board_size, canais)
    x, canonico = build_backbone(inp, net)
    largura = LARGURA_DENSA_PADRAO if largura_densa is None else int(largura_densa)

    def projeta(nome_curto, filtros):
        if _e_espacial(x):
            h = layers.Conv2D(filtros, 1, use_bias=False, name=f"{nome_curto}_c")(x)
            h = layers.GroupNormalization(groups=2, name=f"{nome_curto}_n")(h)
            h = layers.Activation("relu", name=f"{nome_curto}_a")(h)
            h = layers.Flatten(name=f"{nome_curto}_f")(h)
            if nome_curto != "pi":
                h = layers.Dense(largura, activation="relu", name=f"{nome_curto}_d")(h)
            return h
        return layers.Dense(largura, activation="relu", name=f"{nome_curto}_d")(x)

    p = projeta("pi", 4)
    q = projeta("psi", 4)
    v = projeta("v", 2)

    logits_a = layers.Dense(
        n_opcoes * n_actions, name="logits_a_d",
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
        bias_initializer="zeros",
    )(p)
    logits_a = layers.Reshape((n_opcoes, n_actions), name="logits_a")(logits_a)

    logits_z = layers.Dense(
        n_opcoes * n_actions * n_opcoes, name="logits_z_d",
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
        bias_initializer="zeros",
    )(q)
    logits_z = layers.Reshape((n_opcoes, n_actions, n_opcoes),
                              name="logits_z")(logits_z)

    valor = layers.Dense(
        n_opcoes, name="value",
        kernel_initializer=keras.initializers.Orthogonal(gain=1.0),
        bias_initializer="zeros",
    )(v)

    return keras.Model(inp, [logits_a, logits_z, valor],
                       name=nome or f"soap{n_opcoes}_{canonico}")


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
