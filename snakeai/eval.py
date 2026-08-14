"""Avaliação — o protocolo oficial do benchmark.

Este módulo responde à única pergunta que importa: **quanto esse agente tira, de verdade?**
Ele é deliberadamente independente de TensorFlow e Keras — recebe uma *função de política*,
não um modelo. Isso permite avaliar qualquer coisa pelo mesmo caminho: uma rede Keras, uma
tabela, uma heurística escrita à mão, ou a política aleatória que define o piso. E permite
testar a avaliação sem GPU.

Protocolo fixado pelo contrato de comparabilidade (`docs/COMPARABILITY.md`):

* 1.000 episódios, tabuleiro 10x10;
* política **greedy** (sem exploração);
* `seed = 123`;
* **sem** filtro de segurança na curva principal;
* métrica = `score` (comida comida), nunca comprimento.

Sobre o viés que este módulo corrige
------------------------------------
A forma ingênua de avaliar é rodar N ambientes em paralelo e parar assim que 1.000
episódios terminarem. Isso **subestima o agente**: episódios curtos terminam primeiro e
entram na amostra, enquanto os longos — que são justamente os bons — ainda estão correndo
quando a contagem fecha. Quanto melhor o agente, pior o viés.

A correção é simples: cada ambiente contribui com o mesmo número de episódios (os
primeiros que ele terminar), em vez de a amostra ser "os primeiros a terminar no total".
"""

from __future__ import annotations

import math

import numpy as np

from .env.vec_snake import N_ACTIONS, VecSnake

__all__ = [
    "MASK_NEG",
    "evaluate",
    "random_baseline",
    "random_policy",
    "keras_policy",
    "apply_safety_filter",
    "verdict",
]

MASK_NEG = -1e9

#: Piso documentado no README: política aleatória com máscara, 1.000 episódios, 10x10.
PISO_ALEATORIO_10X10 = 1.08


# --------------------------------------------------------------------- políticas
def random_policy(rng=None):
    """Política uniforme sobre as ações permitidas — o piso do benchmark.

    Não é "aleatória pura": ela respeita a máscara, ou seja, já evita a morte imediata.
    É o piso honesto, porque qualquer agente do benchmark também tem a máscara.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    def politica(obs, mask):
        return np.where(mask, rng.random(mask.shape), -np.inf).astype(np.float32)

    return politica


def keras_policy(model, batch_size=None):
    """Embrulha um modelo Keras (actor-critic) numa função de política.

    O import de TensorFlow acontece aqui dentro, de propósito: quem só quer avaliar uma
    heurística não precisa ter TF instalado.
    """
    import tensorflow as tf  # noqa: PLC0415  (lazy de propósito)

    @tf.function(reduce_retracing=True)
    def _forward(obs, mask):
        saida = model(obs, training=False)
        logits = saida[0] if isinstance(saida, (list, tuple)) else saida
        return tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))

    def politica(obs, mask):
        return _forward(
            tf.convert_to_tensor(obs), tf.convert_to_tensor(mask)
        ).numpy()

    return politica


# ------------------------------------------------------------- filtro de segurança
def apply_safety_filter(env: VecSnake, logits, margin=1.0, penalty=50.0):
    """Penaliza ações que deixariam a cobra num bolso menor que o próprio corpo.

    Pós-processamento de inferência, **não aprendido**: entre as ações que a rede
    considera boas, desencoraja as que se fecham num espaço sem saída (flood-fill a partir
    da nova cabeça). Por isso ele nunca entra na curva principal do benchmark — vira
    coluna separada da tabela.

    A penalidade é grande mas finita: se *todas* as opções forem ruins, a ordem relativa
    que a rede preferia é preservada e o agente escolhe a menos pior.
    """
    out = np.array(logits, dtype=np.float32, copy=True)
    for a in range(N_ACTIONS):
        pos, _ = env._next_head(np.full(env.n, a, dtype=np.int32))
        lethal = env._lethal(pos)
        for i in range(env.n):
            if lethal[i]:
                out[i, a] = MASK_NEG
            elif env.free_space_from(i, pos[i]) < margin * env.length[i]:
                out[i, a] -= penalty
    return out


# ------------------------------------------------------------------- avaliação
def evaluate(
    policy,
    board_size=10,
    episodes=1000,
    num_envs=250,
    greedy=True,
    safety=False,
    seed=123,
    max_steps=200_000,
    rng=None,
):
    """Roda o protocolo oficial e devolve `(stats, scores)`.

    Parâmetros
    ----------
    policy : callable
        `policy(obs, mask) -> logits (N, 3)`. Já deve aplicar a máscara aos logits;
        `evaluate` não confia nisso e reaplica de qualquer forma.
    episodes : int
        Quantos episódios compõem a amostra. O contrato usa 1.000.
    greedy : bool
        `True` = argmax (o padrão do benchmark). `False` = amostra da softmax.
    safety : bool
        Liga o flood-fill. Fora da curva principal, por construção.
    seed : int
        Semente do ambiente. Fixa em 123 no contrato, para que a sequência de comidas
        seja a mesma para todos os algoritmos.

    Cada ambiente contribui com o mesmo número de episódios — ver a nota sobre viés no
    topo do módulo.
    """
    env = VecSnake(num_envs, board_size, rng=np.random.default_rng(seed))
    rng = rng if rng is not None else np.random.default_rng(seed + 1)
    obs, mask = env.reset()
    apos_passo = getattr(policy, "apos_passo", None)

    por_env = math.ceil(episodes / num_envs)
    coletados = [[] for _ in range(num_envs)]
    #: Por que cada episódio da amostra terminou. Score sozinho não distingue "o agente
    #: joga mal" de "o agente anda em círculo": um DQN greedy no começo do treino tira
    #: 0,05 morrendo **100% por fome**, e a leitura correta disso não é "não aprendeu", é
    #: "a política determinística entrou em ciclo". São problemas diferentes.
    motivos = {"fome": 0, "colisao": 0, "tabuleiro_cheio": 0}
    faltam = num_envs
    passos = 0

    while faltam > 0 and passos < max_steps:
        logits = np.asarray(policy(obs, mask), dtype=np.float32)
        logits = np.where(mask, logits, MASK_NEG)
        if safety:
            logits = apply_safety_filter(env, logits)

        if greedy:
            acoes = logits.argmax(axis=1).astype(np.int32)
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            acoes = (p.cumsum(axis=1) > rng.random((num_envs, 1))).argmax(axis=1).astype(np.int32)

        obs, mask, r, done, info = env.step(acoes)
        passos += 1

        # Políticas com estado recorrente (DreamerV3) precisam saber o que de fato
        # aconteceu: a ação escolhida — que pode não ser o argmax, se o filtro de
        # segurança agiu — e onde o episódio terminou, para zerar o estado latente ali.
        # Políticas sem memória simplesmente não expõem este método.
        if apos_passo is not None:
            apos_passo(acoes, done)

        # `info["scores"]` é o score **final** do episódio, já contando a comida do
        # último passo. Ler `env.score` antes do passo perde exatamente um ponto nos
        # episódios que terminam comendo — que são precisamente as vitórias. Ver
        # `test_eval.py::test_a_winning_episode_scores_the_last_apple`.
        truncados = set(info["trunc_idx"].tolist())
        for j, i in enumerate(np.nonzero(done)[0]):
            if len(coletados[i]) < por_env:
                s_final = int(info["scores"][j])
                coletados[i].append(s_final)
                if i in truncados:
                    motivos["fome"] += 1
                elif s_final == board_size * board_size - 3:
                    motivos["tabuleiro_cheio"] += 1
                else:
                    motivos["colisao"] += 1
                if len(coletados[i]) == por_env:
                    faltam -= 1

    scores = np.array([s for lista in coletados for s in lista][:episodes], dtype=np.int32)
    if scores.size == 0:
        raise RuntimeError("nenhum episódio terminou — aumente `max_steps`")

    perfeito = board_size * board_size - 3
    # A taxa de vitória sai da **amostra coletada**, não de um contador do laço: o laço
    # continua rodando os ambientes que já cumpriram a cota, e somar as vitórias deles
    # daria uma taxa que não corresponde aos episódios de fato medidos.
    stats = {
        "episodes": int(scores.size),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std": float(scores.std()),
        "score_max": int(scores.max()),
        "score_p95": float(np.percentile(scores, 95)),
        "win_rate": float((scores == perfeito).mean()),
        "perfect_possible": perfeito,
        "env_steps_used": int(passos),
        "completo": bool(faltam == 0),
    }
    total_motivos = max(1, sum(motivos.values()))
    stats.update({f"fim_{k}": v / total_motivos for k, v in motivos.items()})
    return stats, scores


def random_baseline(board_size=10, episodes=1000, num_envs=250, seed=123):
    """O piso: política uniforme sobre as ações permitidas.

    É o número contra o qual todo resultado do benchmark é lido. Num 10x10 ele vale
    ~1,08 — qualquer coisa que não esteja bem acima disso não aprendeu nada.
    """
    stats, _ = evaluate(
        random_policy(np.random.default_rng(seed)),
        board_size=board_size,
        episodes=episodes,
        num_envs=num_envs,
        greedy=False,
        seed=seed,
    )
    return stats["score_mean"]


# ---------------------------------------------------------------------- veredito
def verdict(policy, board_size=10, episodes=1000, num_envs=250, com_filtro=True, seed=123):
    """A resposta objetiva para "aprendeu mesmo?".

    Roda, na mesma execução, três regimes e devolve a tabela:

    ===========================  =============================================
    regime                       o que mede
    ===========================  =============================================
    aleatório com máscara        o piso — quanto se tira sem aprender nada
    agente (greedy)              a política pura, sem nenhuma ajuda externa
    agente + filtro de segurança o teto prático, com o flood-fill ligado
    ===========================  =============================================

    Se a linha do meio não estiver bem acima do piso, não aprendeu — e aí o problema é de
    hiperparâmetro ou de tempo de treino, não do código.
    """
    linhas = []

    piso = random_baseline(board_size, episodes, num_envs, seed)
    linhas.append({"regime": "aleatório com máscara", "score_mean": piso})

    st, sc = evaluate(policy, board_size=board_size, episodes=episodes,
                      num_envs=num_envs, greedy=True, seed=seed)
    linhas.append({"regime": "agente (greedy)", "scores": sc, **st})

    if com_filtro:
        # o flood-fill é laço Python: menos ambientes, para não ficar lento
        stf, scf = evaluate(policy, board_size=board_size, episodes=episodes,
                            num_envs=min(num_envs, 64), greedy=True, safety=True,
                            seed=seed)
        linhas.append({"regime": "agente + filtro de segurança", "scores": scf, **stf})

    return {
        "piso": piso,
        "perfeito": board_size * board_size - 3,
        "ganho_sobre_o_piso": linhas[1]["score_mean"] / max(piso, 1e-9),
        "linhas": linhas,
    }


def format_verdict(resultado):
    """Formata o retorno de `verdict` como tabela de texto."""
    larg = 30
    out = [f"{'regime':<{larg}}{'média':>8}{'mediana':>9}{'máx':>6}{'cheio':>8}", "-" * (larg + 31)]
    for ln in resultado["linhas"]:
        med = f"{ln['score_median']:.0f}" if "score_median" in ln else "-"
        mx = f"{ln['score_max']}" if "score_max" in ln else "-"
        wr = f"{ln['win_rate']:.1%}" if "win_rate" in ln else "-"
        out.append(f"{ln['regime']:<{larg}}{ln['score_mean']:>8.2f}{med:>9}{mx:>6}{wr:>8}")
    out.append("-" * (larg + 31))
    out.append(
        f"score perfeito: {resultado['perfeito']}   |   "
        f"ganho sobre o piso: {resultado['ganho_sobre_o_piso']:.1f}x"
    )
    ag = resultado["linhas"][1]
    if "fim_fome" in ag:
        out.append(
            f"como terminou: fome {ag['fim_fome']:.0%} · colisão {ag['fim_colisao']:.0%}"
            f" · tabuleiro cheio {ag['fim_tabuleiro_cheio']:.0%}"
        )
        # Morrer de fome é o fim NORMAL aqui: a máscara de morte impede a colisão, então
        # até a política aleatória termina 85% dos episódios por fome. O que denuncia o
        # ciclo é a combinação — quase nenhuma colisão **e** score abaixo do piso, ou seja,
        # a cobra anda para sempre sem nunca comer.
        if ag["fim_colisao"] < 0.05 and ag["score_mean"] < resultado["piso"]:
            out.append(
                "  ⚠ nunca colide e não come: a política determinística entrou em ciclo.\n"
                "    Não é 'jogou mal' — é falta de exploração na hora de agir. Normal cedo\n"
                "    num DQN greedy, e é por isso que o score de TREINO (ε-greedy) fica\n"
                "    acima do de AVALIAÇÃO (greedy) nesta fase."
            )
    return "\n".join(out)
