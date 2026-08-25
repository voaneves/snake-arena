"""Ver a cobra jogar — GIF de um episódio, sem pygame.

O Colab não tem display, então renderizar pelo jogo original não é opção. Aqui o episódio
vira uma sequência de imagens direto da grade `occ` do `VecSnake`, e o GIF é o artefato
que se olha para entender *como* o agente joga — coisa que nenhuma curva conta.

Vale mais do que parece: um agente com score médio 20 que morre sempre se prendendo no
próprio corpo e outro que morre por fome têm a mesma linha no gráfico e problemas
completamente diferentes.
"""

from __future__ import annotations

import numpy as np

from .vec_snake import VecSnake

__all__ = ["quadros_do_episodio", "render_episode", "PALETA_JOGO"]

#: Fundo, corpo, comida, cabeça — nas cores do gráfico da arena, para o GIF e as figuras
#: parecerem do mesmo projeto.
PALETA_JOGO = np.array(
    [
        [26, 26, 25],      # fundo (a superfície escura do gráfico)
        [27, 175, 122],    # corpo (aqua da paleta)
        [235, 104, 52],    # comida (laranja da paleta)
        [252, 252, 251],   # cabeça (tinta clara)
    ],
    dtype=np.uint8,
)


def _quadro(env: VecSnake, i=0, escala=16):
    grade = np.zeros((env.b, env.b), dtype=np.int32)
    grade[env.occ[i] > 0] = 1
    grade[env.food[i, 0], env.food[i, 1]] = 2
    grade[env.head[i, 0], env.head[i, 1]] = 3
    return PALETA_JOGO[grade].repeat(escala, 0).repeat(escala, 1)


def quadros_do_episodio(politica, board_size=10, safety=False, max_steps=2000,
                        seed=7, escala=16, canal_fome=False):
    """Roda um episódio com `politica` e devolve `(quadros, score, motivo)`.

    `politica` é a mesma interface de `snakeai.eval`: `politica(obs, mask) -> logits`.
    Assim o GIF mostra exatamente a política que o benchmark mediu, sem caminho paralelo.
    """
    from ..eval import MASK_NEG, apply_safety_filter

    # `canal_fome` tem que acompanhar o ambiente de treino: uma rede de 6 canais
    # recebendo observação de 5 quebra aqui, e o GIF é gerado no fim do treino — tarde
    # demais para descobrir. Ver `snakeai.eval.evaluate`.
    env = VecSnake(1, board_size, rng=np.random.default_rng(seed),
                   canal_fome=canal_fome)
    obs, mask = env.reset()
    quadros = [_quadro(env, escala=escala)]
    score, motivo = 0, "limite de passos"
    # Políticas com memória — o `PoliticaRecorrente` do DreamerV3, o `PoliticaComOpcoes`
    # do SOAP — precisam saber qual ação de fato saiu para avançar o estado interno.
    # `snakeai.eval` já respeitava este contrato; aqui não, e o resultado era um GIF
    # gravado com o latente congelado no valor inicial: o agente do vídeo não era o
    # agente da curva, e o vídeo é justamente o artefato que se olha para entender *como*
    # ele joga. Políticas sem memória não expõem o método e nada muda para elas.
    apos_passo = getattr(politica, "apos_passo", None)

    for _ in range(max_steps):
        logits = np.asarray(politica(obs, mask), dtype=np.float32)
        logits = np.where(mask, logits, MASK_NEG)
        if safety:
            logits = apply_safety_filter(env, logits)
        a = logits.argmax(axis=1).astype(np.int32)

        score_antes = int(env.score[0])
        comprimento_antes = int(env.length[0])
        fome_antes = int(env.hunger[0])
        obs, mask, r, d, info = env.step(a)
        if apos_passo is not None:
            apos_passo(a, d)
        quadros.append(_quadro(env, escala=escala))

        if d[0]:
            # `info["scores"]` é o score **final**, já com a maçã do último passo. Ler
            # `env.score` antes do passo perde exatamente um ponto nos episódios que
            # terminam comendo — que são precisamente as vitórias, e o GIF de uma vitória
            # saía rotulado com 96 num tabuleiro cujo perfeito é 97. Mesmo defeito que o
            # `eval.py` corrige e trava com teste.
            finais = info.get("scores")
            score = int(finais[0]) if finais is not None and len(finais) else score_antes
            if comprimento_antes >= board_size * board_size - 1:
                motivo = "tabuleiro cheio"
            elif fome_antes + 1 >= env.starve_base + 2 * comprimento_antes:
                motivo = "fome"
            else:
                motivo = "colisão"
            break
    else:
        score = int(env.score[0])

    return quadros, score, motivo


def render_episode(politica, caminho="episodio.gif", fps=15, **kw):
    """Grava o GIF e devolve `(caminho, score, motivo)`."""
    import imageio.v2 as imageio

    quadros, score, motivo = quadros_do_episodio(politica, **kw)
    imageio.mimsave(caminho, quadros, fps=fps, loop=0)
    return caminho, score, motivo
