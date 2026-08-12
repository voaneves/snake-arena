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
                        seed=7, escala=16):
    """Roda um episódio com `politica` e devolve `(quadros, score, motivo)`.

    `politica` é a mesma interface de `snakeai.eval`: `politica(obs, mask) -> logits`.
    Assim o GIF mostra exatamente a política que o benchmark mediu, sem caminho paralelo.
    """
    from ..eval import MASK_NEG, apply_safety_filter

    env = VecSnake(1, board_size, rng=np.random.default_rng(seed))
    obs, mask = env.reset()
    quadros = [_quadro(env, escala=escala)]
    score, motivo = 0, "limite de passos"

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
        quadros.append(_quadro(env, escala=escala))

        if d[0]:
            score = score_antes
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
