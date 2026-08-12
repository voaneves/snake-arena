"""A dinâmica que a árvore de busca percorre.

Existe para que **um só MCTS** sirva a dois mundos:

* `DinamicaReal` — o `VecSnake`. Exata, gratuita, e é o que faz o AlphaZero fazer sentido
  em Snake.
* `DinamicaAprendida` — a rede de dinâmica do MuZero. Aproximada, cara de treinar, e
  necessária só quando o simulador **não** está disponível durante a busca.

Ter as duas atrás da mesma interface é o que torna a comparação honesta: a diferença entre
AlphaZero e MuZero neste repositório passa a ser exatamente *o que a árvore percorre*, com
o algoritmo de busca, o PUCT e o backup literalmente idênticos.
"""

from __future__ import annotations

import numpy as np

from ..env.vec_snake import N_ACTIONS, VecSnake

__all__ = ["DinamicaReal", "DinamicaAprendida"]


class DinamicaReal:
    """Um passo do jogo de verdade, em lote, a partir de estados arbitrários."""

    usa_mascara = True

    def __init__(self, board_size=10, starve_base=None):
        self.board_size = int(board_size)
        self.starve_base = starve_base
        self._env = None

    def _ambiente(self, n):
        if self._env is None or self._env.n != n:
            self._env = VecSnake(n, self.board_size, starve_base=self.starve_base,
                                 rng=np.random.default_rng(0))
        return self._env

    def passo(self, estados, acoes):
        """`(novos_estados, obs, mask, recompensa, terminal)`.

        Os estados são dicionários de arrays do `VecSnake.get_state()`.
        """
        env = self._ambiente(len(acoes))
        env.set_state(estados)
        obs, mask, rew, done, _ = env.step(np.asarray(acoes, dtype=np.int32))
        return env.get_state(), obs, mask, rew, done

    @staticmethod
    def empilhar(estados_por_arvore):
        return {c: np.stack([e[c] for e in estados_por_arvore])
                for c in estados_por_arvore[0]}

    @staticmethod
    def fatiar(estados, i):
        return {c: estados[c][i].copy() for c in estados}


class DinamicaAprendida:
    """A rede de dinâmica do MuZero: `(estado_oculto, ação) → (estado', recompensa)`.

    Diferenças que importam em relação à dinâmica real, e que estão aqui de propósito para
    ficarem visíveis:

    * **Não há terminação.** O modelo não prevê fim de episódio; ele aprende que morrer
      rende `−1` e segue rolando. É como o MuZero original trata o assunto.
    * **Não há máscara dentro da árvore.** A máscara vale na raiz, onde o estado é real. Da
      raiz para baixo o estado é uma abstração aprendida, e não existe "ação ilegal" nela —
      o modelo tem que aprender sozinho que certas ações rendem `−1`.

    O estado é o tensor oculto `(N, B, B, largura)`.
    """

    usa_mascara = False

    def __init__(self, fn_dinamica):
        self.fn = fn_dinamica

    def passo(self, estados, acoes):
        novo, recompensa = self.fn(estados, np.asarray(acoes, dtype=np.int32))
        n = len(acoes)
        mask = np.ones((n, N_ACTIONS), dtype=bool)
        done = np.zeros(n, dtype=bool)
        # a "observação" de um nó interno **é** o estado oculto: a rede de predição lê dele
        return novo, novo, mask, np.asarray(recompensa, dtype=np.float32), done

    @staticmethod
    def empilhar(estados_por_arvore):
        return np.stack(estados_por_arvore)

    @staticmethod
    def fatiar(estados, i):
        return estados[i].copy()
