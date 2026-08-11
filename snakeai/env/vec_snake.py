"""`VecSnake` — Snake vetorizado, N tabuleiros independentes evoluindo em lote.

Este módulo é **a fonte única de verdade do ambiente**. Todo algoritmo do `snake-arena`
treina e é avaliado aqui, sem exceção — é isso que torna as curvas comparáveis. Ele não
importa TensorFlow nem Keras: é NumPy puro, roda em qualquer lugar e é rápido o bastante
para que o gargalo do treino seja a GPU, não o jogo.

O truque que faz ser rápido: em vez de uma lista de posições por cobra, guardamos uma
grade `occ` de inteiros onde `occ[n, y, x]` é **quantos passos faltam para aquela célula
ficar livre**. A cabeça recebe `occ = comprimento`; a cada passo o mundo inteiro decrementa
em 1 e a cauda some sozinha. Tudo vira operação NumPy em lote sobre `(N, B, B)` — nada de
laço Python por cobra.

Como bônus, essa grade *já é* a feature mais informativa que existe para Snake: normalizada
por comprimento, ela diz à rede **quando** cada célula vai desocupar, que é exatamente a
informação necessária para a cobra passar rente ao próprio corpo sem se prender.

Convenções fixadas pelo contrato de comparabilidade (`docs/COMPARABILITY.md`):

* tabuleiro 10x10, `starve_base = 100`;
* observação `(N, B, B, 5)` egocêntrica;
* 3 ações relativas com máscara de morte imediata;
* recompensa `+1` comer, `-1` morrer, `0` passo;
* **score = comida comida**, começando em zero. Nunca comprimento.
"""

from __future__ import annotations

import numpy as np

__all__ = ["VecSnake", "DIRS", "TURN", "N_ACTIONS", "N_CHANNELS", "DEFAULT_SEED"]

# Direções: 0=cima(-y), 1=direita(+x), 2=baixo(+y), 3=esquerda(-x)  (sentido horário)
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)
# Ações relativas: 0=vira à esquerda, 1=segue reto, 2=vira à direita
TURN = np.array([-1, 0, 1], dtype=np.int32)

N_ACTIONS = 3
N_CHANNELS = 5
DEFAULT_SEED = 42


class VecSnake:
    """`num_envs` tabuleiros independentes de Snake evoluindo em lote.

    Observação: `(num_envs, B, B, 5)` float32, **egocêntrica** — o tabuleiro é rotacionado
    para que a cobra sempre olhe para cima. Isso colapsa as 4 simetrias de rotação e deixa
    a rede ~4x mais eficiente em amostras.

    Canais
    ------
    0. corpo (binário, sem a cabeça)
    1. cabeça
    2. decaimento da cauda: `occ / comprimento` em (0, 1]
    3. comida
    4. plano constante = comprimento / B**2  (a rede precisa saber o quão longa está)

    Parâmetros
    ----------
    num_envs : int
        Quantos tabuleiros correm em paralelo.
    board_size : int
        Lado do tabuleiro. O contrato oficial usa 10.
    starve_base : int, opcional
        Paciência base antes de morrer de fome; o limite efetivo é
        `starve_base + 2 * comprimento`. Padrão: `board_size ** 2`.
    rng : np.random.Generator, opcional
        Gerador próprio. Passe um com semente fixa para reprodutibilidade.
    """

    def __init__(self, num_envs=256, board_size=10, starve_base=None, rng=None):
        if board_size < 6:
            raise ValueError("tabuleiro pequeno demais para o corpo inicial (mínimo 6)")
        self.n = int(num_envs)
        self.b = int(board_size)
        self.cells = self.b * self.b
        self.starve_base = self.cells if starve_base is None else int(starve_base)
        self.rng = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)

        self.occ = np.zeros((self.n, self.b, self.b), dtype=np.int32)
        self.head = np.zeros((self.n, 2), dtype=np.int32)
        self.food = np.zeros((self.n, 2), dtype=np.int32)
        self.dir = np.zeros(self.n, dtype=np.int32)
        self.length = np.zeros(self.n, dtype=np.int32)
        self.steps = np.zeros(self.n, dtype=np.int32)
        self.hunger = np.zeros(self.n, dtype=np.int32)
        self.score = np.zeros(self.n, dtype=np.int32)

        self._reset_idx(np.arange(self.n))

    # ------------------------------------------------------------------- reset
    def _reset_idx(self, idx):
        """Reinicia apenas os ambientes em `idx`, em lote."""
        if idx.size == 0:
            return
        k = idx.size
        b = self.b
        self.occ[idx] = 0
        # cabeça longe das bordas para caber o corpo inicial de 3
        self.head[idx] = self.rng.integers(2, b - 2, size=(k, 2), dtype=np.int32)
        self.dir[idx] = self.rng.integers(0, 4, size=k, dtype=np.int32)
        self.length[idx] = 3
        self.steps[idx] = 0
        self.hunger[idx] = 0
        self.score[idx] = 0

        d = DIRS[self.dir[idx]]                       # (k, 2)
        for back, ttl in ((0, 3), (1, 2), (2, 1)):    # cabeça, meio, cauda
            p = self.head[idx] - back * d
            np.clip(p, 0, b - 1, out=p)
            self.occ[idx, p[:, 0], p[:, 1]] = ttl

        self._spawn_food(idx)

    def _spawn_food(self, idx):
        """Sorteia comida uniformemente entre as células livres (vetorizado)."""
        if idx.size == 0:
            return
        free = self.occ[idx].reshape(idx.size, -1) == 0
        r = self.rng.random((idx.size, self.cells))
        r[~free] = -1.0
        flat = r.argmax(axis=1)
        self.food[idx, 0] = flat // self.b
        self.food[idx, 1] = flat % self.b

    def reset(self):
        """Reinicia todos os ambientes. Retorna `(obs, mask)`."""
        self._reset_idx(np.arange(self.n))
        return self.obs(), self.action_mask()

    # -------------------------------------------------------------- observação
    def _raw_planes(self):
        """Os 5 canais no referencial do tabuleiro, antes da rotação egocêntrica."""
        b, n = self.b, self.n
        occ = self.occ
        body = (occ > 0).astype(np.float32)
        head = np.zeros((n, b, b), dtype=np.float32)
        rows = np.arange(n)
        head[rows, self.head[:, 0], self.head[:, 1]] = 1.0
        body -= head                                   # cabeça sai do canal de corpo
        decay = occ.astype(np.float32) / self.length[:, None, None].astype(np.float32)
        food = np.zeros((n, b, b), dtype=np.float32)
        food[rows, self.food[:, 0], self.food[:, 1]] = 1.0
        lenpl = np.broadcast_to(
            (self.length.astype(np.float32) / self.cells)[:, None, None], (n, b, b)
        )
        return np.stack([body, head, decay, food, lenpl], axis=-1)

    def obs(self):
        """Planos rotacionados para o referencial da cabeça (sempre olhando p/ cima)."""
        raw = self._raw_planes()
        out = np.empty_like(raw)
        for k in range(4):
            m = self.dir == k
            if m.any():
                out[m] = np.rot90(raw[m], k=k, axes=(1, 2))
        return out

    # ----------------------------------------------------------------- máscara
    def _next_head(self, actions):
        """Posição e direção da cabeça se `actions` fosse aplicada agora."""
        nd = (self.dir + TURN[actions]) % 4
        return self.head + DIRS[nd], nd

    def _lethal(self, pos):
        """True onde a posição mata (parede ou corpo que ainda não desocupou)."""
        b = self.b
        oob = (pos[:, 0] < 0) | (pos[:, 0] >= b) | (pos[:, 1] < 0) | (pos[:, 1] >= b)
        safe_pos = np.where(oob[:, None], 0, pos)
        # a cauda vai embora neste passo -> célula com occ<=1 estará livre
        hit = self.occ[np.arange(self.n), safe_pos[:, 0], safe_pos[:, 1]] > 1
        return oob | (hit & ~oob)

    def _raw_mask(self):
        """`(N, 3)` bool sem o *override* de beco sem saída — a verdade nua."""
        mask = np.empty((self.n, N_ACTIONS), dtype=bool)
        for a in range(N_ACTIONS):
            pos, _ = self._next_head(np.full(self.n, a, dtype=np.int32))
            mask[:, a] = ~self._lethal(pos)
        return mask

    def dead_ends(self):
        """`(N,)` bool: True onde **todas** as três ações matam.

        Existe porque `action_mask()` não permite descobrir isso — lá, um beco sem saída
        aparece como "tudo liberado". Quem precisa distinguir (testes, diagnóstico, o
        filtro de segurança) pergunta aqui.
        """
        return ~self._raw_mask().any(axis=1)

    def action_mask(self):
        """`(N, 3)` bool: True = ação não mata imediatamente.

        Se as três matam, liberamos todas (a cobra morreu de qualquer jeito) — assim a
        distribuição nunca fica sem suporte e o log-prob não vira NaN. Use `dead_ends()`
        para saber quando esse caso ocorreu.
        """
        mask = self._raw_mask()
        mask[~mask.any(axis=1)] = True
        return mask

    # -------------------------------------------------------------------- step
    def step(self, actions, shaping_coef=0.0, gamma=0.99):
        """Avança todos os ambientes um passo.

        Retorna `(obs, mask, reward, done, info)`. Ambientes terminados são resetados
        automaticamente; `obs` já é o do episódio novo, e `info` guarda as estatísticas
        do episódio que acabou.

        `info` contém:
            scores      : score final dos episódios encerrados neste passo
            lengths     : duração em passos desses episódios
            wins        : quantos encheram o tabuleiro
            deaths      : quantos morreram por colisão
            starved     : quantos foram truncados por fome
            trunc_idx   : índices dos truncados por fome
            final_obs   : observação terminal dos truncados (para bootstrap do valor)
            final_mask  : máscara terminal dos truncados
        """
        n, b = self.n, self.b
        rows = np.arange(n)
        actions = np.asarray(actions, dtype=np.int32)

        d_old = np.abs(self.head - self.food).sum(axis=1).astype(np.float32)

        new_head, new_dir = self._next_head(actions)
        dead = self._lethal(new_head)
        new_head = np.where(dead[:, None], self.head, new_head)  # congela quem morreu

        ate = (
            (~dead)
            & (new_head[:, 0] == self.food[:, 0])
            & (new_head[:, 1] == self.food[:, 1])
        )

        # cauda anda quando não comeu
        moved = ~ate & ~dead
        self.occ[moved] = np.maximum(self.occ[moved] - 1, 0)

        self.length += ate.astype(np.int32)
        self.score += ate.astype(np.int32)
        alive = ~dead
        self.head[alive] = new_head[alive]
        self.dir[alive] = new_dir[alive]
        self.occ[rows[alive], self.head[alive, 0], self.head[alive, 1]] = self.length[alive]

        self.steps += 1
        self.hunger = np.where(ate, 0, self.hunger + 1)

        won = self.length >= self.cells
        need_food = ate & ~won
        if need_food.any():
            self._spawn_food(np.nonzero(need_food)[0])

        starve_limit = self.starve_base + 2 * self.length
        starved = (self.hunger >= starve_limit) & ~dead & ~won

        # ---- recompensa
        reward = np.zeros(n, dtype=np.float32)
        reward += ate.astype(np.float32)
        reward -= dead.astype(np.float32)
        reward += won.astype(np.float32) * 2.0
        reward -= starved.astype(np.float32) * 0.5
        if shaping_coef > 0.0:
            # o delta só faz sentido quando a comida não mudou de lugar
            d_new = np.abs(self.head - self.food).sum(axis=1).astype(np.float32)
            phi_old = -d_old / b
            phi_new = -d_new / b
            delta = np.where(dead | won | ate, 0.0, gamma * phi_new - phi_old)
            reward += shaping_coef * delta

        done = dead | won | starved

        # Truncamento por fome: o episódio *continuaria*, então precisamos do valor do
        # estado final para fazer bootstrap. Como o env reseta sozinho, guardamos a
        # observação terminal antes do reset (custa uma passada extra, só quando ocorre).
        starved_idx = np.nonzero(starved)[0]
        final_obs = final_mask = None
        if starved_idx.size:
            final_obs = self.obs()[starved_idx]
            final_mask = self.action_mask()[starved_idx]

        info = {
            "scores": self.score[done].copy(),
            "lengths": self.steps[done].copy(),
            "wins": int(won.sum()),
            "deaths": int(dead.sum()),
            "starved": int(starved.sum()),
            "trunc_idx": starved_idx,
            "final_obs": final_obs,
            "final_mask": final_mask,
        }
        self._reset_idx(np.nonzero(done)[0])
        return self.obs(), self.action_mask(), reward, done, info

    # -------------------------------------------------------------- utilidades
    def free_space_from(self, env_i, pos):
        """Flood-fill: quantas células livres são alcançáveis a partir de `pos`.

        Usado só no filtro de segurança da inferência, nunca no treino.
        """
        b = self.b
        occ = self.occ[env_i]
        seen = np.zeros((b, b), dtype=bool)
        stack = [(int(pos[0]), int(pos[1]))]
        seen[pos[0], pos[1]] = True
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < b and 0 <= nx < b and not seen[ny, nx] and occ[ny, nx] <= 1:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        return count

    # ------------------------------------------------------------- introspecção
    def check_invariants(self):
        """Levanta `AssertionError` se o estado interno estiver inconsistente.

        Barato o bastante para rodar em testes e em depuração; nunca no laço de treino.
        """
        assert (self.occ >= 0).all(), "occ negativo"
        assert ((self.occ > 0).sum(axis=(1, 2)) == self.length).all(), \
            "número de células ocupadas não bate com o comprimento"
        assert (self.occ.reshape(self.n, -1).max(axis=1) == self.length).all(), \
            "a cabeça deveria ser a célula de maior occ"
        rows = np.arange(self.n)
        assert (self.occ[rows, self.head[:, 0], self.head[:, 1]] == self.length).all(), \
            "occ na posição da cabeça não é o comprimento"
        occupied_food = self.occ[rows, self.food[:, 0], self.food[:, 1]] > 0
        assert not occupied_food.any() or (self.length >= self.cells).any(), \
            "comida dentro do corpo"
        assert (self.score == self.length - 3).all(), \
            "score deve ser comprimento - 3"

    def __repr__(self):
        return (
            f"VecSnake(num_envs={self.n}, board_size={self.b}, "
            f"starve_base={self.starve_base})"
        )
