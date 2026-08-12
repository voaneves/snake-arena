"""Memória de repetição — uniforme e priorizada.

A PER do repositório antigo tinha um bug que matava o notebook
*"DQN (RMSprop - PER - Dueling - CNN4)"* na primeira transição::

    self.memory[self.pos] = experience     # IndexError: list assignment index out of range

A lista nunca era pré-alocada. Aqui os buffers são arrays NumPy de tamanho fixo, alocados
no `__init__` — sem `append`, sem realocação, sem surpresa no meio de um treino de horas.

Sobre a sum-tree
----------------
A PER precisa sortear proporcionalmente à prioridade. Fazer isso com uma varredura linear
custa `O(n)` por amostra e domina o tempo de treino quando a memória tem centenas de
milhares de transições. A árvore de somas resolve em `O(log n)`: cada nó guarda a soma das
prioridades da sua subárvore, então sortear é descer a árvore comparando com a soma
parcial. Está implementada aqui em NumPy puro, iterativa — recursão em Python neste laço
seria o gargalo.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SumTree", "ReplayBuffer", "PrioritizedReplayBuffer"]


class SumTree:
    """Árvore de somas sobre `capacity` folhas, em NumPy. Sorteio em `O(log n)`.

    A indexação implícita (`filhos de i são 2i e 2i+1`) exige uma árvore binária
    **completa**, ou seja, número de folhas potência de dois. Uma capacidade de 500
    quebraria a descida com `IndexError` — então arredondamos para cima e deixamos as
    folhas excedentes com prioridade zero, que nunca são sorteadas.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.folhas = 1
        while self.folhas < self.capacity:
            self.folhas *= 2
        self.tree = np.zeros(2 * self.folhas, dtype=np.float64)

    def __setitem__(self, i, valor):
        i = int(i) + self.folhas
        self.tree[i] = valor
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def __getitem__(self, i):
        return self.tree[int(i) + self.folhas]

    def total(self):
        return float(self.tree[1])

    def buscar(self, valores):
        """Índices das folhas onde a soma acumulada ultrapassa cada `valor`. Vetorizado."""
        idx = np.ones(len(valores), dtype=np.int64)
        v = np.asarray(valores, dtype=np.float64).copy()
        while idx[0] < self.folhas:
            esq = 2 * idx
            vai_direita = v > self.tree[esq]
            v = np.where(vai_direita, v - self.tree[esq], v)
            idx = esq + vai_direita.astype(np.int64)
        return np.minimum(idx - self.folhas, self.capacity - 1)


class ReplayBuffer:
    """Memória uniforme, com suporte a retornos de `n` passos.

    Os retornos de n passos são acumulados **na inserção**, com uma fila curta: em vez de
    guardar a transição crua e reconstruir depois, guardamos direto
    `(s_t, a_t, R_t^{(n)}, s_{t+n}, done)`. Fica mais simples e não exige que o buffer
    saiba a ordem temporal das transições — o que importa aqui, porque os dados chegam de
    centenas de ambientes paralelos e a memória vê tudo entrelaçado.
    """

    def __init__(self, capacity, obs_shape, n_actions=3, n_steps=1, gamma=0.99,
                 num_envs=1, rng=None):
        self.capacity = int(capacity)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.num_envs = int(num_envs)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros(self.capacity, dtype=np.int32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.next_mask = np.ones((self.capacity, n_actions), dtype=bool)

        self.pos = 0
        self.size = 0
        # filas de n passos, uma por ambiente
        self._fila = [[] for _ in range(self.num_envs)]

    def __len__(self):
        return self.size

    def _guardar(self, obs, act, rew, next_obs, done, next_mask):
        i = self.pos
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.next_mask[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def add_batch(self, obs, act, rew, next_obs, done, next_mask):
        """Insere um passo de **todos** os ambientes de uma vez."""
        for e in range(len(act)):
            self._add_um(e, obs[e], act[e], rew[e], next_obs[e], done[e], next_mask[e])

    def _add_um(self, env_i, obs, act, rew, next_obs, done, next_mask):
        fila = self._fila[env_i]
        fila.append((obs, act, rew, next_obs, done, next_mask))
        if len(fila) < self.n_steps and not done:
            return
        self._descarregar(fila, apenas_um=not done)
        if done:
            fila.clear()

    def _descarregar(self, fila, apenas_um=True):
        """Emite a transição de n passos mais antiga (ou todas, quando o episódio acaba)."""
        while fila:
            r_acum, g = 0.0, 1.0
            ultimo = None
            for k, (_, _, r, no, d, nm) in enumerate(fila):
                r_acum += g * r
                g *= self.gamma
                ultimo = (no, d, nm)
                if d or k == self.n_steps - 1:
                    break
            o0, a0 = fila[0][0], fila[0][1]
            no, d, nm = ultimo
            self._guardar(o0, a0, r_acum, no, d, nm)
            fila.pop(0)
            if apenas_um:
                return

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._montar(idx), idx, np.ones(batch_size, dtype=np.float32)

    def _montar(self, idx):
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "next_mask": self.next_mask[idx],
        }

    def update_priorities(self, idx, prioridades):
        """Sem efeito na memória uniforme — existe para a interface ser a mesma."""


class PrioritizedReplayBuffer(ReplayBuffer):
    """PER com sum-tree, correção de viés por importance sampling e β crescente.

    Sobre o β: a priorização introduz viés — transições surpreendentes aparecem mais do que
    deveriam. O peso de importância corrige isso, mas corrigir totalmente desde o início
    atrapalha, porque no começo o erro de TD é ruído. Por isso β sobe de `beta0` até 1 ao
    longo do treino: pouca correção quando o sinal é ruim, correção completa quando
    importa. Sem isso a PER converge para um ótimo do problema errado.
    """

    def __init__(self, capacity, obs_shape, alpha=0.6, beta0=0.4, eps=1e-6, **kw):
        super().__init__(capacity, obs_shape, **kw)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.eps = float(eps)
        self.arvore = SumTree(self.capacity)
        self.max_prioridade = 1.0

    def _guardar(self, *a, **kw):
        i = super()._guardar(*a, **kw)
        # transição nova entra com prioridade máxima: precisa ser vista ao menos uma vez
        self.arvore[i] = self.max_prioridade ** self.alpha
        return i

    def sample(self, batch_size, beta=None):
        beta = self.beta0 if beta is None else float(beta)
        total = self.arvore.total()
        if total <= 0:
            return super().sample(batch_size)

        # amostragem estratificada: um sorteio por fatia, em vez de `batch_size` sorteios
        # independentes. Cobre o espectro de prioridades de forma mais estável.
        bordas = np.linspace(0, total, batch_size + 1)
        valores = self.rng.uniform(bordas[:-1], bordas[1:])
        idx = self.arvore.buscar(valores)
        idx = np.clip(idx, 0, max(self.size - 1, 0))

        prob = np.array([self.arvore[i] for i in idx], dtype=np.float64) / total
        prob = np.maximum(prob, 1e-12)
        pesos = (self.size * prob) ** (-beta)
        pesos = (pesos / pesos.max()).astype(np.float32)
        return self._montar(idx), idx, pesos

    def update_priorities(self, idx, prioridades):
        p = np.abs(np.asarray(prioridades, dtype=np.float64)) + self.eps
        for i, pi in zip(np.asarray(idx).ravel(), p.ravel()):
            self.arvore[int(i)] = pi ** self.alpha
        self.max_prioridade = max(self.max_prioridade, float(p.max()))
