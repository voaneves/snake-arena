"""Memória de **sequências contíguas** — o que o DreamerV3 consome.

Três memórias, três unidades de amostragem, e a diferença não é estilo:

* `ReplayBuffer` (DQN) sorteia **transições soltas**. Para um alvo de TD de um passo,
  `(s, a, r, s')` é tudo o que existe.
* `TrajectoryBuffer` (ACER) guarda **segmentos de rollout recentes**, porque o Retrace(λ)
  é uma recursão para trás e precisa da política que gerou aquele segmento.
* Este guarda **sequências contíguas longas de qualquer época do treino**, porque o modelo
  do mundo do Dreamer é recorrente: ele aprende `p(o_{t+1} | o_{≤t}, a_{≤t})`, e uma
  transição solta não tem `o_{≤t}`.

Como está organizado
--------------------
Um anel por ambiente, todos do mesmo tamanho, em arrays `(N, C, ...)` — `N` ambientes,
`C` de capacidade por ambiente. Amostrar é escolher `(ambiente, deslocamento)` e cortar
`T` passos.

O detalhe que decide se funciona: **a janela não pode atravessar a cabeça do anel.** Se
atravessar, a sequência mistura o passo mais recente com o mais antigo — uma
descontinuidade que o modelo aprende como se fosse física do jogo. É um erro silencioso:
nada quebra, o modelo só fica pior. `test_sampled_window_never_crosses_the_ring_head`
tranca isso.

Já cruzar um **fim de episódio** é permitido e desejado: o campo `first` marca onde
começou um episódio novo, e o RSSM zera o estado recorrente ali. É assim que o Dreamer
aprende que morrer termina o retorno.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SequenceBuffer"]


class SequenceBuffer:
    def __init__(self, num_envs, capacidade, obs_shape, n_actions, seed=0):
        self.n = int(num_envs)
        self.c = int(capacidade)
        if self.c < 2:
            raise ValueError("capacidade por ambiente precisa de ao menos 2 passos")
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.n, self.c, *obs_shape), dtype=np.float32)
        self.act = np.zeros((self.n, self.c), dtype=np.int32)
        self.rew = np.zeros((self.n, self.c), dtype=np.float32)
        self.cont = np.zeros((self.n, self.c), dtype=np.float32)
        self.first = np.zeros((self.n, self.c), dtype=np.bool_)
        self.mask = np.ones((self.n, self.c, n_actions), dtype=np.bool_)

        self.cabeca = 0
        self.tamanho = 0

    # ------------------------------------------------------------------ escrita
    def add(self, obs, act, rew, cont, first, mask):
        i = self.cabeca
        self.obs[:, i] = obs
        self.act[:, i] = act
        self.rew[:, i] = rew
        self.cont[:, i] = cont
        self.first[:, i] = first
        self.mask[:, i] = mask
        self.cabeca = (i + 1) % self.c
        self.tamanho = min(self.tamanho + 1, self.c)

    def __len__(self):
        return self.tamanho * self.n

    def pronto(self, T):
        return self.tamanho >= T + 1

    # ---------------------------------------------------------------- amostragem
    def _inicios_validos(self, T):
        """Deslocamentos onde uma janela de `T` passos não atravessa a cabeça do anel.

        Enquanto o anel não deu a volta, os dados vão de 0 a `cabeca`, e qualquer janela
        que termine antes da cabeça vale. Depois de dar a volta, o mais antigo está *em*
        `cabeca`, e as janelas válidas começam em `cabeca` e vão até `cabeca - T`.
        """
        if self.tamanho < self.c:
            return 0, self.tamanho - T
        return self.cabeca, self.cabeca + self.c - T

    def sample(self, lote, T):
        """Devolve um dicionário de arrays `(lote, T, ...)`."""
        if not self.pronto(T):
            raise ValueError(
                f"memória tem {self.tamanho} passos por ambiente, sequência pede {T}")

        ini, fim = self._inicios_validos(T)
        envs = self.rng.integers(0, self.n, size=lote)
        desloc = self.rng.integers(ini, fim + 1, size=lote)
        idx = (desloc[:, None] + np.arange(T)[None, :]) % self.c

        linhas = envs[:, None]
        return {
            "obs": self.obs[linhas, idx],
            "act": self.act[linhas, idx],
            "rew": self.rew[linhas, idx],
            "cont": self.cont[linhas, idx],
            "first": self.first[linhas, idx],
            "mask": self.mask[linhas, idx],
        }
