"""Memória de **trajetórias** — o que o ACER precisa e o replay do DQN não dá.

O DQN sorteia transições soltas: para o alvo de TD, `(s, a, r, s')` basta. O ACER não —
o Retrace(λ) é uma recursão para trás no tempo, e a correção por importance sampling
compara a política atual com a que *gerou aquela sequência*. Sem a ordem temporal e sem a
política de comportamento gravada, nenhuma das duas coisas existe.

Foi exatamente aqui que o ACER legado quebrou. O erro era
``expected shape=(None, 256, 100), found shape=(None, 100)``: a dimensão de tempo tinha
sumido em algum ponto entre a coleta e o update. Aqui os segmentos são guardados com forma
`(T, N, ...)` explícita, e o teste `test_stored_segment_keeps_the_time_axis` trava isso.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TrajectoryBuffer"]


class TrajectoryBuffer:
    """Guarda segmentos de rollout inteiros, com a política de comportamento.

    Cada entrada é um segmento `(T, N, ...)`: `T` passos de `N` ambientes em paralelo.
    Amostrar devolve um segmento inteiro, não uma transição — é a unidade que o Retrace
    consome.

    O campo `mu` é o que torna o algoritmo *off-policy* honesto: a probabilidade que a
    política tinha **no momento da coleta**. A razão `π/μ` sem esse registro seria `π/π`,
    ou seja, 1, e o ACER viraria um A2C caro.
    """

    def __init__(self, capacity, rng=None):
        self.capacity = int(capacity)
        self.dados = []
        self.pos = 0
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def __len__(self):
        return len(self.dados)

    def add(self, obs, mask, act, mu, rew, done, obs_final, mask_final):
        """Guarda um segmento. Todos os arrays têm que começar com `(T, N, ...)`.

        `obs_final` / `mask_final` são o estado **logo depois** do último passo do
        segmento. Sem eles, o bootstrap do Retrace num segmento antigo teria de usar o
        estado atual do ambiente — que não tem relação nenhuma com aquela trajetória. É um
        erro que não levanta exceção: o algoritmo treina e aprende o valor errado.
        """
        T, N = act.shape
        for nome, arr, forma in (
            ("obs", obs, (T, N)), ("mask", mask, (T, N)), ("mu", mu, (T, N)),
            ("rew", rew, (T, N)), ("done", done, (T, N)),
        ):
            if arr.shape[:2] != forma:
                raise ValueError(
                    f"`{nome}` tem forma {arr.shape}; o eixo de tempo precisa vir "
                    f"primeiro: esperado {forma} + resto"
                )

        segmento = {
            "obs_final": np.asarray(obs_final, dtype=np.float32),
            "mask_final": np.asarray(mask_final, dtype=bool),
            "obs": np.asarray(obs, dtype=np.float32),
            "mask": np.asarray(mask, dtype=bool),
            "act": np.asarray(act, dtype=np.int32),
            "mu": np.asarray(mu, dtype=np.float32),
            "rew": np.asarray(rew, dtype=np.float32),
            "done": np.asarray(done, dtype=np.float32),
        }
        if len(self.dados) < self.capacity:
            self.dados.append(segmento)
        else:
            self.dados[self.pos] = segmento
        self.pos = (self.pos + 1) % self.capacity
        return segmento

    def sample(self):
        """Um segmento aleatório, com o eixo de tempo intacto."""
        if not self.dados:
            raise RuntimeError("memória de trajetórias vazia")
        return self.dados[int(self.rng.integers(0, len(self.dados)))]
