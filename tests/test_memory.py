"""Memória de repetição.

O primeiro teste aqui é o que teria evitado o `IndexError` que matava o notebook
*"DQN (RMSprop - PER - Dueling - CNN4)"* do repositório antigo na primeira transição.
"""

import numpy as np
import pytest

from snakeai.memory import PrioritizedReplayBuffer, ReplayBuffer, SumTree

FORMA = (10, 10, 5)


def transicao(n=1, rng=None):
    rng = rng or np.random.default_rng(0)
    return (
        rng.normal(size=(n, *FORMA)).astype(np.float32),
        rng.integers(0, 3, size=n).astype(np.int32),
        rng.normal(size=n).astype(np.float32),
        rng.normal(size=(n, *FORMA)).astype(np.float32),
        np.zeros(n, dtype=np.float32),
        np.ones((n, 3), dtype=bool),
    )


# ------------------------------------------------------------------- sum-tree
def test_sumtree_total_and_lookup():
    t = SumTree(8)
    for i, v in enumerate([1, 2, 3, 4, 0, 0, 0, 0]):
        t[i] = v
    assert t.total() == pytest.approx(10.0)
    # a soma acumulada é [1, 3, 6, 10]; procurar 0,5 cai na folha 0, 2,5 na folha 1...
    achados = t.buscar([0.5, 2.5, 5.5, 9.5])
    assert achados.tolist() == [0, 1, 2, 3]


def test_sumtree_handles_non_power_of_two_capacity():
    """500 folhas: a árvore arredonda para 512 e as excedentes ficam em zero.

    Sem isso, a descida `2*i` sai do array com IndexError na primeira amostragem — o que
    só apareceria depois de encher a memória, no meio de um treino longo.
    """
    t = SumTree(500)
    assert t.folhas == 512
    for i in range(500):
        t[i] = 1.0
    assert t.total() == pytest.approx(500.0)
    achados = t.buscar(np.linspace(0.01, 499.99, 64))
    assert achados.min() >= 0 and achados.max() < 500


def test_sumtree_update_propagates():
    t = SumTree(4)
    t[0] = 5.0
    assert t.total() == pytest.approx(5.0)
    t[0] = 1.0
    assert t.total() == pytest.approx(1.0)


def test_sumtree_sampling_is_proportional():
    t = SumTree(4)
    for i, v in enumerate([1.0, 9.0, 0.0, 0.0]):
        t[i] = v
    rng = np.random.default_rng(0)
    achados = t.buscar(rng.uniform(0, t.total(), 4000))
    frac_1 = (achados == 1).mean()
    assert 0.85 < frac_1 < 0.95, f"folha com 90% da prioridade saiu {frac_1:.2%}"


# -------------------------------------------------------------- buffer básico
def test_buffer_preallocates_and_never_raises_on_first_insert():
    """O bug do repositório antigo: lista não pré-alocada -> IndexError na 1ª transição."""
    for Classe in (ReplayBuffer, PrioritizedReplayBuffer):
        mem = Classe(1000, FORMA, num_envs=4)
        mem.add_batch(*transicao(4))
        assert len(mem) == 4


def test_buffer_is_circular():
    mem = ReplayBuffer(10, FORMA, num_envs=1)
    for _ in range(25):
        mem.add_batch(*transicao(1))
    assert len(mem) == 10
    assert mem.pos == 25 % 10


def test_sample_shapes():
    mem = ReplayBuffer(100, FORMA, num_envs=8)
    for _ in range(20):
        mem.add_batch(*transicao(8))
    lote, idx, pesos = mem.sample(32)
    assert lote["obs"].shape == (32, *FORMA)
    assert lote["act"].shape == lote["rew"].shape == lote["done"].shape == (32,)
    assert lote["next_mask"].shape == (32, 3)
    assert pesos.shape == (32,) and np.allclose(pesos, 1.0)


# -------------------------------------------------------------- n passos
def test_n_step_return_is_the_discounted_sum():
    """`R^(n) = r_t + γ r_{t+1} + ... ` — conferido à mão com recompensas conhecidas."""
    mem = ReplayBuffer(100, FORMA, n_steps=3, gamma=0.5, num_envs=1)
    obs = np.zeros((1, *FORMA), np.float32)
    for r in (1.0, 2.0, 4.0, 8.0):
        mem.add_batch(obs, np.zeros(1, np.int32), np.array([r], np.float32),
                      obs, np.zeros(1, np.float32), np.ones((1, 3), bool))
    # a primeira transição emitida cobre r=1, 2, 4 -> 1 + 0,5*2 + 0,25*4 = 3
    assert mem.rew[0] == pytest.approx(3.0)


def test_n_step_flushes_the_queue_when_the_episode_ends():
    """Terminação curta não pode engolir transições na fila."""
    mem = ReplayBuffer(100, FORMA, n_steps=5, gamma=0.9, num_envs=1)
    obs = np.zeros((1, *FORMA), np.float32)
    for i in range(3):
        d = np.array([1.0 if i == 2 else 0.0], np.float32)
        mem.add_batch(obs, np.zeros(1, np.int32), np.ones(1, np.float32), obs, d,
                      np.ones((1, 3), bool))
    assert len(mem) == 3, "as 3 transições do episódio curto têm que sair da fila"


def test_each_env_keeps_its_own_n_step_queue():
    """Sem filas separadas, transições de ambientes diferentes se misturariam."""
    mem = ReplayBuffer(100, FORMA, n_steps=3, gamma=1.0, num_envs=2)
    obs = np.zeros((2, *FORMA), np.float32)
    for _ in range(3):
        mem.add_batch(obs, np.zeros(2, np.int32), np.array([1.0, 10.0], np.float32),
                      obs, np.zeros(2, np.float32), np.ones((2, 3), bool))
    emitidas = sorted(mem.rew[:len(mem)].tolist())
    assert emitidas == pytest.approx([3.0, 30.0]), \
        "as recompensas dos dois ambientes vazaram uma para a outra"


# -------------------------------------------------------------------- PER
def test_per_returns_importance_weights_in_range():
    mem = PrioritizedReplayBuffer(500, FORMA, num_envs=8)
    for _ in range(20):
        mem.add_batch(*transicao(8))
    _, idx, pesos = mem.sample(32, beta=0.4)
    assert pesos.shape == (32,)
    assert (pesos > 0).all() and pesos.max() == pytest.approx(1.0)


def test_per_favors_high_priority_transitions():
    mem = PrioritizedReplayBuffer(64, FORMA, num_envs=1, alpha=1.0)
    for _ in range(64):
        mem.add_batch(*transicao(1))
    prioridades = np.full(64, 0.01)
    prioridades[7] = 100.0
    mem.update_priorities(np.arange(64), prioridades)
    _, idx, _ = mem.sample(200)
    assert (idx == 7).mean() > 0.5, "a transição prioritária quase não foi sorteada"


def test_new_transitions_enter_with_max_priority():
    """Toda transição precisa ser vista ao menos uma vez antes de ser despriorizada."""
    mem = PrioritizedReplayBuffer(32, FORMA, num_envs=1)
    mem.add_batch(*transicao(1))
    mem.update_priorities([0], [5.0])
    mem.add_batch(*transicao(1))
    assert mem.arvore[1] >= mem.arvore[0], "a transição nova entrou com prioridade baixa"


def test_beta_correction_shrinks_weight_spread():
    """β maior = correção mais forte; os pesos ficam mais desiguais."""
    mem = PrioritizedReplayBuffer(128, FORMA, num_envs=1, alpha=1.0)
    for _ in range(128):
        mem.add_batch(*transicao(1))
    p = np.linspace(0.01, 10.0, 128)
    mem.update_priorities(np.arange(128), p)
    _, _, w_baixo = mem.sample(64, beta=0.1)
    _, _, w_alto = mem.sample(64, beta=1.0)
    assert w_alto.std() > w_baixo.std()


def test_uniform_buffer_ignores_priority_updates():
    """A interface é a mesma; o efeito não. Assim o agente não precisa de `if`."""
    mem = ReplayBuffer(32, FORMA, num_envs=1)
    mem.add_batch(*transicao(1))
    mem.update_priorities([0], [99.0])       # não pode explodir
    assert len(mem) == 1
