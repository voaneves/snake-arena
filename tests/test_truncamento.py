"""Fome é truncamento, não terminação — em **todos** os agentes.

O `VecSnake` marca `done` quando a cobra passa fome, mas exporta `trunc_idx`, `final_obs` e
`final_mask` justamente porque o episódio *continuaria*. Cortar o retorno ali sem o
`γ·V(s_final)` ensina que sobreviver muito termina em −0,5, que é o oposto do que uma cobra
longa precisa aprender — e enviesa **para baixo** exatamente a região do espaço de estados
em que o agente está preso.

Durante muito tempo só o PPO tratava isso. Cada teste aqui vale por um agente, e todos
falham no código anterior. Ver `docs/REVISAO_ALGORITMOS.md` §1.1.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from snakeai.agents.base import AgentBase
from snakeai.env.vec_snake import VecSnake

CFG = dict(net="resnet_tiny", total_steps=10 ** 6, salvar_gif=False, salvar_grafico=False)


def env_faminto(agente, n, seed=0):
    """Troca o ambiente do agente por um que passa fome quase imediatamente."""
    agente.env = VecSnake(n, 10, starve_base=1, rng=np.random.default_rng(seed))
    agente.obs, agente.mask = agente.env.reset()
    return agente


def espia_passos(agente):
    """Guarda o `info` de cada `env.step`, para achar o passo truncado."""
    infos, original = [], agente.env.step

    def espiao(*a, **kw):
        saida = original(*a, **kw)
        infos.append(saida[-1])
        return saida

    agente.env.step = espiao
    return infos


# ===================================================================== o auxiliar
def test_the_helper_adds_the_bootstrap_only_to_the_truncated_envs():
    info = {"trunc_idx": np.array([1, 3]),
            "final_obs": np.zeros((2, 10, 10, 5), np.float32),
            "final_mask": np.ones((2, 3), bool)}
    r = np.array([1.0, -0.5, 0.0, -0.5], np.float32)
    saida = AgentBase.bootstrap_truncados(info, r, np.array([10.0, 4.0]), gamma=0.5)

    np.testing.assert_allclose(saida, [1.0, -0.5 + 5.0, 0.0, -0.5 + 2.0])
    np.testing.assert_allclose(r, [1.0, -0.5, 0.0, -0.5], err_msg="não pode alterar a entrada")


def test_the_helper_is_a_no_op_without_starvation():
    r = np.array([1.0, 0.0], np.float32)
    assert AgentBase.bootstrap_truncados({"trunc_idx": np.array([], int)}, r, [], 0.99) is r


# ===================================================================== ACER
def test_acer_bootstraps_the_truncated_step():
    from snakeai.agents.acer import ACER, ACERConfig

    ag = env_faminto(ACER(ACERConfig(num_envs=4, rollout=4, memory_size=8, **CFG)), 4)
    infos = espia_passos(ag)
    guardados, add = [], ag.memoria.add
    ag.memoria.add = lambda *a, **kw: (guardados.append(a), add(*a, **kw))[1]

    for _ in range(40):
        ag.collect()
        if any(len(i["trunc_idx"]) for i in infos[-4:]):
            break
    t = next(k for k, i in enumerate(infos[-4:]) if len(i["trunc_idx"]))
    info = infos[-4:][t]
    rew_buf = guardados[-1][4]

    pi, q = ag._probs(tf.convert_to_tensor(info["final_obs"]),
                      tf.convert_to_tensor(info["final_mask"]))
    v = np.sum(pi.numpy() * q.numpy(), axis=1)
    bruto = rew_buf[t, info["trunc_idx"]] - ag.cfg.gamma * v
    assert np.all(np.abs(bruto - (-0.5)) < 1e-4), (
        "a recompensa guardada não traz o γ·V(s_final) do estado truncado")


# ===================================================================== DreamerV3
def test_dreamer_does_not_teach_that_starving_is_the_end():
    from snakeai.agents import DreamerV3, DreamerV3Config

    ag = env_faminto(DreamerV3(DreamerV3Config(
        preset="dreamer_tiny", num_envs=4, batch_size=4, seq_len=8, memory_size=50,
        warmup_steps=0, horizonte=5, collect_steps=8, eval_every_steps=10 ** 9,
        log_every_steps=10 ** 9, **CFG)), 4)
    infos = espia_passos(ag)
    guardados, add = [], ag.memoria.add
    ag.memoria.add = lambda *a, **kw: (guardados.append(a), add(*a, **kw))[1]

    for _ in range(30):
        ag.collect()
        if any(len(i["trunc_idx"]) for i in infos):
            break
    k = next(k for k, i in enumerate(infos) if len(i["trunc_idx"]))
    info, cont = infos[k], guardados[k][3]
    assert np.all(cont[info["trunc_idx"]] == 1.0), (
        "a cabeça `cont` aprende que morrer de fome é terminal — e ela nem enxerga o "
        "relógio de fome nos 5 canais do contrato")


# ===================================================================== busca
@pytest.mark.parametrize("qual", ["alphazero", "muzero"])
def test_search_agents_bootstrap_the_truncated_step(qual):
    """Com o valor fixado num número positivo conhecido, o alvo `z` do passo truncado tem
    que refletir `−0,5 + γ·V`. Sem o bootstrap ele vale −0,5 cravado, porque `vivo` fecha
    logo depois e o retorno de n passos não tem mais nada para somar."""
    if qual == "alphazero":
        from snakeai.agents.alphazero import AlphaZero as Classe, AlphaZeroConfig as Cfg
        extra = dict(num_simulations=2)
    else:
        from snakeai.agents.muzero import MuZero as Classe, MuZeroConfig as Cfg
        extra = dict(num_simulations=2, unroll=2)

    ag = env_faminto(Classe(Cfg(num_envs=4, rollout=4, memory_size=200, n_step=3,
                                batch_size=8, **extra, **CFG)), 4)
    VALOR = 10.0
    if qual == "alphazero":
        ag._avaliar = lambda obs, mask: (
            np.full((len(obs), 3), 1 / 3, np.float32), np.full(len(obs), VALOR, np.float32))
    else:
        original = ag._repr_predicao
        ag._repr_predicao = lambda obs, mask: (
            original(obs, mask)[0], original(obs, mask)[1],
            tf.constant(np.full(len(obs), VALOR, np.float32)))

    infos = espia_passos(ag)
    guardados, add = [], ag._guardar
    ag._guardar = lambda *a, **kw: (guardados.append(a), add(*a, **kw))[1]

    T, N = 4, 4
    validos = T if qual == "alphazero" else T - ag.cfg.unroll
    conferidos = []
    for _ in range(60):
        antes = len(infos)
        ag.collect()
        janela = infos[antes:antes + T]
        conferidos = [(t, i) for t, inf in enumerate(janela) if t < validos
                      for i in inf["trunc_idx"]]
        if conferidos:
            break
    assert conferidos, "o cenário não produziu fome dentro da janela guardada"

    if qual == "alphazero":
        z = np.asarray(guardados[-1][3]).reshape(-1, N)          # (T, N)
        valores = [z[t, i] for t, i in conferidos]
    else:
        z = np.asarray(guardados[-1][4])                          # (validos·N, K+1)
        valores = [z[t * N + i, 0] for t, i in conferidos]

    for (t, i), v in zip(conferidos, valores):
        assert v > 1.0, (
            f"z[{t},{i}] = {v:.3f} — o passo truncado ficou com o retorno cortado "
            "em vez de −0,5 + γ·V(s_final)")


# ============================================== §2.4 · defasagem da rede alvo
def test_the_target_network_lag_is_counted_in_gradient_updates():
    """`target_update` contava **passos de ambiente**, e como uma atualização acontece a
    cada `learn_every × num_envs` = 256 passos, os 2.000 nominais viravam ~8 atualizações
    de defasagem — contra as ~2.000 das implementações de referência. Com o alvo colado na
    rede online, o Double DQN perde o efeito e o alvo deixa de ser ponto fixo.
    Ver `docs/REVISAO_ALGORITMOS.md` §2.4."""
    from snakeai.agents.dqn import DQN, DQNConfig

    ag = DQN(DQNConfig(net="resnet_tiny", num_envs=8, learn_every=1, batch_size=8,
                       warmup_steps=0, memory_size=500, target_update=3,
                       total_steps=10 ** 6, salvar_gif=False, salvar_grafico=False))
    sincronias, original = [], ag.target.set_weights

    def espiao(w):
        sincronias.append(ag._atualizacoes)
        return original(w)

    ag.target.set_weights = espiao
    for _ in range(12):
        ag.iterate()

    assert sincronias, "a rede alvo nunca sincronizou"
    # uma sincronia a cada 3 **atualizações**, não a cada 3 passos de ambiente
    intervalos = np.diff([0] + sincronias)
    assert set(intervalos) <= {3}, f"intervalos entre sincronias: {intervalos.tolist()}"


def test_every_value_based_agent_syncs_its_target_enough_times():
    """Mudar a **unidade** de `target_update` sem revisitar os valores absolutos quebra
    o agente pelo outro lado.

    Desde o teste acima o contador anda em atualizações de gradiente, não em passos de
    ambiente. O DQN teve o valor recalculado na hora (2.000 -> 250, ~1,3% do orçamento);
    o Rainbow ficou com os 8.000 canônicos, que na unidade nova sao 41% do orcamento --
    **duas** sincronizacoes no treino inteiro. Um alvo congelado por 40% do treino e tao
    ruim quanto um alvo colado na rede online: nos dois casos o Double DQN para de ter
    efeito. Este teste vale para qualquer agente de valor que venha depois.
    """
    from snakeai.agents.dqn import DQNConfig
    from snakeai.agents.rainbow import RainbowConfig

    for nome, cfg in (("dqn", DQNConfig()), ("rainbow", RainbowConfig())):
        orcamento = cfg.total_steps // (cfg.learn_every * cfg.num_envs)
        sincronias = orcamento // cfg.target_update
        assert sincronias >= 10, (
            f"{nome}: target_update={cfg.target_update} em ~{orcamento:,} atualizações "
            f"dá {sincronias} sincronizações — o alvo fica congelado quase o treino todo"
        )
        assert cfg.target_update >= 50, (
            f"{nome}: target_update={cfg.target_update} cola o alvo na rede online e "
            "anula o efeito do double"
        )
