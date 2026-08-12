"""Onde o tempo de treino é gasto: ambiente (CPU, NumPy) ou rede (GPU)?

A pergunta prática é "vale a pena montar GPU?". A resposta depende de uma fração que dá
para medir: quanto do tempo de parede está dentro de `VecSnake.step` — que é NumPy puro e
**não** acelera com GPU nenhuma — e quanto está dentro do Keras.

Uso:
    python -m tools.perfil_dispositivo            # todos
    python -m tools.perfil_dispositivo ppo dqn
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from snakeai.env.vec_snake import VecSnake

ACUM = {}


def cronometra(alvo, nome):
    """Envolve um método, somando o tempo em `ACUM[nome]`."""
    orig = getattr(alvo, nome.split(".")[-1])
    chave = nome

    def envolto(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            ACUM[chave] = ACUM.get(chave, 0.0) + time.perf_counter() - t0

    setattr(alvo, nome.split(".")[-1], envolto)
    return orig


def bench_env(num_envs=512, passos=200):
    env = VecSnake(num_envs=num_envs, rng=np.random.default_rng(0))
    env.reset()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(passos):
        env.step(rng.integers(0, 3, size=num_envs))
    dt = time.perf_counter() - t0
    return num_envs * passos / dt


def perfila(nome, cfg_cls, agente_cls, iters=3, **kw):
    ACUM.clear()
    orig_step = cronometra(VecSnake, "step")
    try:
        ag = agente_cls(cfg_cls(**kw))
        ag.iterate()  # aquece: traça o grafo, aloca
        ACUM.clear()
        passos0 = ag.global_step
        t0 = time.perf_counter()
        for _ in range(iters):
            ag.iterate()
        total = time.perf_counter() - t0
        passos = ag.global_step - passos0
    finally:
        VecSnake.step = orig_step

    env_t = ACUM.get("step", 0.0)
    return {
        "algo": nome,
        "passos": passos,
        "total_s": total,
        "env_s": env_t,
        "keras_s": total - env_t,
        "frac_env": env_t / total,
        "passos_por_s": passos / total,
    }


def main(quais):
    print(f"{'-' * 78}\nAmbiente puro (NumPy, sem rede nenhuma)")
    for n in (128, 512, 1024):
        print(f"  {n:>5} ambientes: {bench_env(n):>10,.0f} passos/s")

    comum = dict(eval_every_steps=10**9, log_every_steps=10**9,
                 salvar_gif=False, salvar_grafico=False, net="resnet_tiny")
    linhas = []

    if "ppo" in quais:
        from snakeai.agents import PPO, PPOConfig
        linhas.append(perfila("ppo", PPOConfig, PPO, num_envs=512, rollout=24, **comum))

    if "dqn" in quais:
        from snakeai.agents import DQN, DQNConfig
        linhas.append(perfila("dqn", DQNConfig, DQN, num_envs=256, batch_size=256,
                              memory_size=50_000, warmup_steps=0, learn_every=4,
                              total_steps=10**7, **comum))

    if "rainbow" in quais:
        from snakeai.agents import Rainbow, RainbowConfig
        linhas.append(perfila("rainbow", RainbowConfig, Rainbow, num_envs=256,
                              batch_size=256, memory_size=50_000, warmup_steps=0,
                              learn_every=4, total_steps=10**7, **comum))

    if "acer" in quais:
        from snakeai.agents import ACER, ACERConfig
        linhas.append(perfila("acer", ACERConfig, ACER, num_envs=128, rollout=16, **comum))

    if "alphazero" in quais:
        from snakeai.agents import AlphaZero, AlphaZeroConfig
        linhas.append(perfila("alphazero", AlphaZeroConfig, AlphaZero, iters=1,
                              num_envs=16, rollout=8, num_simulations=32,
                              total_steps=10**6, **comum))

    print(f"\n{'-' * 78}\nTreino real, por algoritmo")
    print(f"{'algo':<11}{'passos/s':>10}{'% ambiente':>12}{'% keras':>9}"
          f"{'  5M passos levariam':>22}")
    for r in linhas:
        horas = 5_000_000 / r["passos_por_s"] / 3600
        print(f"{r['algo']:<11}{r['passos_por_s']:>10,.0f}"
              f"{r['frac_env'] * 100:>11.0f}%{(1 - r['frac_env']) * 100:>8.0f}%"
              f"{horas:>19.1f} h")
    return linhas


if __name__ == "__main__":
    todos = ["ppo", "dqn", "rainbow", "acer", "alphazero"]
    main([a for a in sys.argv[1:] if a in todos] or todos)
