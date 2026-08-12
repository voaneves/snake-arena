"""Agentes — cada um implementa `iterate()`; o laço comum vem de `base.py`."""

from .a2c import A2C, A2CConfig
from .acer import ACER, ACERConfig, retrace
from .alphazero import AlphaZero, AlphaZeroConfig
from .base import AgentBase, BaseConfig
from .dqn import DQN, DQNConfig
from .muzero import MuZero, MuZeroConfig
from .ppo import PPO, PPOConfig, compute_gae

__all__ = [
    "AgentBase", "BaseConfig",
    "PPO", "PPOConfig", "compute_gae",
    "A2C", "A2CConfig",
    "ACER", "ACERConfig", "retrace",
    "AlphaZero", "AlphaZeroConfig",
    "DQN", "DQNConfig",
    "MuZero", "MuZeroConfig",
]
