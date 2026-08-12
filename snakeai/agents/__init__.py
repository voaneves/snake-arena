"""Agentes — cada um implementa `iterate()`; o laço comum vem de `base.py`."""

from .a2c import A2C, A2CConfig
from .acktr import ACKTR, ACKTRConfig
from .acer import ACER, ACERConfig, retrace
from .alphazero import AlphaZero, AlphaZeroConfig
from .base import AgentBase, BaseConfig
from .dreamerv3 import DreamerV3, DreamerV3Config
from .dqn import DQN, DQNConfig
from .muzero import MuZero, MuZeroConfig
from .ppo import PPO, PPOConfig, compute_gae
from .rainbow import Rainbow, RainbowConfig

__all__ = [
    "AgentBase", "BaseConfig",
    "PPO", "PPOConfig", "compute_gae",
    "A2C", "A2CConfig",
    "ACER", "ACERConfig", "retrace",
    "ACKTR", "ACKTRConfig",
    "AlphaZero", "AlphaZeroConfig",
    "DQN", "DQNConfig",
    "DreamerV3", "DreamerV3Config",
    "MuZero", "MuZeroConfig",
    "Rainbow", "RainbowConfig",
]
