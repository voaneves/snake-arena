"""Agentes — cada um implementa `iterate()`; o laço comum vem de `base.py`."""

from .a2c import A2C, A2CConfig
from .acektr import ACEKTR, ACEKTRConfig
from .acktr import ACKTR, ACKTRConfig
from .acer import ACER, ACERConfig, retrace
from .alphazero import AlphaZero, AlphaZeroConfig
from .base import AgentBase, BaseConfig
from .dreamerv3 import DreamerV3, DreamerV3Config
from .dqn import DQN, DQNConfig
from .lbc import LBC, LBCConfig, MisturaBoltzmann, vtrace
from .muzero import MuZero, MuZeroConfig
from .ppo import PPO, PPOConfig, compute_gae
from .rainbow import Rainbow, RainbowConfig
from .soap import (SOAP, PoliticaComOpcoes, SOAPConfig, gae_de_opcoes,
                   vantagem_de_opcao)

__all__ = [
    "AgentBase", "BaseConfig",
    "PPO", "PPOConfig", "compute_gae",
    "A2C", "A2CConfig",
    "ACER", "ACERConfig", "retrace",
    "ACKTR", "ACKTRConfig",
    "ACEKTR", "ACEKTRConfig",
    "AlphaZero", "AlphaZeroConfig",
    "DQN", "DQNConfig",
    "DreamerV3", "DreamerV3Config",
    "LBC", "LBCConfig", "MisturaBoltzmann", "vtrace",
    "MuZero", "MuZeroConfig",
    "Rainbow", "RainbowConfig",
    "SOAP", "SOAPConfig", "PoliticaComOpcoes", "gae_de_opcoes", "vantagem_de_opcao",
]
