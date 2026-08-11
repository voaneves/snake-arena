"""Troncos e cabeças de rede — a arquitetura como eixo de comparação."""

from .classic import APELIDOS_LEGADOS, TRONCOS_CLASSICOS
from .registry import (
    TRONCOS,
    build_actor_critic,
    build_backbone,
    build_q_network,
    listar_troncos,
    resumo,
)
from .resnet import PRESETS, resnet

__all__ = [
    "TRONCOS", "TRONCOS_CLASSICOS", "APELIDOS_LEGADOS", "PRESETS",
    "build_actor_critic", "build_backbone", "build_q_network",
    "listar_troncos", "resnet", "resumo",
]
