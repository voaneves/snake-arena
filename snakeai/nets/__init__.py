"""Troncos e cabeças de rede — a arquitetura como eixo de comparação."""

from .classic import APELIDOS_LEGADOS, TRONCOS_CLASSICOS
from .heads import (
    NoisyDense,
    distributional_head,
    dueling_head,
    q_de_distribuicao,
    suporte_c51,
)
from .registry import (
    TRONCOS,
    build_actor_critic,
    build_actor_critic_populacao,
    build_option_actor_critic,
    build_backbone,
    build_policy_q,
    build_q_network,
    listar_troncos,
    resumo,
)
from .muzero import build_dinamica, build_predicao, build_representacao
from .resnet import PRESETS, resnet

__all__ = [
    "TRONCOS", "TRONCOS_CLASSICOS", "APELIDOS_LEGADOS", "PRESETS",
    "build_actor_critic", "build_actor_critic_populacao",
    "build_option_actor_critic", "build_backbone", "build_q_network", "build_policy_q",
    "listar_troncos", "resnet", "resumo",
    "build_representacao", "build_dinamica", "build_predicao",
    "NoisyDense", "dueling_head", "distributional_head", "q_de_distribuicao",
    "suporte_c51",
]
