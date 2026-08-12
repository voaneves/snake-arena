"""Rainbow — os seis componentes da família DQN, todos ligados.

Não é um algoritmo novo: é o `DQN` deste repositório com as seis flags ativadas. Existe
como classe própria por dois motivos práticos, os dois sobre honestidade do gráfico:

1. **Linha própria na arena.** Como variante de DQN, o Rainbow apareceria com a cor do DQN
   e o leitor teria de saber decifrar `dqn · double+dueling+per+noisy+3steps+c51`. Como
   algoritmo próprio, ele tem cor, nome e rótulo direto.
2. **A composição canônica fica no código, não na cabeça de quem configura.** "Rainbow" é
   uma combinação específica do paper; deixá-la como seis argumentos soltos convida a
   variações silenciosas que depois viram "meu Rainbow deu diferente do seu".

Os seis componentes, e o que cada um resolve:

======================  =====================================================
componente              o problema que ataca
======================  =====================================================
``double``              viés otimista do `max` no alvo de TD
``dueling``             separar "este estado é bom" de "esta ação é boa"
``per``                 aprender mais das transições surpreendentes
``n_steps``             propagar recompensa mais rápido que um passo por vez
``noisy``               exploração dependente do estado, sem ε
``n_atoms`` (C51)       aprender a distribuição do retorno, não só a média
======================  =====================================================

Cada um continua mensurável isolado pelo `DQN` — este agente é a soma, não a única forma
de usá-los.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dqn import DQN, DQNConfig

__all__ = ["RainbowConfig", "Rainbow"]


@dataclass
class RainbowConfig(DQNConfig):
    """`DQNConfig` com a composição canônica do paper como padrão."""

    double: bool = True
    dueling: bool = True
    per: bool = True
    noisy: bool = True
    n_steps: int = 3
    n_atoms: int = 51

    #: Rainbow não usa ε-greedy: a exploração vem das noisy nets. Mantido em zero para
    #: que ligar `noisy=False` sem pensar não deixe o agente sem exploração nenhuma.
    eps_start: float = 0.0
    eps_end: float = 0.0

    lr: float = 1e-4          # o paper usa LR menor que o DQN base
    target_update: int = 8_000


class Rainbow(DQN):
    algo = "rainbow"

    def __init__(self, cfg: RainbowConfig = None, variant=None):
        cfg = cfg or RainbowConfig()
        super().__init__(cfg, variant=variant or "completo")
        if not cfg.noisy and cfg.eps_start == 0.0:
            raise ValueError(
                "com `noisy=False` e `eps_start=0` o agente não explora de forma nenhuma. "
                "Ligue um dos dois — ou use `DQN` diretamente para ablação."
            )

    @staticmethod
    def componentes(cfg):
        """Quais dos seis estão de fato ligados. Útil para rotular ablações."""
        return {
            "double": cfg.double, "dueling": cfg.dueling, "per": cfg.per,
            "noisy": cfg.noisy, "n_steps": cfg.n_steps > 1, "c51": cfg.n_atoms > 0,
        }
