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

    #: Rainbow não usa ε-greedy: a exploração vem das noisy nets. Mantido em zero para
    #: que ligar `noisy=False` sem pensar não deixe o agente sem exploração nenhuma.
    eps_start: float = 0.0
    eps_end: float = 0.0

    lr: float = 1e-4          # o paper usa LR menor que o DQN base

    #: **O suporte tem de ser simétrico, e isto não é estética.** Na inicialização os
    #: logits são ~0, então a softmax do C51 é uniforme sobre o suporte e o `Q` inicial é
    #: o **ponto médio** dele — não zero. Com o `[-2, 60]` que este repositório usava, todo
    #: estado nascia valendo **+29**, e esse valor é um ponto fixo do bootstrap: o alvo de
    #: uma transição não terminal é `r + γ³·29 ≈ 28,6`, que é o que a rede já prevê. A
    #: única correção vinha das transições terminais (`-1`), e como a fome é **truncamento**
    #: — bootstrap, `done=0` — 90% dos fins de episódio não corrigiam nada. O resultado
    #: medido: `Q` médio preso em +28,6 por 120 mil passos, o agente aprendendo só a evitar
    #: colisão, e a maçã valendo `+1` sobre uma linha de base de 29 — 3% do sinal.
    #:
    #: `[-24, 24]` satisfaz os dois requisitos, que puxam em direções opostas. **Simétrico**
    #: (ponto médio 0), para o `Q` nascer em zero. E **largo o bastante**: um jogo perfeito
    #: de 97 maçãs a ~10 passos cada com γ=0,995 rende 20,3, então 24 cobre com 18% de folga
    #: — `[-20, 20]` reprova por 0,3 ponto, e o teste pega isso.
    #:
    #: Com 121 átomos o `Δz` é **0,4**, exatamente a resolução do C51 canônico do Atari, e
    #: existe um átomo em zero (índice 60). Uma maçã passa a valer 2,5 átomos em vez dos
    #: 0,8 do suporte antigo, cujo `Δz` de 1,24 era maior que a própria recompensa.
    #: Ver `docs/REVISAO_ALGORITMOS.md` §2.8.
    v_min: float = -24.0
    v_max: float = 24.0
    n_atoms: int = 121

    #: **Contado em atualizações de gradiente**, e o número real é metade do que o
    #: repositório gravava — ver §2.18. O orçamento de 5 M passos compra ~18.500
    #: atualizações reais, não ~39.000. Com `target_update=1.000` isso dava **18,6
    #: sincronizações do alvo no treino inteiro**: a informação de valor se propagava
    #: dezenove vezes em 5 M passos. O DQN da Nature faz ~1.250 e o Rainbow do paper
    #: ~6.250.
    #:
    #: 250 é o valor do DQN base deste repositório, que decola aos 750 k, e dá 74
    #: sincronizações. A razão de 4× sobre o DQN que o comentário antigo defendia foi
    #: construída sobre a contagem dobrada; corrigida a contagem, ela não se sustenta.
    target_update: int = 250


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
