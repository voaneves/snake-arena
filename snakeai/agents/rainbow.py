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
    #: **20, não 3.** O valor canônico do Rainbow é 3, e aqui ele não alcança a recompensa.
    #:
    #: O agente gasta ~12 passos por maçã. Com `n_steps=3` a decisão que o levou até a
    #: comida sai da janela antes de a recompensa chegar: a atribuição de crédito depende
    #: inteiramente do bootstrap, e o bootstrap depende das sincronias do alvo, que são
    #: dezenas num treino inteiro. Com 20 a maçã entra na mesma janela da decisão.
    #:
    #: Medido, com todo o resto igual: a decolagem sai de **~1,85 M** passos para **~700 k**,
    #: e a fome cai de 100% para 69,8% em 150 k passos enquanto o score de treino vai de
    #: 2,24 a 8,45. Antes disso o agente passava um milhão de passos parado em 100% de fome.
    #:
    #: O 20 vem do **Data-Efficient Rainbow** (van Hasselt et al., 2019), que usa
    #: `multi-step 20` justamente no regime de poucos dados. É um desvio declarado do
    #: Rainbow canônico, e o `γ**n` cai de 0,985 para 0,905 — o que também reduz o peso do
    #: bootstrap, que é o mecanismo que estava faltando. Ver §2.25.
    n_steps: int = 20

    #: Rainbow não usa ε-greedy: a exploração vem das noisy nets. Mantido em zero para
    #: que ligar `noisy=False` sem pensar não deixe o agente sem exploração nenhuma.
    eps_start: float = 0.0
    eps_end: float = 0.0

    #: **3e-4, o mesmo do DQN base deste repositório** — não os 6,25e-5 do paper.
    #:
    #: O padrão era 1e-4, justificado por "o paper usa LR menor que o DQN base". Usa mesmo,
    #: mas para **200 M de frames**; o orçamento aqui é 5 M, quarenta vezes menor. Herdar a
    #: taxa de um regime quarenta vezes mais longo é o mesmo erro de escala que estava no
    #: `target_update` (§2.20), com o mesmo formato de argumento.
    #:
    #: Medido: com `lr=1e-4` a decolagem acontece por volta de **4,6 M** passos e o
    #: orçamento acaba antes de a curva virar; com `3e-4` ela acontece aos **1,85 M**, e a
    #: execução termina em 26,99 ainda na inclinação máxima (fome caindo de 49% para 28%,
    #: passos por episódio subindo de 380 para 466 nos últimos 450 k). Ver
    #: `docs/REVISAO_ALGORITMOS.md` §2.21.
    lr: float = 3e-4

    #: Fica em 4, o padrão do DQN base — **e isto é uma decisão contra a referência**.
    #:
    #: O Rainbow do `Kaixhin` treina com lote 32 uma vez a cada 4 passos, ou seja sorteia
    #: **8 amostras por passo de ambiente**; cada transição é revisitada ~8 vezes. Com
    #: `learn_every=4` e lote 512 nós sorteamos `512/256 = 2,0` — um quarto disso.
    #: `learn_every=1` daria exatamente 8,0, e chegou a ser o padrão aqui por algumas horas.
    #:
    #: Voltou para 4 porque a execução que **de fato funcionou** — decolagem aos 700 k com
    #: `n_steps=20` — rodou com 4, e trocar as duas coisas juntas mediria a soma. O
    #: `learn_every=1` continua sendo a hipótese mais forte para a próxima ablação, e custa
    #: 4× o trabalho de gradiente. Ver §2.23.
    learn_every: int = 4

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

    #: 250, que é o valor do DQN base — e também está **abaixo** da referência, que
    #: sincroniza a cada 2.000 atualizações (`Kaixhin`: 8.000 passos com uma atualização a
    #: cada 4). Pela unidade certa — quantas atualizações a rede alvo fica parada — nós
    #: sincronizamos 8× mais que o canônico.
    #:
    #: Esta linha oscilou entre 1.000, 250 e 1.000 ao longo da investigação. Ficou em 250
    #: pelo mesmo motivo do `learn_every`: é o valor da execução que funcionou. A tensão é
    #: real e está declarada — com poucas atualizações no total, ou o alvo é fiel e propaga
    #: pouco, ou propaga e é infiel — e o `n_steps=20` aliviou justamente essa tensão, já
    #: que `γ**20 = 0,905` reduz o peso do bootstrap. Ver §2.20 e §2.23.
    target_update: int = 250


class Rainbow(DQN):
    algo = "rainbow"

    def __init__(self, cfg: RainbowConfig = None, variant=None):
        cfg = cfg or RainbowConfig()
        super().__init__(cfg, variant=variant or self._variante(cfg))
        if not cfg.noisy and cfg.eps_start == 0.0:
            raise ValueError(
                "com `noisy=False` e `eps_start=0` o agente não explora de forma nenhuma. "
                "Ligue um dos dois — ou use `DQN` diretamente para ablação."
            )

    @staticmethod
    def _variante(cfg):
        """`completo`, mais uma marca por desvio da composição canônica.

        A composição canônica mora no código — e a **identidade da execução** também tem de
        morar. Até agora uma execução com `n_steps` diferente só se distinguia se quem a
        rodou lembrasse de passar `variant="completo+n3"` na mão. Esquecer faria as duas
        dividirem `(algo, variant, seed)` e virarem **uma** curva na arena: a de 0,57
        arrastaria a de 65,43 sem deixar rastro, e a média resultante não descreveria
        execução nenhuma.

        Os nomes que saem daqui são os mesmos que as execuções de agosto receberam à mão,
        então o histórico não se move.
        """
        padrao = type(cfg)
        marcas = []
        if cfg.n_steps != padrao.n_steps:
            marcas.append(f"n{cfg.n_steps}")
        for nome in ("double", "dueling", "per", "noisy"):
            if not getattr(cfg, nome):
                marcas.append(f"sem_{nome}")
        if cfg.n_atoms != padrao.n_atoms:
            marcas.append(f"c51x{cfg.n_atoms}" if cfg.n_atoms else "sem_c51")
        return "+".join(["completo"] + marcas)

    @staticmethod
    def componentes(cfg):
        """Quais dos seis estão de fato ligados. Útil para rotular ablações."""
        return {
            "double": cfg.double, "dueling": cfg.dueling, "per": cfg.per,
            "noisy": cfg.noisy, "n_steps": cfg.n_steps > 1, "c51": cfg.n_atoms > 0,
        }
