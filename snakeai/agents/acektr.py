"""ACEKTR — o ACKTR com EK-FAC no lugar do K-FAC.

*Eigenvalue-corrected Kronecker factorization* (George et al., 2018). Uma troca só, no
mesmo lugar em que o ACKTR já trocou o gradiente comum pelo natural: **o
pré-condicionador**. Herda o rollout, o GAE, o agendamento de entropia, o bootstrap de
truncamento, a região de confiança e a calibração — tudo. `_cria_precondicionador` é o
único método sobrescrito, e `tests/test_ekfac.py` confere que continua sendo o único.

O que o EK-FAC conserta
-----------------------
O K-FAC faz duas coisas de uma vez e só uma delas se justifica. De
`A ⊗ G = (U_A ⊗ U_G)(S_A ⊗ S_G)(U_A ⊗ U_G)ᵀ` ele tira **uma base** — os autovetores, a KFE
— e **uma escala por eixo** dessa base. A base é uma aproximação defensável dos autovetores
da Fisher; as escalas são obrigadas a ter forma de produto, `λ_A(j)·λ_G(i)`, e essa
restrição não vem de lugar nenhum além de ter saído junto na fatoração.

O EK-FAC fica com a base e mede as escalas: `s*_{ji} = E_n[((U_Aᵀ ∇W_n U_G)_{ji})²]`, o
segundo momento verdadeiro do gradiente projetado. Pelo Teorema 2 do paper, `s*` é a melhor
escala diagonal possível naquela base em norma de Frobenius; pelo Teorema 3, o EK-FAC nunca
é pior que o K-FAC. Não é uma heurística com um parâmetro a mais — é o mínimo de um problema
de mínimos quadrados do qual o K-FAC é um ponto qualquer.

E sai barato porque o gradiente por amostra de uma camada é um produto externo, e projetar
um produto externo é projetar cada lado: a média dos quadrados vira **um produto de
matrizes**. Ver `snakeai/kfac.py`, classe `EKFac`.

A previsão que este repositório pode testar
--------------------------------------------
O docstring do `ACKTR` registra uma medição incômoda: numa execução de 5 M passos, a KL
**entregue** ficou sistematicamente acima da pedida — 11,8× no primeiro quinto, caindo para
4,4× no último. O diagnóstico escrito lá foi: `Δᵀ∇ = ΔᵀF̃Δ` usa a Fisher *aproximada*,
enquanto a KL medida é a da política de verdade; onde `F̃` subestima a curvatura, o passo
sai grande demais. E o fator encolher conforme os fatores amadurecem é consistente com isso.

Se esse diagnóstico estiver certo, o EK-FAC tem que **encolher o fator**: ele aproxima `F`
melhor, por teorema, na mesma base. O `kl_fator` da calibração é exatamente esse número, já
registrado a cada atualização. É uma previsão falsificável e barata:

* `kl_fator` do ACEKTR mais perto de 1 que o do ACKTR → o diagnóstico se sustenta, e a
  correção de autovalores era a peça que faltava;
* `kl_fator` igual → o desvio vem de outro lugar (a diagonalidade por blocos, a hipótese de
  homogeneidade espacial da convolução, ou a própria aproximação quadrática da KL), e a
  seção da revisão precisa ser reescrita.

Nos dois casos o repositório aprende algo que hoje é suposição. Ver `docs/EKFAC.md`.

E a previsão que **não** se pode fazer
--------------------------------------
Que o ACEKTR ganhe do ACKTR na arena. Aproximar melhor a Fisher é uma afirmação sobre a
matriz, não sobre o score final — e a primeira execução longa do ACKTR mostrou que neste
domínio a dispersão entre sementes (desvio 9,63) engole diferenças bem maiores que as
plausíveis aqui. O que este algoritmo entrega com certeza é uma **medida**:
`ekfac_desvio` diz o quanto a Fisher deste problema deixa de ser um produto de Kronecker,
que é uma pergunta que ninguém tinha respondido sobre o `snake-arena`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kfac import EKFac
from .acktr import ACKTR, ACKTRConfig

__all__ = ["ACEKTRConfig", "ACEKTR"]


@dataclass
class ACEKTRConfig(ACKTRConfig):
    #: Média móvel de `s*` entre atualizações — e **mais rápida** que a de `A` e `G`, de
    #: propósito.
    #:
    #: `kfac_ema = 0,95` faz sentido porque `A` e `G` são acumulados *através* das
    #: reconstruções da base: eles nunca são zerados, e o que a média móvel absorve é só o
    #: ruído de lote. `s*` não é assim — ele descreve os eixos de **uma** base e é
    #: reiniciado no palpite do K-FAC toda vez que a base muda. Com `inv_every = 10`, uma
    #: média móvel de 0,95 gastaria a janela inteira saindo do palpite e o EK-FAC nunca
    #: chegaria a usar o que mediu. Meia-vida de uma atualização deixa as medições
    #: dominarem em três ou quatro passos, com folga dentro da janela.
    #:
    #: `1.0` desliga a medição e o EK-FAC vira **exatamente** o K-FAC. Não é uma
    #: curiosidade: é o controle que `tests/test_ekfac.py` usa para provar que a única
    #: diferença entre os dois agentes é a correção de autovalores.
    ema_escalas: float = 0.5


class ACEKTR(ACKTR):
    """ACKTR com EK-FAC. Ver o docstring do módulo para o porquê da troca única."""

    algo = "acektr"

    def __init__(self, cfg: ACEKTRConfig = None, model=None, variant=None):
        super().__init__(cfg or ACEKTRConfig(), model=model, variant=variant)

    def _cria_precondicionador(self):
        c = self.cfg
        return EKFac(self.model, damping=c.damping, ema=c.kfac_ema,
                     inv_every=c.inv_every,
                     ema_escalas=getattr(c, "ema_escalas", 0.5))

    @staticmethod
    def _variante_da_regiao(cfg):
        """As marcas do ACKTR, mais as duas que são deste algoritmo.

        `inv_every` ganha marca **aqui e não lá** porque muda de significado: no K-FAC ele
        é só a frequência de refatoração; no EK-FAC ele é o eixo de amortização que o paper
        propõe — base rara, escalas sempre — e portanto define de que regime a execução é.
        Ver `docs/EKFAC.md`.
        """
        marcas = [ACKTR._variante_da_regiao(cfg)]
        if cfg.inv_every != type(cfg).inv_every:
            marcas.append(f"base{cfg.inv_every}")
        if getattr(cfg, "ema_escalas", 0.5) >= 1.0:
            marcas.append("sem_correcao")
        return "+".join(marcas)

    def update(self, lote):
        saida = super().update(lote)
        #: O quanto `s*` já se afastou do palpite do K-FAC. É o número que distingue
        #: "o EK-FAC não ajudou aqui" de "o EK-FAC não está fazendo nada" — dois
        #: resultados idênticos na curva e opostos na leitura.
        saida["ekfac_desvio"] = self.kfac.desvio_de_kronecker()
        return saida
