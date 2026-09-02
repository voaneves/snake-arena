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

A primeira tentativa de responder isso **não valeu** — e por quê
----------------------------------------------------------------
A execução de 01/09 (`acektr/resnet_small/seed0`) fechou em 71,07 com 17,6% de tabuleiros
cheios, contra 89,78 e 89,7% do ACKTR, e a mediana de `kl_fator` saiu 19,98 contra 18,71 —
o que se leria como "o EK-FAC não aproxima melhor **e** ainda joga pior". As duas leituras
são inválidas, pelo mesmo motivo: **o par não estava pareado**.

O `A2CConfig.rollout` foi de 16 para 5 em 21/08, um dia depois das três execuções gravadas
do ACKTR, e o `ACKTRConfig` herdava esse campo. A execução do ACEKTR rodou com `T = 5`; as
três do ACKTR, com `T = 16`. Interpolando `train_score_mean` na mesma grade de passos de
ambiente:

===========================  ======  ======  ======  ======  ======
execução                     1,0 M   1,5 M   2,0 M   3,0 M   5,0 M
===========================  ======  ======  ======  ======  ======
faixa das 3 sementes, T = 16 26–29   31–37   40–64   67–72   73–81
ACEKTR, T = 5                29,3    36,8    44,1    55,5    63,5
===========================  ======  ======  ======  ======  ======

Até 1,5 M o ACEKTR está **no topo** da faixa. A separação começa em 2 M — logo depois de
`shaping_frac` levar o shaping a zero em 1,25 M, quando a recompensa deixa de ser densa e
o crédito passa a depender da janela do GAE. É o que a conta prevê: com `γλ = 0,945`,
`0,945⁵ = 76%` do peso fica no bootstrap contra `0,945¹⁶ = 40%`.

E **não foi falta de passo**, que era a suspeita óbvia. Somando `√KL` sobre as
atualizações, o ACEKTR acumulou **202** contra 57–73 das três sementes do ACKTR: ele andou
3,6× mais e chegou 20 pontos abaixo. O que faltou foi direção, não distância — o que também
descarta subir `kl_max` como conserto.

Então `ACKTRConfig` voltou a declarar `rollout = 16`, e a §5 da `docs/EKFAC.md` continua
**sem resposta**: ela precisa de duas execuções na mesma semente e no mesmo orçamento de
crédito, que é o que a próxima execução deste notebook produz.

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
    ema_escalas: float = 0.8

    #: **A base rara, as escalas sempre** — o regime que o paper propõe, e que o
    #: `docs/EKFAC.md` §3.2 registra como deliberadamente desligado até aqui.
    #:
    #: O default do ACKTR é 10, e lá ele significa só "de quantas em quantas atualizações
    #: refatorar". No EK-FAC ele significa outra coisa: é o eixo de amortização. `s*`
    #: descreve os eixos de **uma** base e é reiniciado no palpite do K-FAC toda vez que a
    #: base muda — com uma janela de 10 e `ema_escalas = 0,5`, a medição mal saía do
    #: palpite antes de ser jogada fora. Com 50, sobram ~30 atualizações por janela
    #: rodando com escalas de fato medidas, e a `eigh` cara sai 5× menos vezes.
    #:
    #: `ema_escalas` sobe junto, de 0,5 para 0,8: a janela agora comporta uma média sobre
    #: ~5 lotes em vez de ~2, e `s*` medido em 2 lotes é ruidoso — dividir por um
    #: autovalor subestimado por ruído amplifica exatamente a direção que o lote não
    #: sabia estimar. É a troca de um estimador enviesado e liso (o do K-FAC) por um não
    #: enviesado e liso, em vez de por um não enviesado e barulhento.
    #:
    #: Isto sai do pareamento `12_acektr × 08_acktr` — que passa a medir duas variáveis —
    #: e por isso a variante ganha `+base50` **sempre**, mesmo sendo o default daqui.
    inv_every: int = 50

    #: Ligada aqui, ao contrário do ACKTR. Ver `ACKTRConfig.kl_cal_debias`: o ACKTR mantém
    #: `False` para continuar reproduzindo as três execuções gravadas; o ACEKTR não tem
    #: execução boa para preservar, e o transitório de ~50 atualizações é 8% do orçamento.
    kl_cal_debias: bool = True

    #: Prior do fator de calibração. As execuções longas assentaram entre 15 e 25, e o
    #: diagnóstico curto (`docs/diag_acktr_kl.json`) mediu 7,0 a 7,4 na forma do contrato.
    #: 15 fica no meio, e o erro é assimétrico: cauteloso demais custa alguns passos
    #: curtos no começo, ousado demais colapsa a entropia e não tem volta.
    kl_fator_inicial: float = 15.0

    @classmethod
    def credito_longo(cls, **kw):
        """O braço que dobra a janela de crédito: `rollout = 32`.

        Sai do mesmo achado que restaurou o 16. Depois que o shaping zera, `0,945^T` decide
        quanto do peso do GAE sobra no bootstrap — 76% com T=5, 40% com T=16, **16% com
        T=32**. E a execução de 01/09 mostrou que distância não é o gargalo: ela acumulou
        `Σ√KL` de 202 contra 57–73 do ACKTR e chegou 20 pontos abaixo. Se movimento sobra e
        direção falta, cortar as atualizações pela metade (610 → 305) para dobrar o horizonte
        é a troca que faz sentido testar — e é falsificável: se o score cair, movimento
        também estava mordendo, e a leitura de cima precisa de asterisco.

        Ganha `sufixo_variante="T32"` porque rollout é orçamento, não região de confiança: o
        nome da variante não o marca sozinho, e duas janelas de crédito diferentes com a
        mesma identidade `(algo, variant, seed)` viram uma curva só na arena. É a mesma
        mecânica do `A2CConfig.esparso()`.
        """
        kw.setdefault("sufixo_variante", "T32")
        return cls(rollout=32, **kw)


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
        # comparado ao default do **ACKTR**, não ao daqui: `inv_every` é o eixo de
        # amortização do paper, e uma execução no regime dele não é a mesma coisa que uma
        # execução pareada com o `08_acktr`. Como o default do ACEKTR passou a ser 50, a
        # marca aparece sempre — que é o ponto: ela é o que impede a identidade
        # `(algo, variant, seed)` de colidir com a execução de 01/09, que rodou com 10.
        if cfg.inv_every != ACKTRConfig.inv_every:
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
