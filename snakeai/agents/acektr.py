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
crédito.

A segunda tentativa também não valeu, e o motivo é outro
--------------------------------------------------------
Com o rollout restaurado, a execução de 02/09 (`resnet_small+base50`) fechou em 74,47 com
**0,4%** de tabuleiros cheios — média maior que a de 01/09 (71,07) e taxa de vitória 44×
menor. O confundidor dessa vez foi meu: `inv_every` tinha ido de 10 para 50, que é o regime
de amortização do paper. `ekfac_desvio` mostra o estrago no dente de serra — pico de
**69,6** dentro da primeira janela contra 0,2–0,4 com janela de 10 —, e o pico escala com o
comprimento da janela, o que quer dizer que ele mede **base velha**, não violação de
Kronecker. `kl_fator` foi junto: 46,2 contra 18,7–20,0 das execuções de base fresca. Ver
`ACEKTRConfig.inv_every`.

Sobreviveu uma coisa daquela execução: `kl_cal_debias` fez o que prometia. A entropia em
1 M de passos foi 0,196, a mais alta desta família (ACKTR: 0,066 · 0,143 · 0,084; ACEKTR de
01/09: 0,085).

O par limpo — base fresca, `T = 16`, escalas acumuladas — ainda não foi rodado. É o que a
próxima execução deste notebook produz.

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
    #: Média móvel exponencial de `s*`. **Só é usada quando `escalas_acumuladas=False`**;
    #: continua aqui porque `1.0` é o controle que desliga a medição e faz o EK-FAC virar
    #: **exatamente** o K-FAC, que é o que `tests/test_ekfac.py` usa para provar que a
    #: única diferença entre os dois agentes é a correção de autovalores.
    ema_escalas: float = 0.8

    #: **O conserto certo do problema que a `docs/EKFAC.md` §3.2 tinha identificado.**
    #:
    #: A queixa original era real: com uma janela de 10 e `ema_escalas = 0,5`, `s*` mal
    #: saía do palpite do K-FAC antes de a base ser reconstruída, e o que ele usava no meio
    #: do caminho era uma medição de ~2 lotes. A resposta que tentamos foi alongar a janela
    #: para 50 — e a medição de 02/09 mostrou que o custo disso (base velha) é muito maior
    #: que o benefício. Ver `inv_every`.
    #:
    #: A resposta certa é trocar o **estimador**, não a janela. Dentro de uma janela de 10
    #: atualizações a rede quase não muda, então não há deriva a esquecer: a média
    #: **acumulada** é o estimador de mínima variância, enquanto a exponencial joga fora
    #: metade da informação a cada passo para se proteger de uma deriva que não aconteceu.
    #: Com o palpite do K-FAC entrando como **uma** pseudo-observação,
    #:
    #:     s*_k = (λ_A⊗λ_G + Σ_{i≤k} s_i) / (1 + k)
    #:
    #: o prior vale 100% em `k = 0` (o controle bit a bit continua valendo), 50% em
    #: `k = 1`, 10% em `k = 9` — e a variância cai como `1/k` em vez de ficar parada em
    #: ~2 lotes. Mesmo frescor do K-FAC, autovalores de fato medidos.
    #:
    #: Ganha a marca `+s_acum` **sempre**, mesmo sendo o default: é um desvio da
    #: implementação de referência (que usa média móvel) e é o que separa esta execução
    #: das gravadas em 01/09 e 02/09 na identidade `(algo, variant, seed)`.
    escalas_acumuladas: bool = True

    #: **A base rara era o regime errado aqui, e a medição de 02/09 é inequívoca.**
    #:
    #: O paper propõe amortizar: reconstruir a base a cada 50–500 passos e recalcular as
    #: escalas a cada passo. `docs/EKFAC.md` §3.2 registrava `inv_every = 10` como um
    #: handicap deliberado, e ele foi para 50. A execução que saiu disso fechou em 74,47
    #: de score com **0,4% de tabuleiros cheios** — média maior que a da execução anterior
    #: (71,07) e taxa de vitória 44× menor (17,6%).
    #:
    #: `ekfac_desvio` diz por quê, e o dente de serra é literal:
    #:
    #: =============  =======  =======  =======  =======  =======  =======
    #: atualização    1        15       29       43       **51**   57
    #: =============  =======  =======  =======  =======  =======  =======
    #: `inv_every=50` 0,06     35,3     59,7     **69,6**  (base)  0,29
    #: `inv_every=10` 0,36     0,37     0,37     0,23      —       0,17
    #: =============  =======  =======  =======  =======  =======  =======
    #:
    #: Duas ordens de grandeza, e elas **escalam com o comprimento da janela**. Se o
    #: número medisse violação de Kronecker — que é o que o EK-FAC existe para corrigir —
    #: ele não dependeria de quando a base foi construída. Ele estava medindo **base
    #: velha**, e a §4 daquele documento, que dizia que as duas coisas "não se separam
    #: neste número", estava errada: bastava variar `inv_every`.
    #:
    #: E base velha no EK-FAC é pior que base velha no K-FAC, o que não é simétrico. O
    #: K-FAC com `A`, `G` velhos ainda é um pré-condicionador PSD coerente — só descreve
    #: uma curvatura de 10 passos atrás. O EK-FAC mistura `s*` medido **agora** com eixos
    #: de 50 passos atrás: nesses eixos `s*` pode ser minúsculo onde a curvatura real é
    #: grande, e aí divide-se por quase nada numa direção de curvatura alta. `kl_fator`
    #: mediu isso: **46,2** (p90 58,1) contra 18,7–20,0 das execuções com base fresca.
    #:
    #: A premissa do paper — o modelo muda pouco entre reconstruções — vale em aprendizado
    #: supervisionado com milhares de passos. Aqui o treino inteiro tem **610**
    #: atualizações e cada uma anda `kl_max` de KL: 50 atualizações são 8% da execução e
    #: uma política inteiramente diferente. A amortização não é de graça neste regime, e o
    #: conserto do §3.2 não é janela maior, é **estimador melhor** — ver
    #: `escalas_acumuladas`.
    inv_every: int = 10

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
                     ema_escalas=getattr(c, "ema_escalas", 0.8),
                     escalas_acumuladas=getattr(c, "escalas_acumuladas", True))

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
        if getattr(cfg, "ema_escalas", 0.8) >= 1.0:
            marcas.append("sem_correcao")
        elif getattr(cfg, "escalas_acumuladas", True):
            # sempre, mesmo sendo o default — ver `ACEKTRConfig.escalas_acumuladas`
            marcas.append("s_acum")
        return "+".join(marcas)

    def update(self, lote):
        saida = super().update(lote)
        #: O quanto `s*` já se afastou do palpite do K-FAC. É o número que distingue
        #: "o EK-FAC não ajudou aqui" de "o EK-FAC não está fazendo nada" — dois
        #: resultados idênticos na curva e opostos na leitura.
        saida["ekfac_desvio"] = self.kfac.desvio_de_kronecker()
        return saida
