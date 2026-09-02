"""ACKTR — A2C com gradiente natural via K-FAC e região de confiança.

*Actor-Critic using Kronecker-factored Trust Region* (Wu et al., 2017). É o A2C com uma
única troca: onde o A2C anda na direção `∇`, o ACKTR anda em `F⁻¹∇`, com o tamanho do passo
escolhido para que a divergência KL entre a política velha e a nova não passe de um alvo.

Por que ele fecha uma dívida deste repositório
----------------------------------------------
Quatro notebooks do `colab-rl` tentaram K-FAC — `snakeai_dqn_kfac_cnn3`,
`snakeai_dqn_kfac_kl_divergence_cnn3`, `kfac_optimizer_test`, `new_kfac`. Nenhum roda hoje:
dependiam de `tensorflow.contrib.kfac`, que sumiu no TF2. A pergunta por trás deles — *vale
a pena aproximar a curvatura?* — ficou sem resposta por sete anos.

Aqui ela tem resposta medida, e de graça, por causa de uma escolha de projeto anterior: o
`A2C` já existe e já é o controle experimental do PPO. **ACKTR é o A2C com K-FAC ligado, e
nada mais.** Herda `collect`, herda o GAE, herda o agendamento de entropia, herda o
bootstrap de truncamento. A diferença entre as duas curvas na arena é atribuível ao
gradiente natural, e a mais nada.

Dois detalhes que decidem se funciona
-------------------------------------
**A KL escolhe o passo, não o learning rate.** Com `Δ = F⁻¹∇`, a KL induzida por um passo
`ηΔ` vale `½η²·Δᵀ∇`. Igualando ao alvo sai `η = √(2·kl_max / Δᵀ∇)`. Isso é o que permite ao
ACKTR usar passos que derrubariam um A2C: quando a curvatura é baixa ele anda muito, quando
é alta ele encolhe sozinho. O `lr` vira apenas um **teto**.

**A curvatura vem de uma perda separada.** As estatísticas de `G` saem de `log π(a')` com
`a'` amostrada da política — não da perda de RL. Ver `snakeai/kfac.py`, seção "Fisher de
verdade".

O que a primeira execução longa mostrou sobre a região de confiança
-------------------------------------------------------------------
Numa execução de 5 M passos (`resnet_small`, semente 0), a KL **medida depois do passo**
ficou sistematicamente acima do alvo — e o registro está aqui porque a primeira leitura que
fizemos destes números estava errada em duas frentes ao mesmo tempo.

============  ==============  ========  ==========
quinto        KL mediana      × alvo    entropia
============  ==============  ========  ==========
1             0,0237          11,8      0,158
2             0,0248          12,4      0,069
3             0,0150           7,5      0,053
4             0,0105           5,2      0,041
5             0,0088           4,4      0,041
============  ==============  ========  ==========

O estouro é **maior no começo e diminui ao longo do treino** — o contrário do que se lê
olhando as últimas linhas do log, que são pontos isolados de 0,03–0,06 e não a mediana. E a
correlação entre `log(KL)` e entropia é fraca (−0,26), então "a política ficou determinística
demais" **não** explica: o pior estouro acontece justamente quando a entropia é a mais alta
da execução.

A explicação que sobra é a própria aproximação. `Δᵀ∇ = ΔᵀF̃Δ`, com `F̃` a Fisher *aproximada*
— bloco-diagonal por camada, e cada bloco um produto de Kronecker. A KL medida é a da
política de verdade. Onde `F̃` subestima a curvatura real, `Δ` fica grande demais naquelas
direções e a KL prevista sai baixa. Que o erro encolha conforme a média móvel dos fatores
amadurece é consistente com isso. (Não é o amortecimento: com `Δ = (F̃ + λI)⁻¹∇`, tem-se
`Δᵀ∇ = ΔᵀF̃Δ + λ‖Δ‖²`, que **super**estima a forma quadrática e portanto *encolhe* o passo.)

Duas consequências práticas, ambas medidas:

* Em **100% das atualizações** o passo veio da fórmula da KL, nunca do teto do `lr`. Os
  `lr_start`/`lr_end` do ACKTR não limitaram nada nesta execução — quem governa é `kl_max`.
* `kl_max = 0,002` entrega, na prática, KL ≈ 0,01. O parâmetro é um alvo *aproximado* com um
  fator de escala que depende da qualidade de `F̃`. Apertá-lo encolhe todo passo por `√k`.

`stats["kl"]` existe exatamente para que isso seja visível em vez de suposto — e a lição de
método é que a mediana por fase diz uma coisa que as últimas linhas do log dizem ao contrário.

Custo
-----
Uma retropropagação extra por atualização (a perda de Fisher) e as fatorações de Cholesky
a cada `inv_every` passos. Como a atualização acontece uma vez por rollout — 512 × 16 =
8.192 passos de ambiente — o custo se dilui. `stats["kfac_ms"]` mede quanto sobrou.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ..eval import MASK_NEG
from ..kfac import (KFac, captura_kfac, perda_fisher_categorica,
                    perda_fisher_gaussiana)
from .a2c import A2C, A2CConfig

__all__ = ["ACKTRConfig", "ACKTR"]


@dataclass
class ACKTRConfig(A2CConfig):
    #: **Redeclarado, e não herdado.** O `A2CConfig` fixa 5 — o `t_max` canônico do A3C —
    #: por um argumento que não sobrevive à região de confiança: sem clipping, o A2C anda
    #: uma distância fixa e precisa de rollouts curtos para não usar dados velhos. Aqui o
    #: tamanho do passo é a KL, e o rollout decide **outra coisa**: onde mora o crédito.
    #:
    #: Com `γλ = 0,995 × 0,95 = 0,945`, a fração do peso do GAE que sobra no bootstrap
    #: `V(s_{t+T})` é `0,945^T` — **76% com T = 5, 40% com T = 16**. Enquanto o shaping
    #: está ligado a recompensa é densa e isso não importa; depois que ele decai a zero
    #: (`shaping_frac = 0,25`, ou 1,25 M dos 5 M) a única recompensa é comida e morte, e
    #: com a cobra longa há dezenas de passos entre uma comida e outra.
    #:
    #: A medição que fixou este valor, interpolada na mesma grade de passos de ambiente
    #: (`train_score_mean`):
    #:
    #: ===================  ======  ======  ======  ======  ======
    #: execução             1,0 M   1,5 M   2,0 M   3,0 M   5,0 M
    #: ===================  ======  ======  ======  ======  ======
    #: 3 sementes, T = 16   26–29   31–37   40–64   67–72   73–81
    #: ACEKTR, T = 5        29,3    36,8    44,1    55,5    63,5
    #: ===================  ======  ======  ======  ======  ======
    #:
    #: Até 1,5 M o T = 5 está **no topo** da faixa; de 2 M em diante ele sai por baixo e
    #: não volta. O ponto de separação é o fim do shaping, que é o que a conta acima
    #: prevê. E não é falta de passo: o T = 5 acumulou `Σ√KL` de **202** contra 57–73 das
    #: três sementes de T = 16 — ele andou 3,6× mais e chegou mais perto do chão.
    #:
    #: As três execuções de `acktr/resnet_small` gravadas rodaram com 16. O 5 entrou junto
    #: com o `A2CConfig` em 21/08, um dia **depois** delas, e ninguém reexecutou — de modo
    #: que o default de hoje não reproduz nenhum resultado do repositório. Isto é
    #: restauração, não escolha nova.
    rollout: int = 16

    #: Teto do passo. No ACKTR o tamanho normalmente é decidido pela KL; o `lr` só impede
    #: que um lote de curvatura quase nula peça um passo absurdo.
    lr_start: float = 0.5
    lr_end: float = 0.1

    #: Alvo de KL por atualização. Wu et al. usam 0,001–0,002 no Atari; aqui o alvo
    #: **entregue** é o que importa, porque o pedido passa pela Fisher aproximada antes de
    #: virar passo — e é isso que `kl_calibrado` fecha.
    #:
    #: Três sementes com este valor, e uma leitura que **não** funcionou:
    #:
    #: =======  =============  =======  ========
    #: semente  KL entregue    final    cheio
    #: =======  =============  =======  ========
    #: 0        0,0068         89,78    89,7%
    #: 1        0,0097         70,67    43,7%
    #: 2        0,0143         78,13    60,7%
    #: =======  =============  =======  ========
    #:
    #: Média 79,52, desvio **9,63**. Cheguei a escrever aqui que existia um ótimo interior
    #: perto de 0,0068; a semente 2 desmente — ela entrega o **dobro** de KL e joga melhor
    #: que a semente 1. A KL entregue não explica a dispersão, e o que sobra é semente.
    #:
    #: Para comparação, o PPO no orçamento padrão faz 80,90 com desvio 1,80: mesma média,
    #: **5,3× menos dispersão**. É o resultado honesto sobre o ACKTR neste ambiente — não
    #: que ele seja pior, e sim que ele é imprevisível. Ver
    #: `docs/ORCAMENTO_DE_GRADIENTE.md`.
    kl_max: float = 1.5e-2

    #: Amortecimento de Tikhonov. Alto demais e o ACKTR vira A2C com passo esquisito;
    #: baixo demais e a inversa amplifica direções que o lote mal estimou.
    damping: float = 1e-2

    #: Média móvel dos fatores entre atualizações. Absorve o ruído de amostragem da Fisher.
    kfac_ema: float = 0.95

    #: A cada quantas atualizações refatorar. As Cholesky custam O(d³) nos fatores, que
    #: são pequenos — mas `A` da primeira convolução ainda é 288×288.
    inv_every: int = 10

    #: Peso da parte gaussiana na perda de Fisher. Wu et al. usam 1,0 quando ator e crítico
    #: compartilham tronco, que é o caso aqui.
    fisher_vf_coef: float = 1.0

    #: **Calibra a região de confiança pela KL que de fato aconteceu.**
    #:
    #: Sem isto, `kl_max` é um alvo nominal: a execução de 5 M passos pediu 0,002 e
    #: entregou ~0,01, porque `Δᵀ∇` usa a Fisher *aproximada* e a KL medida é a da política
    #: de verdade. Ligado, o agente estima o fator sistemático `c = KL_medida / alvo_pedido`
    #: por média móvel e pede `kl_max / c` — de modo que a KL **entregue** convirja para
    #: `kl_max`.
    #:
    #: **Ligado por padrão desde a medição.** Era eixo de ablação e virou o comportamento
    #: oficial: sem calibrar, a mesma configuração e a mesma semente entregaram 83,91 num
    #: Colab de agosto e 64,53 num Kaggle depois — o fator não controlado entre a Fisher
    #: aproximada e a KL real muda com o hardware, e o resultado deixa de ser reprodutível.
    #: Desligar isto é a ablação, e a variante ganha `+kl_nominal` para dizer isso.
    kl_calibrado: bool = True

    #: Média móvel do fator. Alta porque `c` é ruidoso lote a lote.
    kl_cal_ema: float = 0.98

    #: Limites do fator, para um lote patológico não travar a calibração num extremo.
    kl_cal_min: float = 0.05
    kl_cal_max: float = 200.0

    #: **Corrige o atraso de partida da calibração.** `_fator_kl` é uma média móvel com
    #: `kl_cal_ema = 0,98`, ou seja constante de tempo de ~50 atualizações — e o orçamento
    #: inteiro tem **610**. Partindo de 1,0, ela gasta ~8% do treino subindo até o fator
    #: verdadeiro (15 a 25 nas execuções medidas), e nesse intervalo o alvo efetivo é até
    #: 20× maior que o pedido, bem na hora em que a política ainda é aleatória. As três
    #: sementes registram esse transitório: `kl_fator` mediano no primeiro quinto foi
    #: 15,5 · 13,5 · 5,4, contra 18,5 · 15,7 · 9,1 no último.
    #:
    #: Ligado, a média passa a ser **debiasada** — mantém-se `s` e o peso `w` acumulado e
    #: usa-se `s/w`, como o `1 − β^t` do Adam — de modo que a segunda atualização já usa o
    #: fator medido, sem abrir mão da suavização depois. Com `kl_fator_inicial` como
    #: prior de peso pequeno, a **primeira** também sai razoável.
    #:
    #: Default `False` no ACKTR de propósito: as três execuções gravadas rodaram sem isto,
    #: e mudar o default silenciosamente faria `08_acktr` deixar de reproduzi-las. O
    #: `ACEKTRConfig` liga.
    kl_cal_debias: bool = False

    #: Prior do fator, usado só quando `kl_cal_debias`. `1,0` é "não sei nada", que é o
    #: comportamento histórico. As medições do `docs/diag_acktr_kl.json` (7,4 · 7,0 · 1,2)
    #: e o regime das execuções longas (15 a 25) sustentam um prior alto — e o erro é
    #: assimétrico: começar cauteloso demais custa alguns passos curtos, começar ousado
    #: demais colapsa a entropia e não tem volta.
    kl_fator_inicial: float = 1.0

    #: Peso do prior, na mesma unidade do peso acumulado da média (que satura em 1). Com
    #: 0,05 o prior vale ~70% na primeira atualização, 18% na décima e 3% na quinquagésima.
    kl_cal_peso0: float = 0.05

    optimizer: str = "sgd"

    # ------------------------------------------------------------------------------
    # §2.36 — os dois suspeitos do estouro da KL que não são a Fisher.
    # ------------------------------------------------------------------------------

    #: Momento do SGD que aplica a direção natural. `escala_kl` devolve `η` tal que **um**
    #: passo `ηΔ` induz `kl_max`; com momento, o deslocamento em regime é até `ηΔ/(1−μ)`,
    #: e a KL vai com o **quadrado** disso. Medir `momento = 0` isola o efeito.
    momento: float = 0.9
    #: `lr = η·(1−μ)`, que é o que o `baselines` faz (`MomentumOptimizer(lr*(1-momentum),
    #: momentum)`). É o conserto **certo**, porque preserva a redução de variância do
    #: momento em vez de jogá-la fora junto com o estouro.
    descontar_momento: bool = False

    @property
    def opt_extra(self):
        """Repassado a `cria_otimizador`. O `nesterov` acompanha o momento: com `μ = 0`
        ele não tem o que antecipar, e deixá-lo ligado só confunde a leitura do braço."""
        return {"momentum": self.momento, "nesterov": self.momento > 0}


class ACKTR(A2C):
    """A2C + K-FAC. Ver o docstring do módulo para o porquê da herança direta."""

    algo = "acktr"

    def __init__(self, cfg: ACKTRConfig = None, model=None, variant=None):
        super().__init__(cfg or ACKTRConfig(), model=model, variant=variant)
        c = self.cfg
        self.kfac = self._cria_precondicionador()
        # `on_model_reloaded` recria o otimizador; o K-FAC tem que acompanhar, senão os
        # índices de `trainable_variables` apontam para o modelo antigo.
        self._ultimo = {}
        #: Fator sistemático entre a KL pedida e a entregue. Começa em 1 — ou seja, a
        #: primeira atualização é idêntica à da versão não calibrada, e a correção só
        #: aparece conforme a medição chega.
        self._fator_kl = float(getattr(c, "kl_fator_inicial", 1.0))
        #: Numerador e peso da média debiasada. O prior entra com peso `kl_cal_peso0`, e
        #: como o peso de uma EMA satura em 1, ele se dilui sozinho conforme as medições
        #: chegam — sem `if passo < N` nenhum.
        self._cal_peso = float(getattr(c, "kl_cal_peso0", 0.05))
        self._cal_soma = self._fator_kl * self._cal_peso
        if variant is None:
            self.variant = self._com_sufixo(self._variante_da_regiao(c),
                                            getattr(c, "sufixo_variante", ""))

    def _cria_precondicionador(self):
        """Qual curvatura este agente usa.

        Existe como método, e não como uma linha dentro do `__init__`, porque é o **único**
        ponto que o `ACEKTR` sobrescreve. Enquanto for só isto, a diferença entre as duas
        curvas na arena é atribuível à correção de autovalores e a mais nada — e
        `tests/test_ekfac.py` confere que continua sendo só isto.
        """
        c = self.cfg
        return KFac(self.model, damping=c.damping, ema=c.kfac_ema,
                    inv_every=c.inv_every)

    @staticmethod
    def _variante_da_regiao(cfg):
        """A variante diz em que região de confiança a execução rodou.

        O padrão — calibrado no alvo medido — não ganha marca nenhuma: é o ACKTR oficial.
        Qualquer desvio aparece no nome, porque `load_all` agrupa por
        `(algo, variant, seed)` e duas regiões de confiança diferentes com a mesma
        identidade viram uma curva só. Foi assim que o ACKTR de 12/08 e o de agora quase
        se fundiram na arena.
        """
        marcas = []
        if not cfg.kl_calibrado:
            marcas.append("kl_nominal")
        # comparado ao default da **própria** classe: o ACEKTR liga por padrão, então para
        # ele o que precisa aparecer no nome é o desligamento, e vice-versa.
        if cfg.kl_cal_debias != type(cfg).kl_cal_debias:
            marcas.append("kl_cal_debias" if cfg.kl_cal_debias else "kl_cal_v1")
        if cfg.kl_max != type(cfg).kl_max:
            marcas.append(f"kl{cfg.kl_max:g}")
        return "+".join([cfg.net] + marcas)

    # ------------------------------------------------------------------ um passo
    def _forward_e_gradientes(self, obs, mask, act, adv, ret, ent_coef, vf_coef):
        """Um forward, duas retropropagações: a da perda real e a da perda de Fisher.

        Não é `tf.function`: a captura do K-FAC precisa reexecutar `call` a cada chamada e
        o `precondiciona` roda em eager de qualquer forma. Como isto acontece **uma vez por
        rollout**, o overhead de Python se dilui em milhares de passos de ambiente — o que
        foi medido, não suposto (`tools/perfil_dispositivo.py`).
        """
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

        with captura_kfac(self.kfac.camadas) as cap:
            with tf.GradientTape(persistent=True) as tape:
                logits, valor = self.model(obs, training=True)
                valor = tf.squeeze(valor, -1)
                logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))

                logp_all = tf.nn.log_softmax(logits)
                logp = tf.gather(logp_all, act, batch_dims=1)
                pg = -tf.reduce_mean(logp * adv)
                vl = 0.5 * tf.reduce_mean(tf.square(valor - ret))

                probs = tf.exp(logp_all)
                seguro = tf.where(mask, logp_all, tf.zeros_like(logp_all))
                ent = -tf.reduce_mean(tf.reduce_sum(probs * seguro, axis=-1))

                perda = pg + vf_coef * vl - ent_coef * ent

                # A perda de Fisher **não** é a perda de RL: ela define a métrica, não o
                # objetivo. O sinal negativo é porque queremos o gradiente do
                # log-likelihood, e `perda_fisher_categorica` já devolve a média de log π.
                pf = (-perda_fisher_categorica(logits, mask)
                      + self.cfg.fisher_vf_coef * perda_fisher_gaussiana(valor[:, None]))

            grads = tape.gradient(perda, self.model.trainable_variables)
            gs = tape.gradient(pf, [z for _, _, z in cap])
        del tape

        return grads, cap, gs, pg, vl, ent, logp_all

    def update(self, lote):
        cfg = self.cfg
        ent_coef = self.ent_coef()
        t0 = time.perf_counter()

        grads, cap, gs, pg, vl, ent, logp_velho = self._forward_e_gradientes(
            tf.convert_to_tensor(lote["obs"]), tf.convert_to_tensor(lote["mask"]),
            tf.convert_to_tensor(lote["act"]), tf.convert_to_tensor(lote["adv"]),
            tf.convert_to_tensor(lote["ret"]), ent_coef, cfg.vf_coef,
        )
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.kfac.acumula(cap, gs)
        naturais = self.kfac.precondiciona(grads)
        alvo_efetivo = cfg.kl_max / self._fator_kl if cfg.kl_calibrado else cfg.kl_max
        eta = self.kfac.escala_kl(naturais, grads, alvo_efetivo, self.lr())
        if cfg.descontar_momento and cfg.momento > 0:
            # `escala_kl` dimensiona UM passo. Com momento, o deslocamento acumulado é
            # `η/(1−μ)` vezes a direção — então pedir `η·(1−μ)` devolve à atualização em
            # regime o tamanho que a fórmula da KL calculou.
            eta = eta * (1.0 - cfg.momento)
        t_kfac = time.perf_counter() - t0

        # Quanto do `clipnorm` está de fato mordendo. O Keras clipa **por variável**,
        # dentro do `apply_gradients`, e sobre a direção já pré-condicionada — então este
        # número diz se o passo que saiu é o que a fórmula da KL pediu ou o que o clip
        # deixou passar. Sem ele, `clipnorm` é um mediador silencioso: foi ele que fez
        # dois braços do §2.36 medirem outra coisa sem avisar. São ~27 normas por
        # atualização, custo desprezível.
        teto_g = cfg.max_grad_norm or 0.0
        if teto_g > 0:
            normas = tf.stack([tf.norm(n) for n in naturais if n is not None])
            frac_clipado = float(tf.reduce_mean(
                tf.cast(normas > teto_g, tf.float32)))
        else:
            frac_clipado = 0.0

        self.optimizer.learning_rate.assign(float(eta))
        self.optimizer.apply_gradients(zip(naturais, self.model.trainable_variables))

        kl = self._kl_medida(lote, logp_velho)

        if cfg.kl_calibrado:
            # `c` é medido contra o que foi **pedido** nesta atualização, não contra
            # `kl_max`: pedir `kl_max/c` e depois comparar com `kl_max` realimentaria a
            # própria correção e a faria divergir.
            c = kl / max(alvo_efetivo, 1e-12)
            d = cfg.kl_cal_ema
            if getattr(cfg, "kl_cal_debias", False):
                # média móvel debiasada: `s/w` em vez de `s`. Sem isto a média parte de
                # `_fator_kl` e leva 1/(1−d) ≈ 50 atualizações para chegar ao valor
                # medido — 8% de um orçamento de 610, gastos com o alvo efetivo até 20×
                # maior que o pedido.
                self._cal_soma = d * self._cal_soma + (1 - d) * c
                self._cal_peso = d * self._cal_peso + (1 - d)
                bruto = self._cal_soma / max(self._cal_peso, 1e-12)
            else:
                bruto = d * self._fator_kl + (1 - d) * c
            self._fator_kl = float(np.clip(bruto, cfg.kl_cal_min, cfg.kl_cal_max))

        return {
            "pg": float(pg), "vf": float(vl), "ent": float(ent),
            "lr": float(eta), "lr_teto": float(self.lr()), "ent_coef": ent_coef,
            "kl": kl, "kl_alvo": cfg.kl_max, "kl_alvo_efetivo": float(alvo_efetivo),
            "kl_fator": self._fator_kl, "frac_clipado": frac_clipado,
            "epochs_done": 1,
            "kfac_ms": t_kfac * 1e3, "fwd_ms": t_fwd * 1e3,
        }

    def _kl_medida(self, lote, logp_velho):
        """KL real depois do passo. O alvo é de segunda ordem; isto é o que aconteceu.

        Sem esta medida, `kl_max` seria um parâmetro que ninguém sabe se está sendo
        respeitado — e a aproximação quadrática se degrada exatamente quando o passo é
        grande, que é quando importa.
        """
        logits, _ = self.model(tf.convert_to_tensor(lote["obs"]), training=False)
        mask = tf.convert_to_tensor(lote["mask"])
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        novo = tf.nn.log_softmax(logits)
        p_velho = tf.exp(logp_velho)
        kl = tf.reduce_sum(tf.where(mask, p_velho * (logp_velho - novo),
                                    tf.zeros_like(novo)), axis=-1)
        return float(tf.reduce_mean(kl))

    # -------------------------------------------------------------------- relato
    def on_model_reloaded(self):
        super().on_model_reloaded()
        self.kfac = self._cria_precondicionador()

    def resumo_kfac(self):
        return self.kfac.resumo()
