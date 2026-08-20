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

    optimizer: str = "sgd"


class ACKTR(A2C):
    """A2C + K-FAC. Ver o docstring do módulo para o porquê da herança direta."""

    algo = "acktr"

    def __init__(self, cfg: ACKTRConfig = None, model=None, variant=None):
        super().__init__(cfg or ACKTRConfig(), model=model, variant=variant)
        c = self.cfg
        self.kfac = KFac(self.model, damping=c.damping, ema=c.kfac_ema,
                         inv_every=c.inv_every)
        # `on_model_reloaded` recria o otimizador; o K-FAC tem que acompanhar, senão os
        # índices de `trainable_variables` apontam para o modelo antigo.
        self._ultimo = {}
        #: Fator sistemático entre a KL pedida e a entregue. Começa em 1 — ou seja, a
        #: primeira atualização é idêntica à da versão não calibrada, e a correção só
        #: aparece conforme a medição chega.
        self._fator_kl = 1.0
        if variant is None:
            self.variant = self._com_sufixo(self._variante_da_regiao(c),
                                            getattr(c, "sufixo_variante", ""))

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
        t_kfac = time.perf_counter() - t0

        self.optimizer.learning_rate.assign(float(eta))
        self.optimizer.apply_gradients(zip(naturais, self.model.trainable_variables))

        kl = self._kl_medida(lote, logp_velho)

        if cfg.kl_calibrado:
            # `c` é medido contra o que foi **pedido** nesta atualização, não contra
            # `kl_max`: pedir `kl_max/c` e depois comparar com `kl_max` realimentaria a
            # própria correção e a faria divergir.
            c = kl / max(alvo_efetivo, 1e-12)
            d = cfg.kl_cal_ema
            self._fator_kl = float(np.clip(d * self._fator_kl + (1 - d) * c,
                                           cfg.kl_cal_min, cfg.kl_cal_max))

        return {
            "pg": float(pg), "vf": float(vl), "ent": float(ent),
            "lr": float(eta), "lr_teto": float(self.lr()), "ent_coef": ent_coef,
            "kl": kl, "kl_alvo": cfg.kl_max, "kl_alvo_efetivo": float(alvo_efetivo),
            "kl_fator": self._fator_kl, "epochs_done": 1,
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
        self.kfac = KFac(self.model, damping=self.cfg.damping, ema=self.cfg.kfac_ema,
                         inv_every=self.cfg.inv_every)

    def resumo_kfac(self):
        return self.kfac.resumo()
