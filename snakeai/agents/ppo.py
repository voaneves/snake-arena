"""PPO — a implementação de referência do benchmark.

Enxuta, mas com os detalhes que decidem se um PPO aprende ou vira ruído (a lista do
*"37 Implementation Details of PPO"*):

* **GAE(λ)** com bootstrap correto no truncamento por fome — que é diferente de morte;
* **clipping** da razão **e** do valor;
* normalização de vantagem **por minibatch**;
* **early stop por KL aproximado**, que impede o colapso quando o LR está alto demais;
* **entropia com decaimento** — explora cedo, fica determinística no fim;
* **máscara de ação aplicada aos logits no rollout _e_ no update.** Este é o detalhe que
  mais silenciosamente destrói um PPO com máscara: se o update não reaplica a máscara, o
  `log_prob` calculado lá não bate com o que gerou a ação, a razão vira lixo e o algoritmo
  otimiza uma coisa que não existe.

Sobre o truncamento por fome
----------------------------
Morrer e ficar sem comida são coisas diferentes. Morte é terminação: o retorno acabou, e o
valor do estado seguinte é zero. Fome é **truncamento**: o episódio continuaria, e cortar
ali sem fazer bootstrap ensina o agente que sobreviver muito tempo é ruim. O `VecSnake`
devolve a observação terminal dos truncados justamente para isso, e o `collect` soma
`γ · V(s_final)` à recompensa daquele passo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf

from ..env.vec_snake import N_ACTIONS, N_CHANNELS, VecSnake
from ..eval import MASK_NEG
from ..nets import build_actor_critic
from ..otimizadores import cria_otimizador
from .base import AgentBase, BaseConfig

__all__ = ["PPOConfig", "PPO", "compute_gae", "variancia_explicada"]


@dataclass
class PPOConfig(BaseConfig):
    num_envs: int = 512
    #: 32, e não 96, desde a ablação de orçamento: com o mesmo orçamento de ambiente, o
    #: rollout curto multiplica por ~16 as atualizações de gradiente. Ver
    #: `docs/ORCAMENTO_DE_GRADIENTE.md` e `PPOConfig.esparso`.
    rollout: int = 32

    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    vf_clip: float = 0.2
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.002
    max_grad_norm: float = 0.5
    lr_start: float = 3e-4
    lr_end: float = 5e-5
    #: O eixo que substitui o K-FAC. Ver `snakeai/otimizadores.py`.
    optimizer: str = "adam"
    epochs: int = 4
    minibatches: int = 32
    target_kl: float = 0.03

    #: Shaping potencial, com coeficiente que decai a zero em `shaping_frac` do treino.
    #: Decair a zero é o que garante que a política ótima final seja a do problema real.
    shaping_start: float = 0.5
    shaping_frac: float = 0.25

    #: Sexto canal com o relógio da fome. **Fora do contrato** — ver `VecSnake`. Ligar isto
    #: sem marcar `comparable=False` levanta erro, porque a entrada da rede muda e a curva
    #: deixa de ser comparável com qualquer outra do repositório.
    canal_fome: bool = False

    @classmethod
    def esparso(cls, **kw):
        """O orçamento de gradiente **anterior** — a configuração que a ablação aposentou.

        Existe para reproduzir o braço de controle, não para uso normal. Os 5 M passos do
        contrato são de *ambiente*; quantas atualizações de gradiente se tira deles é
        escolha livre, e a escolha antiga gastava pouquíssimo:

        =========================  ==============  ==============
        \                          `esparso()`     padrão
        =========================  ==============  ==============
        `rollout`                  96              32
        amostras por iteração      49.152          16.384
        iterações em 5 M passos    ~102            ~305
        `epochs` × `minibatches`   3 × 8           4 × 32
        tamanho do minilote        6.144           512
        **atualizações no total**  **~2.400**      **~38.300**
        =========================  ==============  ==============

        Medido em três sementes: 62,19 contra 80,90 de score, 4,4% contra 60,1% de
        tabuleiro cheio, e desvio entre sementes de 9,79 contra 1,80 — o orçamento não só
        levantou a média como **colapsou a dispersão** por um fator de 5,4. Ver
        `docs/ORCAMENTO_DE_GRADIENTE.md`.

        A variante ganha o sufixo `_esparso`: as duas configurações competem no mesmo
        contrato, mas não são a mesma coisa, e identidade `(algo, variant, seed)` repetida
        vira uma curva só na arena.
        """
        kw.setdefault("sufixo_variante", "esparso")
        return cls(rollout=96, epochs=3, minibatches=8, **kw)

    def __post_init__(self):
        super().__post_init__()
        if self.canal_fome and self.comparable:
            raise ValueError(
                "canal_fome=True muda a observação de 5 para 6 canais e portanto a "
                "entrada da rede. Marque comparable=False e escreva o caveat.")

    @property
    def batch_size(self):
        return self.num_envs * self.rollout


# ------------------------------------------------------------------------- GAE
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """GAE(λ) padrão, em NumPy.

    O bootstrap de truncamento já foi somado à recompensa em `collect`, então aqui todo
    `done` pode ser tratado como terminal — sem isso o valor do episódio *seguinte*
    vazaria para o anterior.
    """
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    ultimo = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        v_prox = last_value if t == T - 1 else values[t + 1]
        continua = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_prox * continua - values[t]
        ultimo = delta + gamma * lam * continua * ultimo
        adv[t] = ultimo
    return adv, adv + values


def variancia_explicada(valor, retorno):
    """`1 − Var(retorno − valor) / Var(retorno)` — o crítico explica quanto do retorno?

    1 é um crítico perfeito, 0 é um crítico que não vale mais que prever a média, e
    negativo é um crítico que atrapalha. É a métrica que diz se a vantagem do GAE está
    sendo calculada sobre uma baseline útil ou sobre ruído — e é a evidência que decide se
    o `vf_clip` em unidades absolutas está travando o crítico, porque com o valor preso a
    ±`vf_clip` por iteração ele nunca alcança a escala do retorno.
    Ver `docs/REVISAO_ALGORITMOS.md` §2.2.
    """
    retorno = np.asarray(retorno, dtype=np.float64)
    var = retorno.var()
    if var < 1e-12:                      # retorno constante: a razão não significa nada
        return float("nan")
    return float(1.0 - (retorno - np.asarray(valor, dtype=np.float64)).var() / var)


# ------------------------------------------------------------------ forward TF
@tf.function(reduce_retracing=True)
def policy_forward(model, obs, mask):
    logits, valor = model(obs, training=False)
    logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
    return logits, tf.squeeze(valor, -1)


@tf.function(reduce_retracing=True)
def sample_actions(model, obs, mask):
    logits, valor = policy_forward(model, obs, mask)
    acoes = tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0]
    logp_all = tf.nn.log_softmax(logits)
    logp = tf.gather(logp_all, acoes, batch_dims=1)
    return acoes, logp, valor


def make_optimizer(cfg, model):
    """Cria o Adam e **constrói os slots na hora**.

    Sem o `build()` explícito, o Adam só cria os momentos na primeira chamada de
    `apply_gradients` — que acontece dentro de uma `tf.function` já traçada, e aí estoura
    *"tf.function only supports singleton tf.Variables created on the first call"*.
    Na prática isso quebrava o segundo `PPO(...)` da sessão: retomar de um checkpoint, ou
    simplesmente rodar a célula de treino duas vezes no Colab.
    """
    opt = cria_otimizador(getattr(cfg, "optimizer", "adam"), cfg.lr_start,
                          clipnorm=cfg.max_grad_norm)
    opt.build(model.trainable_variables)
    return opt


class PPO(AgentBase):
    algo = "ppo"

    def __init__(self, cfg: PPOConfig = None, model=None, variant=None):
        cfg = cfg or PPOConfig()
        super().__init__(cfg, variant=variant or cfg.net)
        keras.utils.set_random_seed(cfg.seed)
        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed),
                            canal_fome=getattr(cfg, "canal_fome", False))
        # A rede é construída a partir do **ambiente**, não de uma constante: se as duas
        # fontes discordarem, o erro aparece só na primeira multiplicação de matriz, com
        # uma mensagem sobre formas que não diz nada sobre canal de fome.
        self.model = model or build_actor_critic(cfg.board_size, cfg.net,
                                                 canais=self.env.n_channels)
        self.optimizer = make_optimizer(cfg, self.model)
        self.obs, self.mask = self.env.reset()

    def on_model_reloaded(self):
        self.optimizer = make_optimizer(self.cfg, self.model)

    # ------------------------------------------------------------ agendamentos
    def lr(self):
        return self.linear(self.cfg.lr_start, self.cfg.lr_end)

    def ent_coef(self):
        return self.linear(self.cfg.ent_coef_start, self.cfg.ent_coef_end)

    def shaping(self):
        f = self.frac()
        return max(0.0, self.cfg.shaping_start * (1.0 - f / self.cfg.shaping_frac))

    # ----------------------------------------------------------------- rollout
    def collect(self):
        cfg = self.cfg
        T, N = cfg.rollout, cfg.num_envs
        # do ambiente, não da constante: com `canal_fome` são 6, e um buffer de 5 falharia
        # só aqui, com uma mensagem sobre formas que não menciona o canal de fome
        b, c = cfg.board_size, self.env.n_channels

        obs_buf = np.empty((T, N, b, b, c), dtype=np.float32)
        mask_buf = np.empty((T, N, N_ACTIONS), dtype=bool)
        act_buf = np.empty((T, N), dtype=np.int32)
        logp_buf = np.empty((T, N), dtype=np.float32)
        val_buf = np.empty((T, N), dtype=np.float32)
        rew_buf = np.empty((T, N), dtype=np.float32)
        done_buf = np.empty((T, N), dtype=np.float32)

        shaping = self.shaping()
        scores, passos_ep, vitorias = [], [], 0

        for t in range(T):
            obs_buf[t] = self.obs
            mask_buf[t] = self.mask
            a, lp, v = sample_actions(self.model,
                                      tf.convert_to_tensor(self.obs),
                                      tf.convert_to_tensor(self.mask))
            a = a.numpy()
            act_buf[t], logp_buf[t], val_buf[t] = a, lp.numpy(), v.numpy()

            self.obs, self.mask, r, d, info = self.env.step(a, shaping, cfg.gamma)
            self.registra_fim(info)
            rew_buf[t] = r
            done_buf[t] = d.astype(np.float32)

            if info["trunc_idx"].size:       # fome é truncamento, não terminação
                _, vf = policy_forward(self.model,
                                       tf.convert_to_tensor(info["final_obs"]),
                                       tf.convert_to_tensor(info["final_mask"]))
                rew_buf[t] = self.bootstrap_truncados(info, rew_buf[t], vf.numpy(),
                                                      cfg.gamma)

            scores.extend(info["scores"].tolist())
            passos_ep.extend(info["lengths"].tolist())
            vitorias += info["wins"]

        _, ultimo_v = policy_forward(self.model,
                                     tf.convert_to_tensor(self.obs),
                                     tf.convert_to_tensor(self.mask))
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, ultimo_v.numpy(),
                               cfg.gamma, cfg.gae_lambda)

        self.global_step += T * N
        self.episodes += len(scores)

        def achata(x, forma):
            return x.reshape((T * N,) + forma)

        lote = {
            "obs": achata(obs_buf, (b, b, c)),
            "mask": achata(mask_buf, (N_ACTIONS,)),
            "act": achata(act_buf, ()),
            "logp": achata(logp_buf, ()),
            "adv": achata(adv, ()),
            "ret": achata(ret, ()),
            "val": achata(val_buf, ()),
        }
        stats = {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "train_score_p95": float(np.percentile(scores, 95)) if scores else None,
            "train_ep_steps": float(np.mean(passos_ep)) if passos_ep else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "shaping": shaping,
        }
        return lote, stats

    # ------------------------------------------------------------------ update
    @staticmethod
    @tf.function(reduce_retracing=True)
    def _train_step(model, optimizer, obs, mask, act, old_logp, adv, ret, old_val,
                    clip_eps, vf_coef, vf_clip, ent_coef):
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
        with tf.GradientTape() as tape:
            logits, valor = model(obs, training=True)
            valor = tf.squeeze(valor, -1)
            # a máscara TEM que ser reaplicada aqui: sem isso o log_prob do update não
            # bate com o do rollout e a razão do PPO vira ruído
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp_all = tf.nn.log_softmax(logits)
            logp = tf.gather(logp_all, act, batch_dims=1)

            razao = tf.exp(logp - old_logp)
            pg1 = -adv * razao
            pg2 = -adv * tf.clip_by_value(razao, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = tf.reduce_mean(tf.maximum(pg1, pg2))

            v_clip = old_val + tf.clip_by_value(valor - old_val, -vf_clip, vf_clip)
            v_loss = 0.5 * tf.reduce_mean(
                tf.maximum(tf.square(valor - ret), tf.square(v_clip - ret))
            )

            probs = tf.exp(logp_all)
            seguro = tf.where(mask, logp_all, tf.zeros_like(logp_all))
            entropia = -tf.reduce_mean(tf.reduce_sum(probs * seguro, axis=-1))

            perda = pg_loss + vf_coef * v_loss - ent_coef * entropia

        grads = tape.gradient(perda, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        log_razao = logp - old_logp
        # estimador k3 do KL: não-negativo e de baixa variância, ao contrário de -log_ratio
        kl = tf.reduce_mean(tf.exp(log_razao) - 1.0 - log_razao)
        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(razao - 1.0), clip_eps), tf.float32)
        )
        return pg_loss, v_loss, entropia, kl, clipfrac

    def update(self, lote):
        cfg = self.cfg
        self.optimizer.learning_rate.assign(self.lr())
        ent = self.ent_coef()
        n = lote["act"].shape[0]
        mb = max(1, n // cfg.minibatches)
        idx = np.arange(n)
        rng = np.random.default_rng(cfg.seed + self.iteration)

        tensores = {k: tf.convert_to_tensor(v) for k, v in lote.items()}
        logs = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        parar = False
        epocas_feitas = 0
        atualizacoes = 0
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s in range(0, n, mb):
                sl = tf.convert_to_tensor(idx[s:s + mb])
                pg, vf, e, kl, cf = self._train_step(
                    self.model, self.optimizer,
                    tf.gather(tensores["obs"], sl), tf.gather(tensores["mask"], sl),
                    tf.gather(tensores["act"], sl), tf.gather(tensores["logp"], sl),
                    tf.gather(tensores["adv"], sl), tf.gather(tensores["ret"], sl),
                    tf.gather(tensores["val"], sl),
                    cfg.clip_eps, cfg.vf_coef, cfg.vf_clip, ent,
                )
                logs["pg"].append(float(pg)); logs["vf"].append(float(vf))
                logs["ent"].append(float(e)); logs["kl"].append(float(kl))
                logs["clipfrac"].append(float(cf))
                atualizacoes += 1
                if float(kl) > cfg.target_kl * 1.5:
                    parar = True
                    break
            epocas_feitas += 1
            if parar:
                break
        saida = {k: float(np.mean(v)) for k, v in logs.items()}
        saida["epochs_done"] = epocas_feitas
        saida["atualizacoes"] = int(atualizacoes)
        saida["ev"] = variancia_explicada(lote["val"], lote["ret"])
        saida["lr"] = float(self.lr())
        saida["ent_coef"] = ent
        return saida

    # ------------------------------------------------------------------ passo
    def iterate(self):
        lote, stats = self.collect()
        stats.update(self.update(lote))
        return stats
