"""AlphaZero — busca em árvore sobre o simulador real, com política e valor aprendidos.

A ideia do AlphaZero em uma frase: **a busca é um operador de melhoria de política**. A
rede propõe, o MCTS refina, e a rede é treinada para imitar o refinamento. Repete. Cada
ciclo torna a proposta melhor, o que torna a busca melhor, que torna o alvo melhor.

Aqui a busca usa o `VecSnake` de verdade — Snake é determinístico e de informação perfeita,
então não há nada a ganhar aprendendo um modelo do mundo (é o que o MuZero faz, e é a
próxima peça). Medido com um valor heurístico bobo (distância de Manhattan até a comida),
**8 simulações por jogada já dão score 24**, contra 0,67 da política aleatória com máscara.
A máquina de busca funciona; o que este agente faz é aprender um valor melhor que a
heurística.

Os dois alvos de treino
-----------------------
* **Política** ← distribuição de visitas do MCTS na raiz. É o "professor": a busca já
  gastou computação descobrindo qual ação é boa, e a rede aprende a chegar lá direto.
* **Valor** ← retorno de `n` passos, com bootstrap no **valor da raiz do MCTS**, não no
  valor da rede. O valor da busca é mais preciso que o da rede (foi refinado por
  simulações), então usar a rede aqui seria jogar fora justamente o ganho.

Quantas simulações a destilação precisa (medido, e é mais do que parece)
------------------------------------------------------------------------
Rodando com **12 simulações** e 3 ações, a busca joga bem — score de treino 11–12 — mas a
política pura da rede fica **abaixo do piso aleatório**, em ~0,25, mesmo depois de 100 mil
passos. O motivo não é bug: com 12 visitas divididas entre 3 ações, a distribuição de
visitas sai quase uniforme, e um alvo quase uniforme não ensina nada. A busca sabe jogar; o
professor é que não consegue explicar.

A destilação só começa a funcionar quando as visitas se concentram, o que exige um
orçamento de busca bem maior — o AlphaZero original usa centenas de simulações por jogada.
Este é o parâmetro que decide se o agente aprende, e ele custa caro: cada simulação é uma
avaliação de rede. Aqui a CPU limita a ~12; numa T4 dá para usar 64–128, que é onde o
algoritmo passa a fazer sentido.

Sobre o protocolo de avaliação
------------------------------
A curva oficial do benchmark mede a **política pura da rede**, greedy, sem busca — igual
para todos os algoritmos. O MCTS na hora da inferência é computação extra no momento de
jogar, exatamente como o filtro de segurança por flood-fill, e por isso vira **coluna
separada** da tabela em vez de entrar na curva principal. Comparar uma política com busca
contra políticas sem busca no mesmo eixo seria repetir, com roupa nova, o erro que este
repositório existe para consertar.
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
from ..search import MCTS
from .base import AgentBase, BaseConfig

__all__ = ["AlphaZeroConfig", "AlphaZero"]


@dataclass
class AlphaZeroConfig(BaseConfig):
    num_envs: int = 64
    rollout: int = 16

    num_simulations: int = 32
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25

    gamma: float = 0.997
    n_step: int = 10
    lr: float = 3e-4
    max_grad_norm: float = 5.0
    batch_size: int = 512
    epochs_por_iter: int = 1
    memory_size: int = 100_000

    #: Temperatura da amostragem na coleta. Cai a zero em `temp_frac` do treino: explora
    #: cedo, joga a sério no fim.
    temp_inicio: float = 1.0
    temp_fim: float = 0.25
    temp_frac: float = 0.5

    vf_coef: float = 1.0
    ent_coef: float = 0.0     # a exploração vem do ruído de Dirichlet, não da entropia

    #: Simulações usadas na coluna "com busca" da tabela. Zero desliga.
    sims_avaliacao: int = 32


class AlphaZero(AgentBase):
    algo = "alphazero"

    def __init__(self, cfg: AlphaZeroConfig = None, variant=None):
        cfg = cfg or AlphaZeroConfig()
        super().__init__(cfg, variant=variant or f"sims{cfg.num_simulations}")
        keras.utils.set_random_seed(cfg.seed)

        self.model = build_actor_critic(cfg.board_size, cfg.net)
        self.optimizer = keras.optimizers.Adam(cfg.lr, clipnorm=cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)

        self.mcts = MCTS(self._avaliar, board_size=cfg.board_size, gamma=cfg.gamma,
                         num_simulations=cfg.num_simulations, c_puct=cfg.c_puct,
                         dirichlet_alpha=cfg.dirichlet_alpha,
                         dirichlet_frac=cfg.dirichlet_frac,
                         starve_base=self.env.starve_base,
                         rng=np.random.default_rng(cfg.seed + 2))

        forma = (cfg.board_size, cfg.board_size, N_CHANNELS)
        self._buf_obs = np.zeros((cfg.memory_size, *forma), dtype=np.float32)
        self._buf_mask = np.ones((cfg.memory_size, N_ACTIONS), dtype=bool)
        self._buf_pi = np.zeros((cfg.memory_size, N_ACTIONS), dtype=np.float32)
        self._buf_z = np.zeros(cfg.memory_size, dtype=np.float32)
        self._pos, self._cheio = 0, 0

    def on_model_reloaded(self):
        self.optimizer = keras.optimizers.Adam(self.cfg.lr,
                                               clipnorm=self.cfg.max_grad_norm)
        self.optimizer.build(self.model.trainable_variables)
        self.mcts.avaliar = self._avaliar

    # ------------------------------------------------------------- rede -> MCTS
    @tf.function(reduce_retracing=True)
    def _frente(self, obs, mask):
        logits, valor = self.model(obs, training=False)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        return tf.nn.softmax(logits), tf.squeeze(valor, -1)

    def _avaliar(self, obs, mask):
        """A interface que o MCTS consome: `(priors, valores)`, ambos NumPy."""
        p, v = self._frente(tf.convert_to_tensor(np.asarray(obs, dtype=np.float32)),
                            tf.convert_to_tensor(np.asarray(mask)))
        return p.numpy(), v.numpy()

    def politica(self):
        """Política **pura da rede**, sem busca — é o que a curva oficial mede."""
        def fn(obs, mask):
            logits, _ = self.model(obs, training=False)
            return np.where(mask, np.asarray(logits), MASK_NEG).astype(np.float32)
        return fn

    def avaliar_com_busca(self, episodes=1000, num_simulations=None, seed=123):
        """Roda o protocolo oficial, mas escolhendo com MCTS.

        Não passa por `snakeai.eval` porque a busca precisa do **estado** do ambiente, e a
        interface de política só recebe observação e máscara. O protocolo (episódios,
        semente, greedy) é o mesmo.
        """
        cfg = self.cfg
        n = min(cfg.eval_envs, 64)
        env = VecSnake(n, cfg.board_size, rng=np.random.default_rng(seed))
        busca = MCTS(self._avaliar, board_size=cfg.board_size, gamma=cfg.gamma,
                     num_simulations=num_simulations or cfg.sims_avaliacao,
                     c_puct=cfg.c_puct, starve_base=env.starve_base,
                     rng=np.random.default_rng(seed))
        obs, mask = env.reset()
        por_env = int(np.ceil(episodes / n))
        coletados = [[] for _ in range(n)]
        faltam, vitorias = n, 0

        while faltam > 0:
            visitas, _ = busca.run(env.get_state(), mask, obs)
            a = visitas.argmax(1).astype(np.int32)
            antes = env.score.copy()
            obs, mask, r, done, info = env.step(a)
            vitorias += info["wins"]
            for i in np.nonzero(done)[0]:
                if len(coletados[i]) < por_env:
                    coletados[i].append(int(antes[i]))
                    if len(coletados[i]) == por_env:
                        faltam -= 1

        scores = np.array([s for l in coletados for s in l][:episodes])
        return {
            "episodes": int(scores.size),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_max": int(scores.max()),
            "score_p95": float(np.percentile(scores, 95)),
            "win_rate": vitorias / max(1, scores.size),
            "num_simulations": busca.num_simulations,
            "completo": True,
        }

    # ------------------------------------------------------------------ coleta
    def temperatura(self):
        f = min(1.0, self.frac() / max(self.cfg.temp_frac, 1e-9))
        return self.cfg.temp_inicio + f * (self.cfg.temp_fim - self.cfg.temp_inicio)

    def _guardar(self, obs, mask, pi, z):
        k = len(z)
        idx = (self._pos + np.arange(k)) % self.cfg.memory_size
        self._buf_obs[idx] = obs
        self._buf_mask[idx] = mask
        self._buf_pi[idx] = pi
        self._buf_z[idx] = z
        self._pos = int((self._pos + k) % self.cfg.memory_size)
        self._cheio = min(self._cheio + k, self.cfg.memory_size)

    def collect(self):
        cfg = self.cfg
        T, N = cfg.rollout, cfg.num_envs
        temp = self.temperatura()

        obs_b = np.empty((T, N, cfg.board_size, cfg.board_size, N_CHANNELS), np.float32)
        mask_b = np.empty((T, N, N_ACTIONS), bool)
        pi_b = np.empty((T, N, N_ACTIONS), np.float32)
        v_raiz = np.empty((T, N), np.float32)
        rew_b = np.empty((T, N), np.float32)
        done_b = np.empty((T, N), np.float32)

        scores, vitorias = [], 0
        for t in range(T):
            obs_b[t], mask_b[t] = self.obs, self.mask
            visitas, valores = self.mcts.run(self.env.get_state(), self.mask, self.obs,
                                             adicionar_ruido=True)
            pi = MCTS.politica_das_visitas(visitas, temp)
            pi_b[t] = pi
            v_raiz[t] = valores
            a = (pi.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)

            self.obs, self.mask, r, d, info = self.env.step(a)
            rew_b[t], done_b[t] = r, d.astype(np.float32)
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        # alvo de valor: retorno de n passos com bootstrap no VALOR DA BUSCA. Usar o
        # valor da rede aqui jogaria fora justamente o refino que a busca produziu.
        z = np.zeros((T, N), np.float32)
        for t in range(T):
            g = np.zeros(N, np.float32)
            desconto = np.ones(N, np.float32)
            vivo = np.ones(N, bool)
            k = 0
            for k in range(min(cfg.n_step, T - t)):
                g += desconto * rew_b[t + k] * vivo
                vivo &= done_b[t + k] < 0.5
                desconto *= cfg.gamma
            if t + k + 1 < T:
                g += desconto * v_raiz[t + k + 1] * vivo
            z[t] = g

        self.global_step += T * N
        self.episodes += len(scores)
        self._guardar(obs_b.reshape(T * N, *obs_b.shape[2:]),
                      mask_b.reshape(T * N, N_ACTIONS),
                      pi_b.reshape(T * N, N_ACTIONS), z.reshape(T * N))

        return {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "temperatura": temp,
            "valor_raiz": float(v_raiz.mean()),
            "memoria": self._cheio,
        }

    # ------------------------------------------------------------------ treino
    @tf.function(reduce_retracing=True)
    def _passo(self, obs, mask, pi_alvo, z, vf_coef, ent_coef):
        with tf.GradientTape() as tape:
            logits, valor = self.model(obs, training=True)
            valor = tf.squeeze(valor, -1)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp = tf.nn.log_softmax(logits)

            # entropia cruzada contra a política da busca: a rede aprende a chegar
            # direto onde o MCTS chegou gastando simulações
            perda_pi = -tf.reduce_mean(tf.reduce_sum(pi_alvo * logp, axis=-1))
            perda_v = tf.reduce_mean(tf.square(valor - z))
            entropia = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp) * logp, axis=-1))
            perda = perda_pi + vf_coef * perda_v - ent_coef * entropia

        grads = tape.gradient(perda, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return perda_pi, perda_v, entropia

    def _aprender(self):
        cfg = self.cfg
        if self._cheio < cfg.batch_size:
            return None
        perdas = []
        for _ in range(cfg.epochs_por_iter):
            idx = self.rng.integers(0, self._cheio, size=cfg.batch_size)
            pi, pv, e = self._passo(
                tf.convert_to_tensor(self._buf_obs[idx]),
                tf.convert_to_tensor(self._buf_mask[idx]),
                tf.convert_to_tensor(self._buf_pi[idx]),
                tf.convert_to_tensor(self._buf_z[idx]),
                cfg.vf_coef, cfg.ent_coef,
            )
            perdas.append((float(pi), float(pv), float(e)))
        p, v, e = (float(np.mean(x)) for x in zip(*perdas))
        return {"perda_pi": p, "perda_v": v, "entropia": e}

    def iterate(self):
        stats = self.collect()
        treino = self._aprender()
        if treino:
            stats.update(treino)
        return stats
