"""MuZero — a mesma busca, sobre um modelo aprendido.

O contraste com o AlphaZero deste repositório é a razão de ele estar aqui: **o algoritmo de
busca é literalmente o mesmo objeto**, o `MCTS`. Muda só o que a árvore percorre — a
`DinamicaReal` (o `VecSnake`) vira `DinamicaAprendida` (a rede `g`). Toda diferença de
resultado entre os dois é atribuível a isso, e a nada mais.

Vale dizer com todas as letras: **em Snake, o MuZero deveria perder para o AlphaZero.** O
simulador está disponível, é exato e é rápido; trocá-lo por uma aproximação aprendida só
pode piorar a busca. O MuZero existe para domínios onde o simulador *não* está disponível
durante o jogo — e medi-lo aqui é medir quanto custa não ter o simulador. Esse é um número
interessante, e é uma pergunta que o benchmark pode responder justamente por ter os dois
lado a lado, sob o mesmo contrato.

O desenrolar de K passos
------------------------
O treino não olha transições isoladas. Ele parte de uma posição da trajetória, aplica `h`
uma vez, e depois `g` `K` vezes seguindo as ações que foram realmente tomadas. Em cada um
dos `K+1` passos há três perdas:

* **política** ← visitas do MCTS naquele passo,
* **valor** ← retorno de n passos com bootstrap no valor da busca,
* **recompensa** ← a recompensa que o ambiente de fato deu.

A perda de recompensa é a única âncora que liga o estado oculto ao mundo. Sem ela o modelo
pode inventar qualquer dinâmica internamente consistente e a busca vira ficção.

Os consertos que vieram do AlphaZero
------------------------------------
Como o `MCTS` é **o mesmo objeto**, os três defeitos que a primeira execução de 5 M passos
do AlphaZero revelou estavam aqui também, palavra por palavra (§2.27–§2.29 da revisão):

* o PUCT dava `Q = 0` a um filho ainda não visitado. É a convenção do AlphaZero, correta
  onde o valor é uma `tanh` em `[-1, 1]`; aqui a cabeça é linear e o valor aprendido é
  positivo, então o bônus `c_puct·P·√N` só cobre a diferença onde o prior já é alto — a
  busca passa a **confirmar** a rede em vez de discordar dela, que é o oposto de ser um
  operador de melhoria de política. Conserto: `fpu` e `q_normalizado` — este último é,
  ironicamente, a normalização min-max do próprio paper do MuZero (Apêndice B);
* o alvo de valor não é normalizado e domina o tronco compartilhado. Conserto:
  `valor_symlog`, com a busca continuando a ler a escala real;
* a mesma distribuição temperada escolhia a ação **e** virava o alvo de treino. Conserto:
  `temp_alvo` e `temp_passos`.

Mais o orçamento de gradiente, o decaimento de `lr` e o bootstrap do fim da janela. Ao
contrário do AlphaZero, aqui não havia execução de controle a preservar — o MuZero nunca
rodou sob o contrato — então tudo já nasce ligado. Ver `docs/BUSCA_DEGENERADA.md`.
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
from ..nets.muzero import build_dinamica, build_predicao, build_representacao
from ..nets.resnet import PRESETS
from ..search import MCTS, DinamicaAprendida
from .base import AgentBase, BaseConfig

__all__ = ["MuZeroConfig", "MuZero"]


@dataclass
class MuZeroConfig(BaseConfig):
    net: str = "resnet_small"
    num_envs: int = 64
    rollout: int = 16

    num_simulations: int = 24
    c_puct: float = 1.5
    #: α ∝ 1/(ações legais); a heurística do paper calibra em ~10/n, que daria 3,3 para
    #: **3** ações. Com 0,5 o ruído punha mais de 90% da massa numa única ação em 15% dos
    #: lances. Ver `docs/BUSCA_DEGENERADA.md`.
    dirichlet_alpha: float = 1.0
    dirichlet_frac: float = 0.25

    # ------------------------------------------------------------------------------
    # Os três consertos do §2.27–§2.29, herdados do AlphaZero. **O `MCTS` é o mesmo
    # objeto**, então os defeitos eram os mesmos — e o MuZero nunca rodou sob o contrato,
    # então aqui eles já nascem ligados, sem execução de controle para preservar.
    # ------------------------------------------------------------------------------

    #: §2.27 — o Q de um filho ainda não visitado. `"zero"` é a convenção do AlphaZero e
    #: está certa onde o valor é uma `tanh` em `[-1, 1]`; aqui a cabeça é linear e o valor
    #: aprendido é positivo, então o bônus `c_puct·P·√N` só cobre a diferença onde o prior
    #: já é alto — a busca passa a confirmar a rede em vez de discordar dela.
    fpu: str = "pai"
    #: §2.27 — normalização min-max do Q dentro da árvore (MuZero, Apêndice B). Devolve
    #: `c_puct` à escala em que foi calibrado. Irônico que faltasse justamente aqui.
    q_normalizado: bool = True
    #: Empate exato no PUCT: `"ordem"` fica sempre com o primeiro filho do dicionário.
    desempate: str = "aleatorio"

    #: §2.28 — treinar o valor em symlog em vez da escala crua. O alvo é um retorno
    #: descontado não normalizado que cresce com o agente; a busca continua recebendo o
    #: valor na escala **real**, porque o backup soma `recompensa + γ·valor`.
    valor_symlog: bool = True

    #: §2.29 — temperatura por lance do episódio (o agendamento do paper) e alvo de
    #: política sem temperar. Ver as notas homônimas em `alphazero.py`.
    temp_passos: int = 30
    temp_alvo: float = 1.0
    #: Fecha o último passo da janela de coleta, que hoje teria alvo sem bootstrap.
    bootstrap_fim_janela: bool = True

    gamma: float = 0.997
    n_step: int = 10
    #: Passos do desenrolar no treino. É o que obriga o modelo a ser útil por mais de um
    #: passo à frente — com K=1 ele vira um crítico caro.
    unroll: int = 5

    lr: float = 3e-4
    #: Decaimento linear do `lr` até o fim do orçamento, como no PPO e no ACKTR. `0`
    #: mantém constante.
    lr_final: float = 5e-5
    max_grad_norm: float = 5.0
    batch_size: int = 256
    memory_size: int = 50_000
    #: Com 1, os 5 M passos compravam ~4.900 atualizações contra as ~38.300 do PPO. O
    #: passo de gradiente aqui é caro (o desenrolar de `unroll` passos), então 8 sai ~30%
    #: mais lento por iteração — bem mais que os ~5% do AlphaZero. Ver §2.1.
    epochs_por_iter: int = 8

    temp_inicio: float = 1.0
    temp_fim: float = 0.25
    temp_frac: float = 0.5

    coef_valor: float = 0.25
    coef_recompensa: float = 1.0

    sims_avaliacao: int = 24


class MuZero(AgentBase):
    algo = "muzero"

    def __init__(self, cfg: MuZeroConfig = None, variant=None):
        cfg = cfg or MuZeroConfig()
        super().__init__(cfg, variant=variant or f"unroll{cfg.unroll}")
        keras.utils.set_random_seed(cfg.seed)

        self.h = build_representacao(cfg.board_size, cfg.net)
        self.g = build_dinamica(cfg.board_size, cfg.net)
        self.f = build_predicao(cfg.board_size, cfg.net)
        self.largura = PRESETS[cfg.net][0]

        self.optimizer = keras.optimizers.Adam(cfg.lr, clipnorm=cfg.max_grad_norm)
        self.optimizer.build(self._variaveis())

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed))
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)

        self.mcts = MCTS(self._avaliar_oculto, board_size=cfg.board_size,
                         gamma=cfg.gamma, num_simulations=cfg.num_simulations,
                         c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                         dirichlet_frac=cfg.dirichlet_frac,
                         fpu=cfg.fpu, q_normalizado=cfg.q_normalizado,
                         desempate=cfg.desempate,
                         dinamica=DinamicaAprendida(self._passo_dinamica),
                         rng=np.random.default_rng(cfg.seed + 2))

        forma = (cfg.board_size, cfg.board_size, N_CHANNELS)
        M, K = cfg.memory_size, cfg.unroll
        self._buf_obs = np.zeros((M, *forma), dtype=np.float32)
        self._buf_mask = np.ones((M, N_ACTIONS), dtype=bool)
        self._buf_act = np.zeros((M, K), dtype=np.int32)
        self._buf_pi = np.zeros((M, K + 1, N_ACTIONS), dtype=np.float32)
        self._buf_z = np.zeros((M, K + 1), dtype=np.float32)
        self._buf_r = np.zeros((M, K), dtype=np.float32)
        self._pos, self._cheio = 0, 0

    def _variaveis(self):
        return (self.h.trainable_variables + self.g.trainable_variables
                + self.f.trainable_variables)

    def on_model_reloaded(self):
        self.optimizer = keras.optimizers.Adam(self.cfg.lr,
                                               clipnorm=self.cfg.max_grad_norm)
        self.optimizer.build(self._variaveis())

    # -------------------------------------------------------- modelo -> busca
    @property
    def model(self):
        """O `AgentBase` salva `self.model`; para o MuZero o que interessa é `h`+`f`."""
        return self._modelo_politica()

    @model.setter
    def model(self, _):
        pass          # o estado real vive em h, g, f

    def _modelo_politica(self):
        """Modelo `observação → [logits, valor]`, para a política pura e o checkpoint."""
        if getattr(self, "_mp", None) is None and hasattr(self, "h"):
            inp = keras.Input(shape=(self.cfg.board_size, self.cfg.board_size,
                                     N_CHANNELS))
            logits, valor = self.f(self.h(inp))
            self._mp = keras.Model(inp, [logits, valor], name="muzero_politica")
        return getattr(self, "_mp", None)

    #: Teto do valor antes do `symexp`, igual ao do AlphaZero: uma cabeça que divergiu
    #: vira número grande e finito em vez de envenenar a árvore inteira.
    LIMITE_SYMLOG = 6.0

    @staticmethod
    def _symlog(x):
        return tf.sign(x) * tf.math.log1p(tf.abs(x))

    @staticmethod
    def _symexp(x):
        x = tf.clip_by_value(x, -MuZero.LIMITE_SYMLOG, MuZero.LIMITE_SYMLOG)
        return tf.sign(x) * tf.math.expm1(tf.abs(x))

    def _valor_real(self, valor):
        """A escala que o MCTS precisa: ele soma `recompensa + γ·valor`, e a recompensa
        é a que a rede `g` prevê, na escala do mundo."""
        return self._symexp(valor) if self.cfg.valor_symlog else valor

    @tf.function(reduce_retracing=True)
    def _repr_predicao(self, obs, mask):
        s = self.h(obs, training=False)
        logits, valor = self.f(s, training=False)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        return s, tf.nn.softmax(logits), self._valor_real(tf.squeeze(valor, -1))

    @tf.function(reduce_retracing=True)
    def _predicao(self, s):
        logits, valor = self.f(s, training=False)
        return tf.nn.softmax(logits), self._valor_real(tf.squeeze(valor, -1))

    @tf.function(reduce_retracing=True)
    def _dinamica_tf(self, s, planos):
        novo, r = self.g([s, planos], training=False)
        return novo, tf.squeeze(r, -1)

    def _planos_de_acao(self, acoes, n):
        b = self.cfg.board_size
        planos = np.zeros((n, b, b, N_ACTIONS), dtype=np.float32)
        planos[np.arange(n), :, :, acoes] = 1.0
        return planos

    def _passo_dinamica(self, estados, acoes):
        """A interface que `DinamicaAprendida` consome."""
        planos = self._planos_de_acao(acoes, len(acoes))
        novo, r = self._dinamica_tf(tf.convert_to_tensor(estados),
                                    tf.convert_to_tensor(planos))
        return novo.numpy(), r.numpy()

    def _avaliar_oculto(self, estados, mask):
        """Priors e valores a partir de estados ocultos — o que o MCTS chama."""
        p, v = self._predicao(tf.convert_to_tensor(np.asarray(estados,
                                                              dtype=np.float32)))
        p = p.numpy()
        m = np.asarray(mask)
        if m.shape == p.shape and not m.all():
            p = np.where(m, p, 0.0)
            p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        return p, v.numpy()

    def politica(self):
        """Política pura de `h`+`f`, sem busca — a curva oficial do contrato."""
        modelo = self._modelo_politica()

        def fn(obs, mask):
            logits, _ = modelo(obs, training=False)
            return np.where(mask, np.asarray(logits), MASK_NEG).astype(np.float32)
        return fn

    def _busca(self, obs, mask, ruido=False, busca=None):
        """Roda o MCTS a partir da observação: `h` uma vez, depois só a dinâmica."""
        s, priors, valores = self._repr_predicao(tf.convert_to_tensor(obs),
                                                 tf.convert_to_tensor(mask))
        arvore = busca if busca is not None else self.mcts
        return arvore.run(s.numpy(), mask, s.numpy(), adicionar_ruido=ruido)

    def avaliar_com_busca(self, episodes=1000, num_simulations=None, seed=123):
        """O protocolo oficial, mas escolhendo com MCTS — a **coluna separada** da tabela.

        A curva do contrato mede a política pura, e é isso que torna as curvas comparáveis:
        a busca gasta `num_simulations` avaliações de rede por jogada contra 1 do PPO.
        Reportar, porém, é obrigação — um algoritmo que existe para buscar, medido só sem
        buscar, é meia medição. Mesmo desenho do `AlphaZero.avaliar_com_busca`, com uma
        diferença que é justamente o ponto do MuZero: a árvore percorre `g`, não o
        `VecSnake`. O ambiente aqui só avança o jogo de verdade entre as jogadas.

        Não passa por `snakeai.eval` porque a interface de política recebe só observação e
        máscara, e a busca precisa devolver contagens de visita. O protocolo — episódios,
        semente, greedy — é o mesmo.
        """
        cfg = self.cfg
        n = min(cfg.eval_envs, 64)
        env = VecSnake(n, cfg.board_size, rng=np.random.default_rng(seed))
        busca = MCTS(self._avaliar_oculto, board_size=cfg.board_size, gamma=cfg.gamma,
                     num_simulations=num_simulations or cfg.sims_avaliacao,
                     c_puct=cfg.c_puct, fpu=cfg.fpu, q_normalizado=cfg.q_normalizado,
                     desempate=cfg.desempate,
                     dinamica=DinamicaAprendida(self._passo_dinamica),
                     rng=np.random.default_rng(seed))
        obs, mask = env.reset()
        por_env = int(np.ceil(episodes / n))
        coletados = [[] for _ in range(n)]
        faltam, vitorias = n, 0

        while faltam > 0:
            visitas, _ = self._busca(obs, mask, busca=busca)
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

    # -------------------------------------------------------------------- coleta
    def temperatura(self):
        """Escalar (fração do treino) ou `(N,)` (por lance do episódio, o do paper).

        Ver a nota homônima em `alphazero.py`: com o agendamento por fração do treino,
        metade do orçamento inteiro é jogada com τ = 1, inclusive nas posições apertadas.
        """
        cfg = self.cfg
        if cfg.temp_passos > 0:
            return np.where(self.env.steps < cfg.temp_passos,
                            cfg.temp_inicio, cfg.temp_fim).astype(np.float64)
        f = min(1.0, self.frac() / max(cfg.temp_frac, 1e-9))
        return cfg.temp_inicio + f * (cfg.temp_fim - cfg.temp_inicio)

    def collect(self):
        cfg = self.cfg
        T, N, K = cfg.rollout, cfg.num_envs, cfg.unroll

        obs_b = np.empty((T, N, cfg.board_size, cfg.board_size, N_CHANNELS), np.float32)
        mask_b = np.empty((T, N, N_ACTIONS), bool)
        pi_b = np.empty((T, N, N_ACTIONS), np.float32)
        v_b = np.empty((T, N), np.float32)
        act_b = np.empty((T, N), np.int32)
        rew_b = np.empty((T, N), np.float32)
        done_b = np.empty((T, N), np.float32)

        scores, vitorias, temps = [], 0, []
        for t in range(T):
            obs_b[t], mask_b[t] = self.obs, self.mask
            # com `temp_passos` a temperatura depende do lance de cada ambiente, e os N
            # ambientes estão em lances diferentes: tem que ser lida a cada passo
            temp = self.temperatura()
            temps.append(float(np.mean(temp)))
            visitas, valores = self._busca(self.obs, self.mask, ruido=True)
            pi = MCTS.politica_das_visitas(visitas, temp)
            # o alvo de treino não precisa ser a distribuição que escolheu a ação: no
            # AlphaZero/MuZero ele é a contagem de visitas crua
            pi_b[t] = (pi if cfg.temp_alvo <= 0
                       else MCTS.politica_das_visitas(visitas, cfg.temp_alvo))
            v_b[t] = valores
            a = (pi.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)
            act_b[t] = a
            self.obs, self.mask, r, d, info = self.env.step(a)
            self.registra_fim(info)
            if info["trunc_idx"].size:       # fome é truncamento, não terminação
                _, _, v_f = self._repr_predicao(
                    tf.convert_to_tensor(info["final_obs"]),
                    tf.convert_to_tensor(info["final_mask"]))
                r = self.bootstrap_truncados(info, r, v_f.numpy(), cfg.gamma)
            rew_b[t], done_b[t] = r, d.astype(np.float32)
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        # alvo de valor por passo: n passos + bootstrap no valor da busca
        # O `n` encolhe no fim da janela. Com `rollout=16` e `n_step=10`, o estado
        # `t + n_step` está fora do buffer para todo `t >= 6` — e a versão anterior
        # simplesmente **não fazia bootstrap** nesses casos: dez dos dezesseis passos
        # tratavam o fim da coleta como fim de episódio, e num jogo de recompensa esparsa
        # isso é um alvo quase sempre igual a zero, que ainda por cima realimenta a busca.
        # Encurtar o horizonte e fazer bootstrap no último estado disponível troca um
        # pouco de viés de horizonte por um alvo que não é puxado para zero.
        # Ver `docs/REVISAO_ALGORITMOS.md` §2.5.
        # `bootstrap_fim_janela` acrescenta uma linha `T`: o valor da REDE no estado em que
        # a coleta parou. Menos preciso que o resto do vetor, que é valor de busca — e
        # ainda assim melhor que tratar o fim da janela como fim de episódio.
        if cfg.bootstrap_fim_janela:
            _, _, v_fim = self._repr_predicao(tf.convert_to_tensor(self.obs),
                                              tf.convert_to_tensor(self.mask))
            v_boot = np.concatenate([v_b, v_fim.numpy().astype(np.float32)[None]], axis=0)
            limite = T                  # o passo T-1 passa a ter para onde olhar
        else:
            v_boot = v_b
            limite = T - 1              # o padrão do §2.5: um passo sem bootstrap, e só ele

        z = np.zeros((T, N), np.float32)
        for t in range(T):
            g = np.zeros(N, np.float32)
            desc = np.ones(N, np.float32)
            vivo = np.ones(N, bool)
            n = min(cfg.n_step, limite - t)     # 0 só no último passo, e só sem bootstrap
            for k in range(n):
                g += desc * rew_b[t + k] * vivo
                vivo &= done_b[t + k] < 0.5
                desc *= cfg.gamma
            if n > 0:
                g += desc * v_boot[t + n] * vivo
            else:                               # t = T-1: não há estado seguinte aqui
                g += rew_b[t]
            z[t] = g

        # cada amostra guarda o desenrolar de K passos que vem depois dela
        validos = max(0, T - K)
        if validos:
            idx = np.arange(validos)
            self._guardar(
                obs_b[idx].reshape(-1, *obs_b.shape[2:]),
                mask_b[idx].reshape(-1, N_ACTIONS),
                np.stack([act_b[idx + k] for k in range(K)], axis=-1).reshape(-1, K),
                np.stack([pi_b[idx + k] for k in range(K + 1)], axis=1)
                  .transpose(0, 2, 1, 3).reshape(-1, K + 1, N_ACTIONS),
                np.stack([z[idx + k] for k in range(K + 1)], axis=-1).reshape(-1, K + 1),
                np.stack([rew_b[idx + k] for k in range(K)], axis=-1).reshape(-1, K),
            )

        self.global_step += T * N
        self.episodes += len(scores)
        return {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "temperatura": float(np.mean(temps)),
            "valor_busca": float(v_b.mean()),
            "memoria": self._cheio,
        }

    def _guardar(self, obs, mask, act, pi, z, r):
        k = len(obs)
        idx = (self._pos + np.arange(k)) % self.cfg.memory_size
        self._buf_obs[idx] = obs
        self._buf_mask[idx] = mask
        self._buf_act[idx] = act
        self._buf_pi[idx] = pi
        self._buf_z[idx] = z
        self._buf_r[idx] = r
        self._pos = int((self._pos + k) % self.cfg.memory_size)
        self._cheio = min(self._cheio + k, self.cfg.memory_size)

    # -------------------------------------------------------------------- treino
    @tf.function(reduce_retracing=True)
    def _passo(self, obs, mask, act, pi_alvo, z, r_alvo, coef_v, coef_r):
        K = tf.shape(act)[1]
        with tf.GradientTape() as tape:
            s = self.h(obs, training=True)
            logits, valor = self.f(s, training=True)
            logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp = tf.nn.log_softmax(logits)

            alvo_v = self._symlog(z) if self.cfg.valor_symlog else z
            perda_pi = -tf.reduce_mean(tf.reduce_sum(pi_alvo[:, 0] * logp, -1))
            perda_v = tf.reduce_mean(tf.square(tf.squeeze(valor, -1) - alvo_v[:, 0]))
            perda_r = tf.constant(0.0)

            for k in range(self.cfg.unroll):
                planos = tf.one_hot(act[:, k], N_ACTIONS)[:, None, None, :]
                planos = tf.tile(planos, [1, self.cfg.board_size,
                                          self.cfg.board_size, 1])
                s, rec = self.g([s, planos], training=True)
                # escala de gradiente de 1/2: sem ela o gradiente que chega em `h` cresce
                # com o número de passos do desenrolar
                s = s * 0.5 + tf.stop_gradient(s) * 0.5

                logits_k, valor_k = self.f(s, training=True)
                logp_k = tf.nn.log_softmax(logits_k)
                perda_pi += -tf.reduce_mean(
                    tf.reduce_sum(pi_alvo[:, k + 1] * logp_k, -1))
                perda_v += tf.reduce_mean(
                    tf.square(tf.squeeze(valor_k, -1) - alvo_v[:, k + 1]))
                # a âncora do modelo no mundo real: sem ela a dinâmica pode inventar
                # qualquer física internamente consistente
                perda_r += tf.reduce_mean(
                    tf.square(tf.squeeze(rec, -1) - r_alvo[:, k]))

            perda = perda_pi + coef_v * perda_v + coef_r * perda_r

        variaveis = (self.h.trainable_variables + self.g.trainable_variables
                     + self.f.trainable_variables)
        grads = tape.gradient(perda, variaveis)
        self.optimizer.apply_gradients(zip(grads, variaveis))
        return perda_pi, perda_v, perda_r

    def _aprender(self):
        cfg = self.cfg
        if self._cheio < cfg.batch_size:
            return None
        lr = cfg.lr
        if cfg.lr_final > 0:
            lr = self.linear(cfg.lr, cfg.lr_final)
            self.optimizer.learning_rate.assign(lr)
        saidas = []
        for _ in range(cfg.epochs_por_iter):
            i = self.rng.integers(0, self._cheio, size=cfg.batch_size)
            p, v, r = self._passo(
                tf.convert_to_tensor(self._buf_obs[i]),
                tf.convert_to_tensor(self._buf_mask[i]),
                tf.convert_to_tensor(self._buf_act[i]),
                tf.convert_to_tensor(self._buf_pi[i]),
                tf.convert_to_tensor(self._buf_z[i]),
                tf.convert_to_tensor(self._buf_r[i]),
                cfg.coef_valor, cfg.coef_recompensa,
            )
            saidas.append((float(p), float(v), float(r)))
        p, v, r = (float(np.mean(x)) for x in zip(*saidas))
        return {"perda_pi": p, "perda_v": v, "perda_r": r, "lr": float(lr),
                "atualizacoes": cfg.epochs_por_iter}

    def iterate(self):
        stats = self.collect()
        treino = self._aprender()
        if treino:
            stats.update(treino)
        return stats

    # ---------------------------------------------------------------- checkpoint
    def salvar(self, tag="last"):
        for nome, rede in (("h", self.h), ("g", self.g), ("f", self.f)):
            rede.save(os.path.join(self.cfg.ckpt_dir,
                                   f"{self.algo}_{tag}_{nome}.keras"))
        super().salvar(tag)

    def retomar(self, tag="last"):
        caminhos = {n: os.path.join(self.cfg.ckpt_dir, f"{self.algo}_{tag}_{n}.keras")
                    for n in ("h", "g", "f")}
        if not all(os.path.exists(c) for c in caminhos.values()):
            return False
        self.h = keras.models.load_model(caminhos["h"])
        self.g = keras.models.load_model(caminhos["g"])
        self.f = keras.models.load_model(caminhos["f"])
        self._mp = None
        return super().retomar(tag)
