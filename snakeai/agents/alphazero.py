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

**Esse "24" tem letra miúda, e ela importa.** A heurística usada na medição é *negativa*
(`−distância`). O PUCT dá `Q = 0` a um filho ainda não visitado, então numa escala negativa
esse `0` é otimista e força a busca a experimentar todo mundo. O valor que este agente
aprende é positivo — recompensa `+1` por maçã e cabeça linear; a execução de 5 M passos
mede `valor_raiz` indo de 0,26 a **3,5**. Nessa escala o mesmo `0` vira pessimismo, e o
bônus de exploração (`c_puct · P · √N`) só o cobre onde o prior já é alto: a busca passa a
confirmar a rede em vez de corrigi-la. Com a mesma heurística somada de uma constante —
ranking idêntico — o score cai de 21,7 para **0,00**.

O resultado dessa execução: política pura em **10,62** (pico de 13,03 em 3,0 M), com
**86,9% dos episódios terminando por fome**, `perda_pi` em 0,016 (a rede reproduz o alvo
quase perfeitamente) e `perda_v` 58× maior que ela. Três problemas somados — a busca que
não discorda, o valor não normalizado que domina o tronco, e a temperatura que transforma o
alvo em rótulo duro. Os consertos estão atrás de flags, desligados por padrão; as ablações
estão em `93_alphazero_ablacoes`. Ver `docs/BUSCA_DEGENERADA.md`.

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

    #: Learning rate no fim do orçamento, decaindo linearmente a partir de `lr`. `0`
    #: mantém o `lr` constante, que é o padrão histórico daqui — e o único agente do
    #: repositório sem decaimento (o PPO e o ACKTR vão de 3e-4 a 5e-5).
    #:
    #: A execução de 5 M passos mostra por que isso importa: depois de 3 M o score de
    #: avaliação oscila entre 9,6 e 12,5 sem tendência, e o `best` (13,03 em 3,0 M) fica
    #: **2,4 pontos acima** do `last` (10,62), que é o número oficial. Passo grande demais
    #: no fim de um treino é exatamente esse desenho de curva.
    lr_final: float = 0.0
    max_grad_norm: float = 5.0
    batch_size: int = 512
    epochs_por_iter: int = 1
    memory_size: int = 100_000

    #: Como o Q de um filho ainda não visitado entra no PUCT: `"zero"` (a convenção do
    #: AlphaZero, e o padrão histórico daqui) ou `"pai"` (o valor do próprio nó).
    #: Ver `MinMax` em `search/mcts.py` e `docs/BUSCA_DEGENERADA.md`.
    fpu: str = "zero"
    #: Normalização min-max do Q dentro da árvore (MuZero, Apêndice B). Devolve `c_puct`
    #: à escala em que foi calibrado, num jogo cujo valor não é limitado a [-1, 1].
    q_normalizado: bool = False

    #: Temperatura da amostragem na coleta. Cai de `temp_inicio` a `temp_fim` ao longo de
    #: `temp_frac` do **treino** — não do episódio.
    #:
    #: E ela não governa só a coleta: sem `temp_alvo`, a mesma distribuição temperada vira
    #: o alvo de treino. Com `temp_fim = 0,25` as contagens de visita são elevadas à quarta
    #: potência, e nas contagens medidas com 32 simulações isso leva a entropia do alvo de
    #: 0,66 a **0,015** — rótulo duro. Da metade do treino em diante a rede aprende a ter
    #: confiança máxima no argmax de uma busca de 32 simulações, que é o oposto de destilar
    #: a distribuição de visitas. Ver `docs/BUSCA_DEGENERADA.md`.
    temp_inicio: float = 1.0
    temp_fim: float = 0.25
    temp_frac: float = 0.5

    #: Agendamento **canônico** do AlphaZero, desligado por padrão: com `temp_passos > 0`
    #: a temperatura passa a depender do lance dentro do episódio (τ = `temp_inicio` nos
    #: primeiros `temp_passos` lances, `temp_fim` no resto) em vez da fração do treino.
    #: Substitui `temp_frac` por completo — não são dois agendamentos somados.
    temp_passos: int = 0

    #: Temperatura usada para construir o **alvo** de política, independente da usada para
    #: agir. `0` mantém o comportamento atual (o alvo é a mesma π temperada que escolheu a
    #: ação); `1.0` é o AlphaZero de verdade — o alvo é a distribuição de visitas crua, e
    #: a temperatura fica sendo só um botão de exploração.
    temp_alvo: float = 0.0

    vf_coef: float = 1.0
    ent_coef: float = 0.0     # a exploração vem do ruído de Dirichlet, não da entropia

    #: Treinar o valor em **symlog** em vez de na escala crua (DreamerV3; a mesma
    #: transformação de `nets/dreamer.py`). Desligado por padrão.
    #:
    #: O AlphaZero original treina o valor contra o resultado da partida, em `[-1, 1]`, e
    #: por isso `perda_v` e `perda_pi` nascem na mesma ordem de grandeza. Aqui o alvo é um
    #: retorno descontado **não normalizado**: com `γ = 0,997` e uma maçã a cada ~37 passos
    #: (o regime medido em 1 M de passos) ele vale ~9, e cresce conforme o agente melhora.
    #: A perda de política é uma entropia cruzada sobre 3 ações, presa perto de `ln 3` — o
    #: tronco compartilhado recebe um gradiente 5 a 25× maior vindo do valor, medido em
    #: `tools/diag_busca.py`. O PPO não tem esse problema porque normaliza a vantagem por
    #: minilote, o que torna o gradiente de política invariante à escala do valor.
    #:
    #: A busca continua vendo o valor na escala **real** — `_frente` desfaz o symlog antes
    #: de devolver. A transformação é só a representação que a rede aprende.
    valor_symlog: bool = False

    #: Empate exato no PUCT: `"ordem"` (o primeiro filho do dicionário, que é sempre
    #: *virar à esquerda*) ou `"aleatorio"`. Ver `MCTS.desempate`.
    desempate: str = "ordem"

    #: Fazer bootstrap no **último** passo da janela de coleta. Com `rollout=16` o passo
    #: `t = T-1` não tem estado seguinte dentro do buffer, e hoje o alvo dele é a
    #: recompensa nua — sem desconto e sem continuação. Num jogo de recompensa esparsa
    #: isso é um zero em 1/16 das amostras, e um zero que ensina "aqui não há futuro".
    #: Ligado, o valor da rede no estado em que a coleta parou fecha a janela.
    #: É o valor da **rede**, não o da busca: no fim da janela não houve busca. Menos
    #: preciso que os outros bootstraps do vetor, e ainda assim melhor que nenhum.
    bootstrap_fim_janela: bool = False

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
                         fpu=cfg.fpu, q_normalizado=cfg.q_normalizado,
                         desempate=cfg.desempate,
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
    @staticmethod
    def _symlog(x):
        return tf.sign(x) * tf.math.log1p(tf.abs(x))

    #: Teto do valor **antes** do `symexp`. `symlog` de um retorno realista neste jogo vale
    #: no máximo `ln(1 + 97) ≈ 4,6` (tabuleiro cheio, sem desconto); `6` dá folga de sobra e
    #: transforma uma cabeça que divergiu em um número grande e finito (`symexp(6) ≈ 403`)
    #: em vez de `symexp(40) ≈ 2·10¹⁷`, que envenena a árvore inteira e não volta. Um treino
    #: de 8 horas não pode depender de a cabeça nunca passar do ponto.
    LIMITE_SYMLOG = 6.0

    @staticmethod
    def _symexp(x):
        x = tf.clip_by_value(x, -AlphaZero.LIMITE_SYMLOG, AlphaZero.LIMITE_SYMLOG)
        return tf.sign(x) * tf.math.expm1(tf.abs(x))

    @tf.function(reduce_retracing=True)
    def _frente(self, obs, mask):
        logits, valor = self.model(obs, training=False)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), MASK_NEG))
        valor = tf.squeeze(valor, -1)
        if self.cfg.valor_symlog:
            # o MCTS soma `recompensa + γ·valor` com recompensas de verdade (+1 por maçã):
            # ele precisa da escala real, não da comprimida
            valor = self._symexp(valor)
        return tf.nn.softmax(logits), valor

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
                     fpu=cfg.fpu, q_normalizado=cfg.q_normalizado,
                     desempate=cfg.desempate,
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
        """Escalar (agendamento por fração do treino) ou `(N,)` (por lance do episódio).

        O padrão é o primeiro. O segundo, ligado por `temp_passos`, é o do paper: a
        estocasticidade serve para diversificar a **abertura**, e no meio do jogo — quando
        o tabuleiro aperta e uma ação fora do melhor ramo mata — a política é greedy. Com o
        agendamento por fração do treino, metade do orçamento inteiro é jogada com τ = 1,
        inclusive nas posições apertadas.
        """
        cfg = self.cfg
        if cfg.temp_passos > 0:
            return np.where(self.env.steps < cfg.temp_passos,
                            cfg.temp_inicio, cfg.temp_fim).astype(np.float64)
        f = min(1.0, self.frac() / max(cfg.temp_frac, 1e-9))
        return cfg.temp_inicio + f * (cfg.temp_fim - cfg.temp_inicio)

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

        obs_b = np.empty((T, N, cfg.board_size, cfg.board_size, N_CHANNELS), np.float32)
        mask_b = np.empty((T, N, N_ACTIONS), bool)
        pi_b = np.empty((T, N, N_ACTIONS), np.float32)
        v_raiz = np.empty((T, N), np.float32)
        rew_b = np.empty((T, N), np.float32)
        done_b = np.empty((T, N), np.float32)

        scores, vitorias = [], 0
        temps = []
        for t in range(T):
            obs_b[t], mask_b[t] = self.obs, self.mask
            # com `temp_passos` a temperatura depende do lance de cada ambiente, e os N
            # ambientes estão em lances diferentes: tem que ser lida a cada passo.
            temp = self.temperatura()
            temps.append(float(np.mean(temp)))
            visitas, valores = self.mcts.run(self.env.get_state(), self.mask, self.obs,
                                             adicionar_ruido=True)
            pi = MCTS.politica_das_visitas(visitas, temp)
            # O alvo de treino não precisa ser a distribuição que escolheu a ação. No
            # AlphaZero ele é a contagem de visitas crua; temperar o alvo faz a rede
            # aprender a própria confiança amplificada, que realimenta a busca.
            pi_b[t] = (pi if cfg.temp_alvo <= 0
                       else MCTS.politica_das_visitas(visitas, cfg.temp_alvo))
            v_raiz[t] = valores
            a = (pi.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)

            self.obs, self.mask, r, d, info = self.env.step(a)
            self.registra_fim(info)
            if info["trunc_idx"].size:       # fome é truncamento, não terminação
                _, v_f = self._avaliar(info["final_obs"], info["final_mask"])
                r = self.bootstrap_truncados(info, r, v_f, cfg.gamma)
            rew_b[t], done_b[t] = r, d.astype(np.float32)
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        # alvo de valor: retorno de n passos com bootstrap no VALOR DA BUSCA. Usar o
        # valor da rede aqui jogaria fora justamente o refino que a busca produziu.
        # O `n` encolhe no fim da janela. Com `rollout=16` e `n_step=10`, o estado
        # `t + n_step` está fora do buffer para todo `t >= 6` — e a versão anterior
        # simplesmente **não fazia bootstrap** nesses casos: dez dos dezesseis passos
        # tratavam o fim da coleta como fim de episódio, e num jogo de recompensa esparsa
        # isso é um alvo quase sempre igual a zero, que ainda por cima realimenta a busca.
        # Encurtar o horizonte e fazer bootstrap no último estado disponível troca um
        # pouco de viés de horizonte por um alvo que não é puxado para zero.
        # Ver `docs/REVISAO_ALGORITMOS.md` §2.5.
        # `bootstrap_fim_janela` acrescenta uma linha `T` ao vetor de valores: o valor da
        # REDE no estado em que a coleta parou. É menos preciso que o resto do vetor, que
        # é valor de busca — mas fecha o buraco do último passo, onde hoje o alvo é a
        # recompensa nua. Ver a nota do campo no config.
        if cfg.bootstrap_fim_janela:
            _, v_fim = self._avaliar(self.obs, self.mask)
            v_boot = np.concatenate([v_raiz, np.asarray(v_fim, np.float32)[None]], axis=0)
            limite = T                  # o passo T-1 passa a ter para onde olhar
        else:
            v_boot = v_raiz
            limite = T - 1              # o padrão do §2.5: um passo sem bootstrap, e só ele

        z = np.zeros((T, N), np.float32)
        for t in range(T):
            g = np.zeros(N, np.float32)
            desconto = np.ones(N, np.float32)
            vivo = np.ones(N, bool)
            n = min(cfg.n_step, limite - t)     # 0 só no último passo, e só sem bootstrap
            for k in range(n):
                g += desconto * rew_b[t + k] * vivo
                vivo &= done_b[t + k] < 0.5
                desconto *= cfg.gamma
            if n > 0:
                g += desconto * v_boot[t + n] * vivo
            else:                               # t = T-1: não há estado seguinte aqui
                g += rew_b[t]
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
            "temperatura": float(np.mean(temps)),
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
            alvo_v = self._symlog(z) if self.cfg.valor_symlog else z
            perda_v = tf.reduce_mean(tf.square(valor - alvo_v))
            entropia = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp) * logp, axis=-1))
            perda = perda_pi + vf_coef * perda_v - ent_coef * entropia

        grads = tape.gradient(perda, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return perda_pi, perda_v, entropia

    def _aprender(self):
        cfg = self.cfg
        if self._cheio < cfg.batch_size:
            return None
        lr = cfg.lr
        if cfg.lr_final > 0:
            lr = self.linear(cfg.lr, cfg.lr_final)
            self.optimizer.learning_rate.assign(lr)
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
        # `atualizacoes` é o eixo do §2.1 e não dá para reconstruir do config — o registro
        # da execução de 5 M nasceu com ele vazio, e é justamente o número que compara este
        # agente com as ~38.300 atualizações do PPO no mesmo orçamento de ambiente.
        return {"perda_pi": p, "perda_v": v, "entropia": e, "lr": float(lr),
                "atualizacoes": cfg.epochs_por_iter}

    def iterate(self):
        stats = self.collect()
        treino = self._aprender()
        if treino:
            stats.update(treino)
        return stats
