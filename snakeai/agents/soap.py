"""SOAP — *Sequential Option Advantage Propagation* (Ishida & Henriques, 2024).

Um algoritmo de **opções**: em vez de uma política, o agente aprende `Z` sub-políticas e
uma política de troca entre elas. A opção corrente é uma variável latente **discreta** que
o ambiente não mostra e ninguém supervisiona — o agente carrega uma crença `ζ_t(z)` sobre
qual opção está ativa, atualizada a cada passo pelo que ele viu e pelo que ele fez.

Por que isto tem sentido no Snake deste repositório
---------------------------------------------------
Porque a observação do contrato **não é markoviana**, e isso está documentado desde o
`docs/CANAL_DE_FOME.md`: os 5 canais não contêm o relógio da fome, e o limite é
`100 + 2·comprimento` passos sem comer. Dois estados visualmente idênticos, um com fome 5 e
outro com fome 105, valem coisas diferentes — e a rede não tem como saber.

O repositório já tentou uma resposta: um sexto canal com `fome / limite`. Ela custou a
comparabilidade (a entrada da rede muda, `comparable=False`) e **não funcionou** — 7,8
pontos abaixo, atrás em 17 dos 18 pontos de avaliação. O SOAP é a terceira resposta e a
única que cabe **dentro** do contrato: a informação que falta na observação passa a viver
num latente que atravessa os passos, sem tocar nos 5 canais.

Se a hipótese estiver certa, as opções vão se separar por regime — "caçar comida" contra
"desenrolar o corpo", ou algo que só o GIF revela. Se estiver errada, elas colapsam numa
só e a curva é a do PPO com 4× as cabeças. As duas leituras estão instrumentadas (ver
`docs/SOAP.md`), e a segunda é um resultado, não um fracasso de implementação.

As peças
--------
**Fatoração.** Sub-política `π_θ(a|s,z)` e transição de opção `π_ψ(z'|s,a,z)`. A segunda é
a contribuição de fatoração do paper: a próxima opção depende da **anterior** e da ação
tomada, não só do estado como no Option-Critic. É o que permite à opção persistir por conta
própria, em vez de ser re-sorteada a cada passo.

**Crença para a frente.** `ζ_t(z) := p(z_t | s_{0:t}, a_{0:t-1})`, atualizada por

``ζ_{t+1}(z') = Σ_z ζ_t(z) π_θ(a_t|s_t,z) π_ψ(z'|s_t,a_t,z) / α_t``

com ``α_t = Σ_z ζ_t(z) π_θ(a_t|s_t,z)`` — a probabilidade marginal da ação que o ambiente
de fato viu. `ζ` é **causal**: só depende do passado, que é a diferença central em relação
ao PPOEM do mesmo paper. O PPOEM usa o *forward-backward* inteiro e portanto atribui opções
em retrospecto, com informação que o agente não terá na hora de agir; o paper mostra que
isso degrada conforme a sequência cresce.

**A vantagem que propaga.** O gradiente de `log α_t` em relação aos parâmetros atravessa
`ζ`, que por sua vez depende de todos os passos anteriores. Derivar isso a mão daria uma
retropropagação pelo tempo; o paper mostra que ela colapsa numa recursão para trás fechada,
a *Generalized Option Advantage*:

``A^GOA_t(z') = Σ_z A^GAE_t(z) ζ_t(z) + (1−d_t)·[U_{t+1}(z') − E_{ζ_{t+1}}U_{t+1}]``
``U_t(z) = Σ_{z'} A^GOA_t(z') p_Θ(a_t,z'|s_t,z) / α_t``

Lida em português: escolher a ação `a_t` e cair na opção `z'` vale a vantagem imediata do
passo (média sobre a crença atual) **mais** o quanto `z'` é uma opção melhor que a média
para o futuro. O segundo termo é o que faz o agente trocar de opção por um motivo — sem
ele, `π_ψ` não recebe gradiente nenhum e as opções viram ruído.

A perda de política, na forma implementada
-------------------------------------------
O paper escreve a perda com `p_Θ` cru e o clipping em torno de `p_Θ_velho`. Aqui ela está
na forma algebricamente equivalente e numericamente melhor:

``L = − Σ_{z,z'} w(z,z') · min(ρ·A^GOA(z'), clip(ρ, 1±ε)·A^GOA(z'))``
``w(z,z') = ζ(z)·p_velho(a_t,z'|s_t,z)/α_t``      ``ρ = p_novo/p_velho``

`w` é a **responsabilidade** do par de opções: a posteriori de `(z_t, z_{t+1})` dado o
histórico e a ação tomada. Ela soma exatamente 1 sobre os pares, então a perda é uma média
ponderada de perdas de PPO, e `ρ` vive perto de 1 como num PPO comum. Multiplicar `p_Θ`
cru, como no paper, dá o mesmo gradiente a menos do fator constante `p_velho`, mas com
escala variando por amostra.

O controle experimental está embutido
-------------------------------------
Com `n_opcoes=1` o SOAP **é** o PPO, e não aproximadamente: `ζ ≡ 1`, `α_t = π(a_t|s_t)`,
`w ≡ 1`, `A^GOA = A^GAE` e a perda vira a do PPO com clipping. `tests/test_soap.py` prova
as três igualdades numericamente. Isso dá à linha da arena um controle que não depende de
ninguém acreditar na implementação — e transforma `n_opcoes` num eixo de ablação medido.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf

from ..env.vec_snake import N_ACTIONS, VecSnake
from ..eval import MASK_NEG
from ..nets import build_option_actor_critic
from ..otimizadores import cria_otimizador
from .base import AgentBase, BaseConfig
from .ppo import variancia_explicada

__all__ = ["SOAPConfig", "SOAP", "PoliticaComOpcoes", "gae_de_opcoes",
           "vantagem_de_opcao"]


# ------------------------------------------------------------------ estimadores
def gae_de_opcoes(rew, valores, done, pi_z, ultimo_v, gamma, lam):
    """GAE(λ) por **par de opções**, e a sua marginal sobre a próxima opção.

    ``A_t(z,z') = r_t + γ(1−d_t)V(s_{t+1},z') − V(s_t,z) + λγ(1−d_t) A_{t+1}(z')``
    ``A_t(z)    = Σ_{z'} π_ψ(z'|s_t,a_t,z) · A_t(z,z')``

    É o GAE do PPO com o crítico condicionado à opção. A recursão para trás usa a
    **marginal** do passo seguinte, `A_{t+1}(z')`, e não o par: no instante `t+1` a opção
    `z'` já é a corrente, e por qual par se chegou nela não importa mais.

    Formas
    ------
    `rew` : `(T, N, Z)` — indexado pela opção **de destino** `z'`. O eixo existe só por
        causa do bootstrap de truncamento por fome, que soma `γ·V(s_final, z')` e portanto
        depende de `z'`; sem truncamento as `Z` colunas são iguais.
    `valores` : `(T, N, Z)` · `done` : `(T, N)` · `pi_z` : `(T, N, Z, Z)` — `[t,n,z,z']`.
    `ultimo_v` : `(N, Z)` — o valor do estado logo após o último passo do segmento.

    Devolve `(a_marginal, a_par)`, `(T, N, Z)` e `(T, N, Z, Z)`.
    """
    T, N, Z = valores.shape
    a_marg = np.zeros((T, N, Z), dtype=np.float32)
    a_par = np.zeros((T, N, Z, Z), dtype=np.float32)
    prox_marg = np.zeros((N, Z), dtype=np.float32)

    for t in reversed(range(T)):
        v_prox = ultimo_v if t == T - 1 else valores[t + 1]
        continua = (1.0 - done[t])[:, None]                      # (N, 1), sobre z'
        # (N, Z') — tudo o que depende só da opção de destino
        futuro = rew[t] + gamma * continua * v_prox + lam * gamma * continua * prox_marg
        a_par[t] = futuro[:, None, :] - valores[t][:, :, None]   # menos V(s_t, z)
        a_marg[t] = np.einsum("nzy,nzy->nz", pi_z[t], a_par[t])
        prox_marg = a_marg[t]
    return a_marg, a_par


def vantagem_de_opcao(a_gae, zeta, zeta_final, done, p_conj, alpha):
    """`A^GOA` — a vantagem que o gradiente da **política de opções** consome.

    A recursão para trás do §5.2 do paper:

    ``A^GOA_t(z') = Σ_z A^GAE_t(z)ζ_t(z) + (1−d_t)[U_{t+1}(z') − E_{ζ_{t+1}}U_{t+1}]``
    ``U_t(z) = Σ_{z'} A^GOA_t(z') p_Θ(a_t,z'|s_t,z) / α_t``

    O primeiro termo não depende de `z'` — é a vantagem do passo, ponderada pela crença
    atual, e é o que o PPO comum já teria. Todo o conteúdo de opção está no segundo: a
    **utilidade centrada** de terminar o passo na opção `z'` em vez da opção média. Como
    ele é centrado, ele não desloca o gradiente da sub-política; ele só redistribui entre
    as opções, que é exatamente o que se quer.

    `U` é a utilidade retropropagada e é onde a recursão fecha: a utilidade de estar em `z`
    agora é a média das vantagens de opção que sair de `z` produz, normalizada por `α_t`.
    Essa normalização é o que impede que uma ação improvável no marginal amplifique o sinal
    de opção — sem ela, a recursão diverge nas primeiras iterações, quando `α` é pequeno.

    Formas: `a_gae`, `zeta` `(T, N, Z)`; `zeta_final` `(N, Z)`; `done` `(T, N)`;
    `p_conj` `(T, N, Z, Z)` com `p_conj[t,n,z,z'] = p_Θ(a_t,z'|s_t,z)`; `alpha` `(T, N)`.
    Devolve `(T, N, Z)`, indexado pela opção de **destino**.
    """
    T, N, Z = a_gae.shape
    goa = np.zeros((T, N, Z), dtype=np.float32)
    # No fim do segmento não há futuro medido: `U = 0`. É a mesma truncatura do GAE, e
    # significa que o último passo de cada rollout não recebe sinal de opção nenhum.
    u_prox = np.zeros((N, Z), dtype=np.float32)
    zeta_prox = zeta_final

    for t in reversed(range(T)):
        base = np.einsum("nz,nz->n", a_gae[t], zeta[t])          # (N,)
        centrado = u_prox - np.einsum("nz,nz->n", u_prox, zeta_prox)[:, None]
        goa[t] = base[:, None] + (1.0 - done[t])[:, None] * centrado
        u_prox = np.einsum("ny,nzy->nz", goa[t], p_conj[t]) / np.maximum(
            alpha[t], 1e-8)[:, None]
        zeta_prox = zeta[t]
    return goa


# ------------------------------------------------------------------- avaliação
class PoliticaComOpcoes:
    """A política do SOAP no formato que `snakeai.eval` consome, **com memória**.

    O `keras_policy` padrão assume uma rede sem estado: recebe `obs`, devolve logits. Aqui
    a ação sai da distribuição **marginal** `Σ_z ζ(z) π_θ(a|s,z)`, e `ζ` depende de todo o
    histórico do episódio. Avaliar sem a recorrência não daria erro — daria um número mais
    baixo, com `ζ` congelado no uniforme, e a conclusão "opções não ajudam aqui" viria de
    um defeito da medição. É a mesma armadilha que o `PoliticaRecorrente` do DreamerV3
    existe para evitar.

    O contrato com `evaluate` tem duas metades:

    * `__call__(obs, mask)` devolve `log` do marginal mascarado — e não os logits de uma
      sub-política. O protocolo é `argmax`, e o argmax do marginal é a ação que a política
      de fato escolhe; o argmax de qualquer `π_θ(·|s,z)` isolada seria outra política.
    * `apos_passo(acoes, done)` recebe a ação que **de fato** aconteceu — que pode não ser
      o argmax, se o filtro de segurança agiu — e onde o episódio terminou, para devolver
      `ζ` ao uniforme ali.

    Sem a segunda metade, `ζ` atravessaria a morte da cobra e levaria a crença de uma
    partida para dentro da próxima.
    """

    def __init__(self, modelo, n_opcoes):
        self.modelo = modelo
        self.z = int(n_opcoes)
        self.n = None
        self._grafo = tf.function(self._frente, reduce_retracing=True)

    def _frente(self, obs):
        return self.modelo(obs, training=False)

    def _garante(self, n):
        if self.n != n:
            self.n = n
            self.zeta = np.full((n, self.z), 1.0 / self.z, dtype=np.float32)

    def __call__(self, obs, mask):
        self._garante(obs.shape[0])
        la, lz, _ = self._grafo(tf.convert_to_tensor(obs, tf.float32))
        la = np.where(mask[:, None, :], la.numpy(), MASK_NEG)
        self._pi_a = _softmax(la)                                # (N, Z, A)
        self._lz = lz.numpy()
        marginal = np.einsum("nz,nza->na", self.zeta, self._pi_a)
        # log do marginal: `evaluate` faz argmax, e log é monótono — mas o `max_steps` do
        # protocolo também amostra da softmax quando `greedy=False`, e aí a escala importa
        return np.where(mask, np.log(marginal + 1e-12), MASK_NEG).astype(np.float32)

    def apos_passo(self, acoes, done):
        a = np.asarray(acoes, np.int32)
        linhas = np.arange(self.n)
        pi_a_t = self._pi_a[linhas, :, a]                        # (N, Z)
        pi_z = _softmax(self._lz[linhas, :, a, :])               # (N, Z, Z')
        alpha = np.maximum(np.einsum("nz,nz->n", self.zeta, pi_a_t), 1e-8)
        conj = pi_a_t[:, :, None] * pi_z
        self.zeta = (np.einsum("nz,nzy->ny", self.zeta, conj) / alpha[:, None]
                     ).astype(np.float32)
        fim = np.asarray(done, bool)
        if fim.any():
            self.zeta[fim] = 1.0 / self.z


def _softmax(x, eixo=-1):
    z = x - x.max(axis=eixo, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=eixo, keepdims=True)


# ----------------------------------------------------------------- configuração
@dataclass
class SOAPConfig(BaseConfig):
    num_envs: int = 512
    rollout: int = 32

    #: Quantas opções. O paper usa 4. **`1` é o controle**: com uma opção só o SOAP é
    #: literalmente o PPO, e a diferença entre as duas curvas é atribuível às opções e a
    #: mais nada.
    n_opcoes: int = 4

    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.002

    #: Bônus de entropia sobre `π_ψ(·|s,a,z)`. **Zero por padrão** — o paper não tem este
    #: termo, e ligá-lo é uma decisão sobre o colapso de opções, não sobre o algoritmo.
    #: Ver `docs/SOAP.md`: se as opções colapsarem, este é o primeiro botão a girar, e a
    #: execução que o gira ganha marca própria na variante.
    ent_opcao_coef: float = 0.0

    max_grad_norm: float = 0.5
    lr_start: float = 3e-4
    lr_end: float = 5e-5
    optimizer: str = "adam"
    epochs: int = 4
    minibatches: int = 32
    target_kl: float = 0.03

    shaping_start: float = 0.5
    shaping_frac: float = 0.25

    canal_fome: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.n_opcoes < 1:
            raise ValueError("n_opcoes precisa ser pelo menos 1")
        if self.canal_fome and self.comparable:
            raise ValueError(
                "canal_fome=True muda a observação de 5 para 6 canais e portanto a "
                "entrada da rede. Marque comparable=False e escreva o caveat.")

    @property
    def batch_size(self):
        return self.num_envs * self.rollout


# ---------------------------------------------------------------------- agente
class SOAP(AgentBase):
    """PPO sobre uma política com opções discretas, com a crença `ζ` carregada no tempo."""

    algo = "soap"

    def __init__(self, cfg: SOAPConfig = None, model=None, variant=None):
        cfg = cfg or SOAPConfig()
        super().__init__(cfg, variant=variant or self._variante(cfg))
        keras.utils.set_random_seed(cfg.seed)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed),
                            canal_fome=cfg.canal_fome)
        self.model = model or build_option_actor_critic(
            cfg.board_size, cfg.net, n_opcoes=cfg.n_opcoes,
            canais=self.env.n_channels)
        self.optimizer = self._novo_otimizador()
        self.obs, self.mask = self.env.reset()
        self.rng = np.random.default_rng(cfg.seed + 1)

        #: A crença sobre a opção corrente, **por ambiente**. Uniforme no início de cada
        #: episódio: no primeiro passo o agente não sabe nada sobre em que regime está, e
        #: fingir que sabe seria escolher uma opção por acidente de inicialização.
        self.zeta = np.full((cfg.num_envs, cfg.n_opcoes), 1.0 / cfg.n_opcoes,
                            dtype=np.float32)

    @staticmethod
    def _variante(cfg):
        """O que desvia do SOAP oficial entra no nome — `load_all` agrupa por variante."""
        marcas = []
        if cfg.n_opcoes != type(cfg).n_opcoes:
            marcas.append(f"op{cfg.n_opcoes}")
        if cfg.ent_opcao_coef:
            marcas.append(f"entz{cfg.ent_opcao_coef:g}")
        return "+".join([cfg.net] + marcas)

    def _novo_otimizador(self):
        opt = cria_otimizador(self.cfg.optimizer, self.cfg.lr_start,
                              clipnorm=self.cfg.max_grad_norm)
        opt.build(self.model.trainable_variables)
        return opt

    def on_model_reloaded(self):
        self.optimizer = self._novo_otimizador()

    # ------------------------------------------------------------ agendamentos
    def lr(self):
        return self.linear(self.cfg.lr_start, self.cfg.lr_end)

    def ent_coef(self):
        return self.linear(self.cfg.ent_coef_start, self.cfg.ent_coef_end)

    def shaping(self):
        f = self.frac()
        return max(0.0, self.cfg.shaping_start * (1.0 - f / self.cfg.shaping_frac))

    # -------------------------------------------------------------- avaliação
    def politica(self):
        return self.politica_do_modelo(self.model)

    def politica_do_modelo(self, modelo):
        return PoliticaComOpcoes(modelo, self.cfg.n_opcoes)

    # ----------------------------------------------------------------- rollout
    @staticmethod
    @tf.function(reduce_retracing=True)
    def _frente(model, obs):
        """Logits **crus** das duas políticas e o valor por opção.

        Crus porque a máscara vale para a ação e não para a opção, e porque a crença é
        atualizada em NumPy — manter a mascarada dentro do grafo só duplicaria a regra.
        """
        return model(obs, training=False)

    def collect(self):
        cfg = self.cfg
        T, N, Z = cfg.rollout, cfg.num_envs, cfg.n_opcoes
        b, c = cfg.board_size, self.env.n_channels
        linhas = np.arange(N)

        obs_buf = np.empty((T, N, b, b, c), dtype=np.float32)
        mask_buf = np.empty((T, N, N_ACTIONS), dtype=bool)
        act_buf = np.empty((T, N), dtype=np.int32)
        zeta_buf = np.empty((T, N, Z), dtype=np.float32)
        alpha_buf = np.empty((T, N), dtype=np.float32)
        val_buf = np.empty((T, N, Z), dtype=np.float32)
        #: `π_θ(a_t|s_t,z)` e `π_ψ(·|s_t,a_t,z)` do momento da coleta. São o denominador
        #: da razão do PPO e a marginalização do GAE; sem eles gravados, a razão viraria
        #: `π/π = 1` e o clipping deixaria de significar qualquer coisa.
        pia_buf = np.empty((T, N, Z), dtype=np.float32)
        piz_buf = np.empty((T, N, Z, Z), dtype=np.float32)
        rew_buf = np.empty((T, N, Z), dtype=np.float32)
        done_buf = np.empty((T, N), dtype=np.float32)

        shaping = self.shaping()
        scores, vitorias = [], 0
        ent_marginal, persistencia, divergencia = [], [], []

        for t in range(T):
            obs_buf[t], mask_buf[t] = self.obs, self.mask
            zeta_buf[t] = self.zeta

            la, lz, v = self._frente(self.model, tf.convert_to_tensor(self.obs))
            la = np.where(self.mask[:, None, :], la.numpy(), MASK_NEG)
            pi_a = _softmax(la)                                   # (N, Z, A)
            val_buf[t] = v.numpy()

            marginal = np.einsum("nz,nza->na", self.zeta, pi_a)
            marginal /= marginal.sum(1, keepdims=True)
            a = (marginal.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)
            act_buf[t] = a

            pi_a_t = pi_a[linhas, :, a]                           # (N, Z)
            pi_z = _softmax(lz.numpy()[linhas, :, a, :])          # (N, Z, Z')
            alpha = np.maximum(np.einsum("nz,nz->n", self.zeta, pi_a_t), 1e-8)
            alpha_buf[t], pia_buf[t], piz_buf[t] = alpha, pi_a_t, pi_z

            ent_marginal.append(
                float(-(marginal * np.log(marginal + 1e-12)).sum(1).mean()))
            # Distância L1 média entre as sub-políticas — o termômetro de colapso. Zero
            # significa que as `Z` opções fazem a mesma coisa e o SOAP virou um PPO caro.
            divergencia.append(float(np.abs(
                pi_a - pi_a.mean(1, keepdims=True)).sum(-1).mean()))

            conj = pi_a_t[:, :, None] * pi_z                      # p_Θ(a_t, z'|s_t, z)
            zeta_prox = np.einsum("nz,nzy->ny", self.zeta, conj) / alpha[:, None]
            persistencia.append(
                float((zeta_prox.argmax(1) == self.zeta.argmax(1)).mean()))

            self.obs, self.mask, r, d, info = self.env.step(a, shaping, cfg.gamma)
            self.registra_fim(info)
            rew_buf[t] = r[:, None]
            done_buf[t] = d.astype(np.float32)

            if info["trunc_idx"].size:      # fome é truncamento, não terminação
                _, _, v_f = self._frente(self.model,
                                         tf.convert_to_tensor(info["final_obs"]))
                v_f = v_f.numpy()                                 # (k, Z)
                for z in range(Z):
                    rew_buf[t, :, z] = self.bootstrap_truncados(
                        info, rew_buf[t, :, z], v_f[:, z], cfg.gamma)

            self.zeta = zeta_prox.astype(np.float32)
            fim = np.nonzero(d)[0]
            if fim.size:
                # a crença não atravessa a morte: o episódio seguinte começa sem saber
                # em que regime está, como o primeiro
                self.zeta[fim] = 1.0 / Z

            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        self.global_step += T * N
        self.episodes += len(scores)

        _, _, v_final = self._frente(self.model, tf.convert_to_tensor(self.obs))
        lote = {
            "obs": obs_buf, "mask": mask_buf, "act": act_buf, "zeta": zeta_buf,
            "alpha": alpha_buf, "val": val_buf, "pi_a": pia_buf, "pi_z": piz_buf,
            "rew": rew_buf, "done": done_buf,
            "v_final": v_final.numpy(), "zeta_final": self.zeta.copy(),
        }
        uso = zeta_buf.reshape(-1, Z).mean(0)
        stats = {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "shaping": shaping,
            "alpha_medio": float(alpha_buf.mean()),
            #: Entropia do **uso** das opções, em nats. `log Z` é uso equilibrado; zero é
            #: uma opção só — o colapso clássico dos métodos de opções.
            "opcao_uso_entropia": float(-(uso * np.log(uso + 1e-12)).sum()),
            "opcao_uso_max": float(np.log(Z)),
            #: Quanto as sub-políticas de fato diferem. Perto de zero, as opções existem
            #: no papel e não no comportamento.
            "opcao_divergencia": float(np.mean(divergencia)),
            #: Fração de passos em que a opção mais provável não mudou. É a medida direta
            #: de "as opções são temporalmente estendidas?" — que é a razão de existir do
            #: arcabouço. Perto de `1/Z` elas estão sendo re-sorteadas a cada passo.
            "opcao_persistencia": float(np.mean(persistencia)),
            "entropia_marginal": float(np.mean(ent_marginal)),
        }
        return lote, stats

    # ------------------------------------------------------------------ alvos
    def _alvos(self, lote):
        """GAE por opção, alvo de valor e `A^GOA`, tudo com os números da coleta.

        Uma vez por rollout, e não uma por época: `ζ`, `α` e as vantagens são funções da
        política **velha**, como no PPO. O que muda entre as épocas é só a razão.
        """
        cfg = self.cfg
        a_gae, _ = gae_de_opcoes(lote["rew"], lote["val"], lote["done"], lote["pi_z"],
                                 lote["v_final"], cfg.gamma, cfg.gae_lambda)
        conj = lote["pi_a"][:, :, :, None] * lote["pi_z"]
        goa = vantagem_de_opcao(a_gae, lote["zeta"], lote["zeta_final"], lote["done"],
                                conj, lote["alpha"])
        v_alvo = lote["val"] + a_gae
        return a_gae, goa, v_alvo, conj

    # ----------------------------------------------------------------- update
    @staticmethod
    @tf.function(reduce_retracing=True)
    def _train_step(model, optimizer, obs, mask, act, zeta, alpha, p_velho, goa, v_alvo,
                    clip_eps, vf_coef, ent_coef, ent_opcao_coef):
        """Um passo de gradiente sobre a política conjunta `p_Θ(a, z'|s, z)`."""
        mask3 = tf.expand_dims(mask, 1)
        with tf.GradientTape() as tape:
            la, lz, valor = model(obs, training=True)
            # a máscara vale no update também — o motivo é o mesmo do PPO: sem ela o
            # `log_prob` daqui não bate com o que gerou a ação e a razão vira lixo
            la = tf.where(mask3, la, tf.fill(tf.shape(la), MASK_NEG))
            pi_a = tf.nn.softmax(la)                                   # (B, Z, A)
            um = tf.expand_dims(tf.one_hot(act, N_ACTIONS), 1)
            pi_a_t = tf.reduce_sum(pi_a * um, axis=-1)                 # (B, Z)

            lz_a = tf.einsum("bzay,ba->bzy", lz, tf.one_hot(act, N_ACTIONS))
            pi_z = tf.nn.softmax(lz_a)                                 # (B, Z, Z')
            p_novo = tf.expand_dims(pi_a_t, -1) * pi_z

            # `w` é a responsabilidade do par `(z, z')`: a posteriori dado o histórico e a
            # ação. Ela soma 1 sobre os pares, então a perda é uma média ponderada de
            # perdas de PPO — e não uma soma de escala arbitrária.
            w = tf.expand_dims(zeta, -1) * p_velho / tf.reshape(alpha, (-1, 1, 1))
            razao = p_novo / tf.maximum(p_velho, 1e-8)
            vant = tf.expand_dims(goa, 1)                              # (B, 1, Z')
            pg1 = razao * vant
            pg2 = tf.clip_by_value(razao, 1.0 - clip_eps, 1.0 + clip_eps) * vant
            pg_loss = -tf.reduce_mean(tf.reduce_sum(w * tf.minimum(pg1, pg2),
                                                    axis=[1, 2]))

            # o crítico é regredido sob a crença: uma opção que o agente acha improvável
            # naquele estado não deve puxar o valor dela para o retorno que ele viu
            v_loss = 0.5 * tf.reduce_mean(
                tf.reduce_sum(zeta * tf.square(valor - v_alvo), axis=-1))

            # A entropia é a do **marginal** — a distribuição que o ambiente vê. É ela que
            # tem que ser comparável com o `ent` do PPO; a entropia média das
            # sub-políticas seria outro número, sistematicamente menor.
            marginal = tf.reduce_sum(tf.expand_dims(zeta, -1) * pi_a, axis=1)
            log_marg = tf.math.log(marginal + 1e-12)
            seguro = tf.where(mask, log_marg, tf.zeros_like(log_marg))
            entropia = -tf.reduce_mean(tf.reduce_sum(marginal * seguro, axis=-1))

            ent_z = -tf.reduce_mean(tf.reduce_sum(
                zeta * tf.reduce_sum(pi_z * tf.math.log(pi_z + 1e-12), axis=-1),
                axis=-1))

            perda = (pg_loss + vf_coef * v_loss - ent_coef * entropia
                     - ent_opcao_coef * ent_z)

        grads = tape.gradient(perda, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        log_razao = tf.math.log(tf.maximum(razao, 1e-8))
        # k3 sobre a política conjunta, ponderado pela responsabilidade: é a KL que
        # importa, porque é ela que o clipping tenta limitar
        kl = tf.reduce_mean(tf.reduce_sum(
            w * (razao - 1.0 - log_razao), axis=[1, 2]))
        clipfrac = tf.reduce_mean(tf.reduce_sum(
            w * tf.cast(tf.abs(razao - 1.0) > clip_eps, tf.float32), axis=[1, 2]))
        return pg_loss, v_loss, entropia, ent_z, kl, clipfrac

    def update(self, lote):
        cfg = self.cfg
        self.optimizer.learning_rate.assign(self.lr())
        ent = self.ent_coef()
        a_gae, goa, v_alvo, conj = self._alvos(lote)

        T, N, Z = cfg.rollout, cfg.num_envs, cfg.n_opcoes
        b, c = cfg.board_size, self.env.n_channels
        n = T * N
        plano = {
            "obs": lote["obs"].reshape(n, b, b, c),
            "mask": lote["mask"].reshape(n, N_ACTIONS),
            "act": lote["act"].reshape(n),
            "zeta": lote["zeta"].reshape(n, Z),
            "alpha": lote["alpha"].reshape(n),
            "p_velho": conj.reshape(n, Z, Z),
            "goa": _normaliza(goa).reshape(n, Z),
            "v_alvo": v_alvo.reshape(n, Z),
        }
        tensores = {k: tf.convert_to_tensor(v) for k, v in plano.items()}
        escalares = [tf.constant(v, tf.float32)
                     for v in (cfg.clip_eps, cfg.vf_coef, ent, cfg.ent_opcao_coef)]

        mb = max(1, n // cfg.minibatches)
        idx = np.arange(n)
        rng = np.random.default_rng(cfg.seed + self.iteration)
        logs = {"pg": [], "vf": [], "ent": [], "ent_opcao": [], "kl": [], "clipfrac": []}
        parar, epocas, atualizacoes = False, 0, 0

        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s in range(0, n, mb):
                sl = tf.convert_to_tensor(idx[s:s + mb])
                pg, vf, e, ez, kl, cf = self._train_step(
                    self.model, self.optimizer,
                    *[tf.gather(tensores[k], sl) for k in
                      ("obs", "mask", "act", "zeta", "alpha", "p_velho", "goa",
                       "v_alvo")],
                    *escalares,
                )
                logs["pg"].append(float(pg)); logs["vf"].append(float(vf))
                logs["ent"].append(float(e)); logs["ent_opcao"].append(float(ez))
                logs["kl"].append(float(kl)); logs["clipfrac"].append(float(cf))
                atualizacoes += 1
                if float(kl) > cfg.target_kl * 1.5:
                    parar = True
                    break
            epocas += 1
            if parar:
                break

        saida = {k: float(np.mean(v)) for k, v in logs.items()}
        # a variância explicada é lida sob a crença, como o crítico é treinado
        v_medio = np.einsum("tnz,tnz->tn", lote["zeta"], lote["val"]).ravel()
        alvo_medio = np.einsum("tnz,tnz->tn", lote["zeta"], v_alvo).ravel()
        saida.update({
            "ev": variancia_explicada(v_medio, alvo_medio),
            "lr": float(self.lr()), "ent_coef": ent,
            "epochs_done": epocas, "atualizacoes": int(atualizacoes),
            "goa_amplitude": float(goa.max(-1).mean() - goa.min(-1).mean()),
        })
        return saida

    # ------------------------------------------------------------------- passo
    def iterate(self):
        lote, stats = self.collect()
        stats.update(self.update(lote))
        return stats


def _normaliza(x):
    """Centra e escala a vantagem sobre o lote inteiro — a convenção do PPO daqui.

    Sobre o lote inteiro, e não por opção: normalizar por opção apagaria justamente a
    diferença entre as opções, que é o sinal que o `A^GOA` carrega.
    """
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)
