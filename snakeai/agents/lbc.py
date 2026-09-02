"""LBC — *Learnable Behavior Control* (Fan et al., ICLR 2023).

Todos os outros nove algoritmos deste repositório exploram por **regra fixa**. O ε do DQN
desce numa reta, o coeficiente de entropia do PPO desce noutra, o σ da `NoisyDense` encolhe
sozinho. Nenhuma dessas regras olha para o que aconteceu: o ε de 0,3 no passo 1 M é 0,3
porque o agendamento manda, e não porque explorar tanto ali esteja rendendo.

O LBC troca a regra por um **problema de otimização**. O comportamento — a distribuição que
de fato gera as ações — passa a ser escolhido dentro de um espaço parametrizado, e a escolha
é aprendida por um meta-controlador que observa o retorno de cada configuração. É a
generalização do Agent57, que escolhia *qual política da população usar*; o LBC escolhe
*uma mistura de todas elas*, e o espaço de comportamento deixa de ser limitado pelo tamanho
da população.

As três peças
-------------
1. **População de políticas.** `N` políticas indexadas por hiperparâmetros próprios — aqui,
   o fator de desconto γ_i. Uma política com γ baixo é míope e agressiva, uma com γ alto é
   paciente: comportamentos qualitativamente diferentes sobre a mesma rede.

2. **Mapeamento híbrido de comportamento** (`hybrid behavior mapping`, §3.1). O
   comportamento é uma **mistura de Boltzmann** sobre todas as políticas:

   ``μ_ψ(a|s) = Σ_i ω_i · softmax(τ_i · logits_i(s))``

   com `ψ = (τ_1..τ_N, ω_1..ω_N)`. É o coração do paper: onde métodos anteriores escolhiam
   *uma* política da população — o espaço de comportamento tinha `N` elementos —, aqui o
   espaço é contínuo e de dimensão `2N`. Os `τ` controlam a entropia política a política
   (τ→0 é uniforme, τ grande é guloso) e os `ω` decidem quanto cada política contribui. Com
   `ω` one-hot recupera-se a seleção de política única do Agent57, que é o caso degenerado.

3. **Seleção por bandit.** Ψ é discretizado em `K` regiões, cada região é um braço de um
   UCB não-estacionário (`snakeai/bandit.py`), e o retorno **não descontado** do episódio é
   a recompensa do braço. Cada ambiente sorteia uma região no início de cada episódio e
   um `ψ` dentro dela — a região é uma região, não um ponto.

Por que precisa de V-trace
--------------------------
A mistura `μ` não é nenhuma das políticas que estão sendo treinadas. Os dados são
**off-policy por construção**, e não por acidente: quanto mais o meta-controlador explora,
mais longe `μ` fica de `π_i`. Treinar `π_i` com esses dados sem corrigir é o erro que
transforma o LBC num A2C com ruído caro.

`vtrace()` (Espeholt et al., 2018) é a correção: pesos de importância `π_i/μ` truncados em
`ρ̄` para o alvo de valor e em `c̄` para a propagação temporal. O truncamento é o que torna o
estimador seguro para dados arbitrariamente distantes — a correção nunca amplifica, só
encolhe. É a mesma ideia do Retrace(λ) do ACER, com a diferença de que aqui o crítico é
`V(s)` e não `Q(s,·)`.

O truncamento também é o que permite **múltiplas épocas** sobre o mesmo rollout, que é como
o LBC gasta o orçamento de gradiente que a §2.1 da revisão mostrou valer ~18 pontos: `μ`
está gravado, então a correção continua válida à medida que `π_i` se afasta. No PPO isso
custaria clipping; aqui sai do próprio estimador.

Desvios declarados em relação ao paper
--------------------------------------
Quatro. Os três primeiros são o contrato deste repositório; o quarto é uma correção. Todos
estão detalhados em `docs/LBC.md`:

* **tronco compartilhado** entre as `N` políticas, em vez de `N` redes independentes;
* **H reduzido a γ**, sem o eixo de *reward shaping* por política;
* **um bandit**, em vez do conjunto de bandits com `c` diferentes do §4.2;
* **região de confiança do PPO** em volta do gradiente do IMPALA — clip da razão, clip do
  valor e parada por KL. Ver `LBCConfig.clip_eps`.

O que a primeira execução ensinou
---------------------------------
A `seed0` com os padrões antigos terminou em 0,57 ponto contra 81,5 do PPO. Não foi uma
questão de ajuste fino: foram três defeitos que se reforçam, e vale registrá-los porque
cada um deles é invisível na curva de score e evidente no diagnóstico.

1. **O `τ` não tinha autoridade sobre o comportamento.** Ele multiplicava logits livres, que
   crescem sem limite; com `‖logits‖ ~ 30`, a faixa inteira `[0,25, 4]` produz `argmax`. O
   espaço de comportamento degenerou num ponto, `μ` virou `π`, `razao_media` ficou em
   1,0000 e o V-trace deixou de corrigir qualquer coisa. Sinal: `entropia_comportamento`
   em 3e-4. Correção: padronizar os logits por estado antes de escalar.

2. **Não havia região de confiança nenhuma sobre 128 passos de gradiente por rollout.** A
   entropia da política alvo caiu para 5e-9 no passo 540 mil e `pg` para 1e-11 — softmax
   saturada é ponto fixo absorvente, porque ali o gradiente da entropia também é zero. A
   execução ficou 2,3 M de passos parada. Correção: surrogate clipado + parada por KL +
   vantagem normalizada, e um coeficiente de entropia realimentado como piso.

3. **O bandit decidia sobre ruído.** A normalização min–max entre braços estica a diferença
   entre o melhor e o pior para `[0, 1]` mesmo quando essa diferença é menor que o erro
   amostral; com temperatura 0,1 isso vira `argmax`. `mab_p_top` chegou a 0,999 no passo
   800 mil, com todos os 512 ambientes no mesmo braço. Correção: piso de ruído no
   denominador, mínimo de puxadas por braço e piso uniforme na distribuição de seleção.

O diagnóstico completo, com as curvas, está em `docs/LBC.md`.

O que o LBC responde na arena
-----------------------------
A comparação que interessa é **LBC × PPO**: mesma rede, mesmo ambiente, mesmo orçamento de
ambiente, mesmo γ na política avaliada. A diferença entre as curvas é o preço/prêmio de
trocar exploração agendada por exploração **selecionada** — que é exatamente a pergunta que
os agendamentos lineares deste repositório nunca puderam responder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf

from ..bandit import BanditUCB
from ..env.vec_snake import N_ACTIONS, VecSnake
from ..eval import MASK_NEG
from ..nets import build_actor_critic_populacao
from ..otimizadores import cria_otimizador
from .base import AgentBase, BaseConfig
from .ppo import variancia_explicada

__all__ = ["LBCConfig", "LBC", "MisturaBoltzmann", "vtrace"]


# --------------------------------------------------------------------- V-trace
def vtrace(rew, valores, done, ultimo_v, rho, c, gamma):
    """Alvo V-trace e vantagem para o gradiente de política (Espeholt et al., 2018).

    ``v_t = V(s_t) + Σ_{k≥t} γ^{k-t} (Π_{j<k} c_j) ρ_k δ_k``, com
    ``δ_k = r_k + γV(s_{k+1}) − V(s_k)``, calculado pela recursão para trás
    ``acc_t = ρ_t δ_t + γ c_t acc_{t+1}``.

    Os dois truncamentos fazem coisas diferentes, e confundi-los é o bug clássico:

    * `ρ̄` (em `rho`) limita **para que ponto fixo** o crítico converge. Com `ρ̄ = 1`, `v`
      é o valor da própria política de comportamento; com `ρ̄ = ∞`, o da política alvo.
    * `c̄` (em `c`) limita **a variância da propagação** para trás no tempo. Ele não muda o
      ponto fixo, só a velocidade com que a informação atravessa o rollout.

    Formas: `rew`, `valores`, `done`, `rho`, `c` são `(T, N)`; `ultimo_v` é `(N,)`. O
    bootstrap do truncamento por fome já foi somado a `rew` na coleta, então aqui todo
    `done` pode ser tratado como terminal — a mesma convenção do `compute_gae` do PPO, e
    pelo mesmo motivo: sem ela o valor do episódio seguinte vaza para o anterior.

    Devolve `(vs, adv)`, ambos `(T, N)`. `adv = ρ_t (r_t + γ v_{t+1} − V(s_t))` — a
    vantagem que o gradiente de política consome, com o `ρ` **já embutido**.
    """
    T, N = rew.shape
    vs = np.zeros((T, N), dtype=np.float32)
    acc = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        v_prox = ultimo_v if t == T - 1 else valores[t + 1]
        continua = 1.0 - done[t]
        delta = rho[t] * (rew[t] + gamma * continua * v_prox - valores[t])
        acc = delta + gamma * continua * c[t] * acc
        vs[t] = valores[t] + acc

    # `v_{t+1}`, com o bootstrap no fim do segmento. Note que é `vs`, não `valores`: a
    # vantagem tem que olhar para o alvo corrigido do passo seguinte, senão o gradiente
    # de política usa uma baseline e um alvo de fontes diferentes.
    vs_prox = np.concatenate([vs[1:], ultimo_v[None].astype(np.float32)], axis=0)
    continua = 1.0 - done
    adv = rho * (rew + gamma * continua * vs_prox - valores)
    return vs, adv.astype(np.float32)


# ------------------------------------------------------- espaço de comportamento
class MisturaBoltzmann:
    """O espaço de comportamento `M_{H,Ψ}` do §4.1, e a discretização de Ψ em braços.

    Ψ tem dimensão `2N` e é contínuo; o bandit precisa de um número finito de braços. O
    paper resolve discretizando Ψ em `K` regiões — e a palavra *região* é o ponto: o braço
    não é um `ψ`, é um conjunto de `ψ`, e amostrar o braço amostra um `ψ` de dentro dele.
    Sem isso o espaço de comportamento voltaria a ser finito, que é justamente a limitação
    que o LBC existe para remover.

    A discretização aqui é o produto de dois eixos:

    * **faixa de τ** — `n_faixas` intervalos log-uniformes em `[tau_min, tau_max]`. Dentro
      da faixa, cada política sorteia o seu `τ_i` **independentemente**, então a região
      contém misturas com temperaturas diferentes entre políticas.
    * **padrão de ω** — `N` padrões concentrados numa política (o caso Agent57: use quase
      só a política `i`) mais um padrão uniforme (use todas). O `ω` sai de uma Dirichlet
      centrada no padrão, nunca do padrão exato.

    São `K = n_faixas × (N + 1)` braços — 16 na configuração padrão. Um produto cartesiano
    completo sobre os `2N` eixos daria `n_faixas^N × ...`, que o bandit não conseguiria
    estimar dentro de 5 M passos: com mais braços que episódios por janela, todo braço fica
    com valor `NaN` e o UCB vira sorteio uniforme caro.
    """

    def __init__(self, n_politicas, tau_min=0.25, tau_max=4.0, n_faixas=4,
                 concentracao=9.0, padronizar=True, rng=None):
        if tau_min <= 0 or tau_max <= tau_min:
            raise ValueError("é preciso 0 < tau_min < tau_max")
        self.n_politicas = int(n_politicas)
        self.n_faixas = int(n_faixas)
        self.concentracao = float(concentracao)
        self.padronizar = bool(padronizar)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        bordas = np.geomspace(tau_min, tau_max, self.n_faixas + 1)
        self.faixas = np.stack([bordas[:-1], bordas[1:]], axis=1)     # (n_faixas, 2)

        # Padrões de ω: um por política (concentrado) + o uniforme. Com N = 1 os dois
        # coincidem, e a população degenerada tem um padrão só.
        padroes = np.eye(self.n_politicas, dtype=np.float64)
        if self.n_politicas > 1:
            padroes = np.vstack([padroes,
                                 np.full(self.n_politicas, 1.0 / self.n_politicas)])
        self.padroes = padroes

    @property
    def n_bracos(self):
        return self.n_faixas * len(self.padroes)

    def _decompoe(self, braco):
        b = np.asarray(braco, dtype=np.int64)
        return b // len(self.padroes), b % len(self.padroes)

    def amostrar(self, bracos):
        """Sorteia um `ψ = (τ, ω)` dentro de cada região. Devolve `(tau, omega)` `(M, N)`."""
        bracos = np.atleast_1d(np.asarray(bracos, dtype=np.int64))
        m, n = bracos.size, self.n_politicas
        i_faixa, i_padrao = self._decompoe(bracos)

        lo, hi = self.faixas[i_faixa, 0], self.faixas[i_faixa, 1]
        # log-uniforme: τ é um fator multiplicativo sobre os logits, e a diferença entre
        # 0,25 e 0,5 é do mesmo tamanho perceptual que entre 2 e 4
        u = self.rng.random((m, n))
        tau = np.exp(np.log(lo)[:, None] + u * (np.log(hi) - np.log(lo))[:, None])

        alpha = 1.0 + self.concentracao * self.padroes[i_padrao]
        omega = np.stack([self.rng.dirichlet(a) for a in alpha])
        return tau.astype(np.float32), omega.astype(np.float32)

    def descricao(self, braco):
        f, p = self._decompoe(int(braco))
        lo, hi = self.faixas[f]
        alvo = "uniforme" if p >= self.n_politicas else f"π{p}"
        return f"τ∈[{lo:.2f}, {hi:.2f}] · ω≈{alvo}"

    def comportamento(self, logits, mask, tau, omega):
        """`μ_ψ(a|s) = Σ_i ω_i softmax(τ_i · ẑ_i)`, mascarado. `(M, ações)`.

        **Por que `ẑ` e não `logits`.** No paper, `Φ_h = A_h = Q_h − V_h`: o que entra na
        softmax é uma *vantagem*, centrada em zero e presa à escala da recompensa. Aqui a
        rede é um ator-crítico comum e o que sai da cabeça é um logit livre — um parâmetro
        que não tem escala nenhuma e que **cresce sem limite** enquanto a política aprende
        a preferir uma ação. Multiplicar esse número por `τ ∈ [0,25, 4]` não controla
        entropia: com `‖logits‖ ~ 30`, até `τ = 0,25` dá 7,5, que já é `argmax`.

        O efeito é a morte silenciosa do algoritmo: o espaço de comportamento `M_{H,Ψ}`
        degenera num ponto só — a política gulosa —, `μ` vira `π`, a razão `π/μ` vira 1, o
        V-trace deixa de corrigir coisa alguma e o LBC passa a ser um A2C caro com um
        bandit decorativo escolhendo entre dezesseis cópias do mesmo comportamento. Na
        execução `seed0` isso aparece como `entropia_comportamento = 3e-4` e
        `razao_media = 1,0000` do passo 540 mil até o fim.

        A correção é padronizar os logits **por estado, sobre as ações válidas**, antes de
        escalar. Com `ẑ` de média 0 e desvio 1, o `τ` recupera a autoridade que o paper lhe
        dá: `τ = 0,25` deixa a mistura a 1,07 nat de entropia (o máximo com três ações é
        1,0986) e `τ = 4` a deixa em 0,008. A faixa inteira do espaço de comportamento
        volta a existir, e volta a existir **de forma independente do que a rede fez com a
        escala dos próprios logits**, que é a propriedade que faltava.

        A máscara é aplicada **depois** de multiplicar por `τ`, e não antes. Mascarar
        antes multiplicaria o `MASK_NEG` por `τ`: com `τ = 4` o valor sai de −1e9 para
        −4e9, que ainda funciona, mas com `τ` pequeno ele encolhe na direção do zero e uma
        ação letal volta a ter probabilidade não desprezível. É o tipo de bug que não
        levanta exceção — a cobra só passa a bater na parede de vez em quando.
        """
        z = np.asarray(logits, dtype=np.float32)
        if self.padronizar:
            val = mask[:, None, :].astype(np.float32)                  # (M, 1, A)
            n_val = np.maximum(val.sum(-1, keepdims=True), 1.0)        # (M, 1, 1)
            media = (z * val).sum(-1, keepdims=True) / n_val           # (M, P, 1)
            var = (((z - media) * val) ** 2).sum(-1, keepdims=True) / n_val
            z = (z - media) / np.sqrt(var + 1e-6)
        z = z * tau[:, :, None]
        z = np.where(mask[:, None, :], z, MASK_NEG)
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        return np.einsum("mp,mpa->ma", omega, p)


# ------------------------------------------------------------------- configuração
@dataclass
class LBCConfig(BaseConfig):
    num_envs: int = 512
    rollout: int = 32

    #: Tamanho da população. `1` é a ablação "reduzir H" da Fig. 5 do paper: uma política
    #: só, com o comportamento vindo apenas de `ψ`.
    n_politicas: int = 3

    #: Um γ por política — é este o `H` deste repositório. Míope, o do contrato, paciente.
    #: O eixo de *reward shaping* por política do paper não existe aqui: o shaping é
    #: aplicado dentro do `VecSnake`, que devolve **uma** recompensa, e replicá-lo no
    #: agente significaria reimplementar o potencial do ambiente. Ver `docs/LBC.md`.
    gammas: tuple = (0.99, 0.995, 0.999)

    #: Qual política é avaliada e salva. O padrão aponta para a de γ = 0,995 — **o mesmo
    #: do PPO, do A2C e do ACKTR**. É o que faz a diferença entre as curvas medir controle
    #: de comportamento, e não fator de desconto.
    indice_alvo: int = 1

    # ---------------------------------------------------------------- V-trace
    #: `ρ̄` — decide o ponto fixo do crítico. 1,0 é o valor canônico do IMPALA.
    rho_barra: float = 1.0
    #: `c̄` — decide só a variância da propagação temporal.
    c_barra: float = 1.0

    # -------------------------------------------------- espaço de comportamento
    tau_min: float = 0.25
    tau_max: float = 4.0
    n_faixas_tau: int = 4
    #: Concentração da Dirichlet em torno do padrão de ω. Alta demais e a região vira um
    #: ponto; baixa demais e todos os braços amostram a mesma coisa.
    concentracao_omega: float = 9.0

    # ------------------------------------------------------------------- MAB
    #: Padroniza os logits por estado antes de escalar por `τ`. **Ligado é o correto** —
    #: ver `MisturaBoltzmann.comportamento`. Desligar reproduz a degeneração do espaço de
    #: comportamento que matou a execução `seed0`, e serve como ablação.
    logits_padronizados: bool = True

    #: `"ucb"` é o algoritmo. `"aleatoria"` é a ablação de seleção da Fig. 5 — o mesmo
    #: espaço de comportamento, escolhido no sorteio. Se as duas curvas coincidirem, a
    #: parte *learnable* do LBC não fez nada neste domínio, e isso é um resultado.
    selecao: str = "ucb"
    ucb_c: float = 1.0
    #: Dureza da softmax que transforma o score do UCB na distribuição de seleção. Com os
    #: valores normalizados em `[0, 1]`, sem ela o bandit não conseguiria concentrar — ver
    #: `snakeai/bandit.py`. 0,25 e não 0,1: com 0,1 o bandit concentrava 99,9% da massa num
    #: braço antes de ter evidência para isso.
    ucb_temperatura: float = 0.25
    #: Retornos por braço na janela do bandit. 64 episódios por braço, com 16 braços, é
    #: cerca de um quarto do que uma iteração de 512 ambientes produz — o bandit enxerga
    #: alguns milhares de passos para trás, não a execução inteira.
    janela_mab: int = 64
    #: Episódios mínimos na janela antes de o braço ter valor estimado. Abaixo disso ele
    #: entra como não-visitado (otimista), e não com a média de duas amostras.
    mab_min_puxadas: int = 8
    #: Massa reservada ao uniforme na distribuição de seleção. É o que garante que **todo**
    #: braço continue sendo medido: com 512 ambientes escolhendo ao mesmo tempo, sem piso o
    #: bandit consegue colocar todos no mesmo braço e deixar de receber dado sobre os
    #: outros quinze — que é o oposto de manter um espaço de comportamento diverso.
    mab_piso_uniforme: float = 0.1

    # -------------------------------------------------------------- otimização
    #: **1e-4 e não 3e-4 como no PPO, e isto é sobre o orçamento de gradiente, não sobre
    #: velocidade de aprendizado.** Ver `docs/LBC.md` §2.11.
    #:
    #: O LBC move a política mais por passo de gradiente que o PPO, por duas razões que se
    #: multiplicam: a perda é a **soma** sobre três políticas e o tronco é compartilhado,
    #: então ele recebe ~3× o gradiente que o tronco do PPO recebe; e os dados são
    #: off-policy, com a vantagem pesada por `ρ` (metade das amostras truncadas), o que a
    #: deixa de cauda pesada — depois de normalizada, são as caudas que mandam no passo.
    #:
    #: Com `lr = 3e-4` isso não aparece como instabilidade: aparece como **fome de
    #: gradiente**. A parada por KL, que deveria ser freio de emergência, dispara no
    #: minilote ~8 de 128, e a execução termina com 3.524 atualizações contra 38.374 do
    #: PPO — 9% do orçamento que o contrato promete a todos (`docs/ORCAMENTO_DE_GRADIENTE.md`).
    #: A curva sobe até o último passo porque o treino simplesmente não acabou.
    #:
    #: A resposta ao `lr` é um penhasco, medido em bancada (mediana de atualizações por
    #: iteração, de 128): `3e-4 → 15`, `1.5e-4 → 29`, `1e-4 → 125`. Em 1e-4 o KL medido cai
    #: para 0,0143 — **abaixo** do teto de 0,03 e no mesmo patamar do PPO (0,0125) —, o que
    #: quer dizer que o freio deixou de ser acionado em vez de ter sido afrouxado. Passo
    #: menor, orçamento inteiro, mesma segurança.
    lr_start: float = 1e-4
    lr_end: float = 3e-5
    optimizer: str = "adam"
    #: 1,0 e não 0,5 como no PPO **porque a perda é a soma sobre as políticas**. Com três
    #: cabeças, a norma do gradiente é ~√3 vezes a de uma; manter o teto do PPO faria o
    #: clip morder em toda iteração e o passo do LBC virar "direção normalizada, tamanho
    #: fixo" — que é um otimizador diferente do que o PPO usa, e a comparação deixaria de
    #: medir controle de comportamento.
    max_grad_norm: float = 1.0
    vf_coef: float = 0.5
    epochs: int = 4
    minibatches: int = 32

    # ------------------------------------------------------- região de confiança
    #: **Quarto desvio declarado** (ver `docs/LBC.md`). O IMPALA faz um passe único sobre
    #: cada rollout; este repositório dá a todo agente o mesmo orçamento de gradiente
    #: (`docs/ORCAMENTO_DE_GRADIENTE.md`), o que aqui significa 4 épocas × 32 minilotes =
    #: 128 passos sobre o mesmo lote. O V-trace autoriza reusar o lote — `μ` está gravado,
    #: então o *alvo de valor* continua correto a cada época. O que ele **não** faz é
    #: limitar o quanto a política anda: o gradiente `−logπ·Â` não tem região de confiança
    #: nenhuma, e aplicá-lo 128 vezes satura a softmax. Foi exatamente isso que aconteceu:
    #: `ent` caiu de 0,95 para 5e-9 no passo 540 mil e `pg` foi para 1e-11 — ponto fixo
    #: absorvente, porque no regime saturado o gradiente da entropia também é zero. A
    #: execução ficou 2,3 M de passos morta antes de qualquer coisa voltar a acontecer.
    #:
    #: A correção é o surrogate clipado do PPO em volta do gradiente do IMPALA. No primeiro
    #: minilote de cada atualização a razão é exatamente 1 e o clip **não muda nada** — o
    #: gradiente é o do IMPALA, letra por letra. Ele só age depois, limitando o quanto a
    #: política pode se afastar do estado em que o lote foi coletado. `clip_eps <= 0`
    #: desliga e reproduz o comportamento antigo.
    clip_eps: float = 0.2
    #: Clip do valor, em unidades absolutas, como no PPO. **Desligado (`0`) por padrão
    #: aqui**, e isso foi medido: com `vf_clip = 0,2` a variância explicada do crítico do
    #: LBC fica em 0,30, e sem ele sobe para 0,86 no mesmo número de iterações. É o mesmo
    #: problema que o `PPO` documenta na §2 do seu módulo — um teto absoluto por
    #: atualização impede o crítico de alcançar a escala do retorno —, só que aqui ele
    #: morde muito mais: o alvo do PPO é um retorno GAE já suavizado, enquanto o do LBC é o
    #: `vs` do V-trace, que carrega a escala crua do score (0 a 97). A região de confiança
    #: que interessa é a da **política**; travar o crítico junto só atrasa a vantagem.
    vf_clip: float = 0.0
    #: Parada antecipada por KL, idêntica à do PPO (`target_kl * 1.5`). É a segunda barreira
    #: e a que pega o caso em que o clip sozinho não segura.
    target_kl: float = 0.03
    #: Normaliza a vantagem **por política**, dentro do minilote. O PPO deste repositório
    #: já faz isso; o LBC não fazia, e a vantagem do V-trace tem escala que muda por ordens
    #: de grandeza durante o treino (o shaping decai a zero em 25% do orçamento, o score
    #: cresce de 0 a 80). Sem normalizar, o passo efetivo da política é imprevisível — e é
    #: o co-autor do colapso de entropia.
    normalizar_vantagem: bool = True

    #: **Não agendado, e de propósito.** Nos outros agentes daqui a entropia decai numa
    #: reta; aqui quem controla a entropia do *comportamento* é o `τ` escolhido pelo
    #: bandit, e um agendamento por cima faria o mesmo trabalho duas vezes e em desacordo.
    #: Mas constante também não serve: 0,01 fixo não impediu a política alvo de saturar em
    #: 5e-9, e uma vez saturada nem o bônus de entropia a tira de lá (o gradiente da
    #: entropia numa softmax saturada é ~0 — o estado é absorvente, não lento).
    #:
    #: A saída é **realimentar**: o coeficiente sobe quando a entropia medida está abaixo
    #: de `ent_alvo` e desce quando está acima. É o mesmo princípio do `α` automático do
    #: SAC — o alvo é a entropia, o coeficiente é só o preço que se paga por ela. Assim o
    #: agendamento continua não existindo, e o piso passa a existir.
    ent_coef: float = 0.02
    #: Entropia mínima da política alvo, em nats. `None` volta ao coeficiente fixo. 0,15 é
    #: onde o PPO deste repositório termina por conta própria (0,13–0,14) — o alvo não
    #: força exploração extra, só proíbe o colapso.
    ent_alvo: float = 0.15
    ent_coef_min: float = 1e-4
    #: Teto alto de propósito: o coeficiente é um **preço**, não um peso escolhido a dedo.
    #: Se a entropia estiver abaixo do alvo com o teto batido, o controlador perdeu a
    #: autoridade e o piso deixa de existir — que é o defeito que ele veio consertar.
    ent_coef_max: float = 0.15
    #: Fator multiplicativo por iteração. 1,25 leva ~10 iterações para dobrar: rápido o
    #: bastante para reagir antes da saturação, lento o bastante para não oscilar.
    ent_ajuste: float = 1.25

    #: Shaping potencial, idêntico ao do PPO — é parte do ambiente que todos veem.
    shaping_start: float = 0.5
    shaping_frac: float = 0.25

    # ------------------------------------------------- o eixo `RS` de `H` (§5.1.2)
    #: **O segundo eixo de `H`, o que faltava.** No paper `h_i = (γ_i, RS_i)`, onde `RS` é
    #: um método de *reward shaping* por política (Badia et al., 2020a) — no Agent57 são os
    #: pesos `β_i` da recompensa intrínseca, pareados com os `γ_i` à moda do NGU: quem
    #: explora tem `β` alto e `γ` baixo, quem explora o que já sabe tem `β` baixo e `γ` alto.
    #: A §2.2 tinha reduzido `H` a γ sozinho, e a §2.12 mediu o preço disso: as três
    #: políticas otimizavam o **mesmo** objetivo, então duas delas não eram "míope
    #: competente" e "paciente competente" — eram simplesmente piores.
    #:
    #: Aqui o análogo de `RS` é o coeficiente do shaping potencial do `VecSnake`,
    #: `Φ = −dist(cabeça, comida)/b`, dado **por política**. Com ele, cada cabeça passa a
    #: maximizar um retorno diferente e legítimo:
    #:
    #: * shaping alto = guiada pelo gradiente local, colhe rápido, horizonte curto;
    #: * shaping zero = cega a esse gradiente, só a recompensa esparsa, precisa aprender
    #:   jogo de corpo inteiro para pontuar.
    #:
    #: `None` mantém o comportamento antigo — um shaping só, o do agendamento, igual para
    #: todas. Com uma tupla, o `VecSnake` é chamado com `shaping_coef = 0` e o agente monta
    #: a recompensa de cada política a partir de `phi_old`/`phi_new` do `info`, usando o
    #: **γ da própria política** (o potencial só é invariante quando o γ bate — no caminho
    #: antigo todas usam o γ da avaliada, que é uma imprecisão herdada).
    shapings: tuple = None

    #: Fração do orçamento em que o shaping de cada política chega a zero. `None` usa
    #: `shaping_frac` para todas. Um valor `<= 0` significa **constante, nunca decai** —
    #: que é como o `β` do NGU se comporta.
    #:
    #: O pareamento sugerido é `shapings=(1.0, 0.5, 0.0)` com `shaping_fracs=(1.0, 0.25, 0.0)`:
    #: a míope fica guiada a execução inteira, a **avaliada é bit-idêntica ao PPO**
    #: (0,5 zerando em 25%), e a paciente nunca vê o gradiente local. É o que faz a
    #: diversidade **persistir** — com um agendamento só, as três voltam a ser a mesma
    #: coisa depois de 25% do orçamento, que é exatamente o defeito.
    shaping_fracs: tuple = None

    canal_fome: bool = False

    def __post_init__(self):
        super().__post_init__()
        if len(self.gammas) != self.n_politicas:
            raise ValueError(
                f"n_politicas={self.n_politicas} mas gammas tem {len(self.gammas)} "
                "valores. Os dois descrevem a mesma população — passe os dois juntos, "
                "p.ex. LBCConfig(n_politicas=1, gammas=(0.995,), indice_alvo=0)."
            )
        if not -self.n_politicas <= self.indice_alvo < self.n_politicas:
            raise ValueError(
                f"indice_alvo={self.indice_alvo} fora da população de "
                f"{self.n_politicas} políticas")
        if self.selecao not in ("ucb", "aleatoria"):
            raise ValueError(f"selecao desconhecida: {self.selecao!r}")
        if self.ent_alvo is not None and not (
                0.0 < self.ent_coef_min <= self.ent_coef_max):
            raise ValueError("é preciso 0 < ent_coef_min <= ent_coef_max")
        for nome in ("shapings", "shaping_fracs"):
            v = getattr(self, nome)
            if v is not None and len(v) != self.n_politicas:
                raise ValueError(
                    f"{nome} tem {len(v)} valores mas n_politicas={self.n_politicas}. "
                    "Os dois descrevem a mesma população — passe os dois juntos.")
        if self.shaping_fracs is not None and self.shapings is None:
            raise ValueError(
                "shaping_fracs sem shapings não faz nada: o agendamento por política só "
                "existe quando há um coeficiente por política.")
        if self.canal_fome and self.comparable:
            raise ValueError(
                "canal_fome=True muda a observação de 5 para 6 canais e portanto a "
                "entrada da rede. Marque comparable=False e escreva o caveat.")

    @property
    def batch_size(self):
        return self.num_envs * self.rollout

    @property
    def gamma(self):
        """O γ da política avaliada. É ele que o shaping do ambiente usa."""
        return float(self.gammas[self.indice_alvo])


# ------------------------------------------------------------------------ agente
class LBC(AgentBase):
    """LBC-BM: mistura de Boltzmann como comportamento, bandit como meta-controlador."""

    algo = "lbc"

    def __init__(self, cfg: LBCConfig = None, model=None, variant=None):
        cfg = cfg or LBCConfig()
        super().__init__(cfg, variant=variant or self._variante(cfg))
        keras.utils.set_random_seed(cfg.seed)

        self.env = VecSnake(cfg.num_envs, cfg.board_size,
                            rng=np.random.default_rng(cfg.seed),
                            canal_fome=cfg.canal_fome)
        self.model = model or build_actor_critic_populacao(
            cfg.board_size, cfg.net, n_politicas=cfg.n_politicas,
            canais=self.env.n_channels)
        self.optimizer = self._novo_otimizador()
        self.obs, self.mask = self.env.reset()

        #: Índice absoluto da política avaliada — `indice_alvo` aceita negativo.
        self.indice_alvo = int(range(cfg.n_politicas)[cfg.indice_alvo])
        self.gammas = np.asarray(cfg.gammas, dtype=np.float64)

        self.espaco = MisturaBoltzmann(
            cfg.n_politicas, tau_min=cfg.tau_min, tau_max=cfg.tau_max,
            n_faixas=cfg.n_faixas_tau, concentracao=cfg.concentracao_omega,
            padronizar=cfg.logits_padronizados,
            rng=np.random.default_rng(cfg.seed + 3))
        self.mab = BanditUCB(self.espaco.n_bracos, c=cfg.ucb_c, janela=cfg.janela_mab,
                             temperatura=cfg.ucb_temperatura,
                             min_puxadas=cfg.mab_min_puxadas,
                             piso_uniforme=cfg.mab_piso_uniforme,
                             rng=np.random.default_rng(cfg.seed + 4))
        self.rng = np.random.default_rng(cfg.seed + 1)

        #: Coeficiente de entropia **realimentado** — estado do agente, não agendamento.
        #: Ver `LBCConfig.ent_alvo`.
        self._ent_coef = float(cfg.ent_coef)

        #: Um braço e um `ψ` **por ambiente**, trocados quando o episódio daquele ambiente
        #: acaba. Trocar por iteração em vez de por episódio quebraria a atribuição de
        #: crédito do bandit: um episódio de Snake atravessa vários rollouts, e o retorno
        #: seria creditado a um braço que só esteve no ar no fim dele.
        self.braco = np.zeros(cfg.num_envs, dtype=np.int64)
        self.tau = np.ones((cfg.num_envs, cfg.n_politicas), dtype=np.float32)
        self.omega = np.full((cfg.num_envs, cfg.n_politicas),
                             1.0 / cfg.n_politicas, dtype=np.float32)
        self._novo_comportamento(np.arange(cfg.num_envs))

    @staticmethod
    def _variante(cfg):
        """O que **desvia** do LBC oficial entra no nome da variante.

        `load_all` agrupa por `(algo, variant, seed)`: sem isto, a execução com seleção
        aleatória — que é uma ablação, não o algoritmo — dividiria identidade com a
        oficial e as duas virariam uma curva só na arena.
        """
        marcas = []
        if cfg.selecao != "ucb":
            marcas.append("selecao_" + cfg.selecao)
        if cfg.n_politicas != type(cfg).n_politicas:
            marcas.append(f"pop{cfg.n_politicas}")
        if cfg.shapings is not None:
            marcas.append("H_shaping")
        if not cfg.logits_padronizados:
            marcas.append("logits_crus")
        if cfg.clip_eps <= 0:
            marcas.append("sem_clip")
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

    def shaping(self):
        f = self.frac()
        return max(0.0, self.cfg.shaping_start * (1.0 - f / self.cfg.shaping_frac))

    def shaping_por_politica(self):
        """Coeficiente de shaping de **cada** política agora, `(P,)`. `None` = eixo desligado.

        É o `RS_i` de `h_i = (γ_i, RS_i)` — ver `LBCConfig.shapings`. `frac <= 0` significa
        constante: o `β` do NGU não decai, e é isso que faz a diversidade sobreviver ao
        ponto em que o agendamento do contrato zera.
        """
        cfg = self.cfg
        if cfg.shapings is None:
            return None
        f = self.frac()
        inicio = np.asarray(cfg.shapings, dtype=np.float32)
        fracs = (np.full(cfg.n_politicas, cfg.shaping_frac, dtype=np.float32)
                 if cfg.shaping_fracs is None
                 else np.asarray(cfg.shaping_fracs, dtype=np.float32))
        saida = np.empty(cfg.n_politicas, dtype=np.float32)
        for i in range(cfg.n_politicas):
            fr = float(fracs[i])
            saida[i] = (inicio[i] if fr <= 0.0
                        else max(0.0, float(inicio[i]) * (1.0 - f / fr)))
        return saida

    # ------------------------------------------------------------- comportamento
    def _novo_comportamento(self, idx):
        """Sorteia braço e `ψ` para os ambientes em `idx` — chamado no fim do episódio."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        if self.cfg.selecao == "ucb":
            bracos = self.mab.amostrar(idx.size)
        else:
            bracos = self.rng.integers(0, self.mab.n, size=idx.size)
        tau, omega = self.espaco.amostrar(bracos)
        self.braco[idx] = bracos
        self.tau[idx] = tau
        self.omega[idx] = omega

    @staticmethod
    @tf.function(reduce_retracing=True)
    def _frente(model, obs):
        """Logits **crus** `(B, P, A)` e valores `(B, P)`.

        Crus porque o comportamento mascara depois de escalar por `τ` — ver
        `MisturaBoltzmann.comportamento`.
        """
        logits, valor = model(obs, training=False)
        return logits, valor

    def politica_do_modelo(self, modelo):
        """A política avaliada é **uma** da população: a de índice `indice_alvo`.

        O `keras_policy` genérico não serve aqui. Ele pega `saida[0]` e mascara, o que
        para este modelo seria o tensor `(B, P, A)` inteiro — a avaliação receberia uma
        forma errada, ou pior, a política da primeira cabeça independentemente de qual foi
        declarada como alvo. A escolha da cabeça é uma decisão do algoritmo e precisa
        aparecer no código, não sair de um índice acidental.
        """
        i = self.indice_alvo

        @tf.function(reduce_retracing=True)
        def frente(obs, mask):
            logits, _ = modelo(obs, training=False)
            l = logits[:, i, :]
            return tf.where(mask, l, tf.fill(tf.shape(l), MASK_NEG))

        def fn(obs, mask):
            return frente(tf.convert_to_tensor(obs),
                          tf.convert_to_tensor(mask)).numpy()
        return fn

    def politica(self):
        return self.politica_do_modelo(self.model)

    # ----------------------------------------------------------------- rollout
    def collect(self):
        cfg = self.cfg
        T, N, P = cfg.rollout, cfg.num_envs, cfg.n_politicas
        b, c = cfg.board_size, self.env.n_channels

        obs_buf = np.empty((T, N, b, b, c), dtype=np.float32)
        mask_buf = np.empty((T, N, N_ACTIONS), dtype=bool)
        act_buf = np.empty((T, N), dtype=np.int32)
        #: `μ(a_t|s_t)` da mistura que **de fato** escolheu a ação. É o denominador de
        #: todo peso de importância do update; sem ele gravado, `π/μ` viraria `π/π` e o
        #: V-trace não corrigiria nada — o mesmo cuidado que o `TrajectoryBuffer` toma
        #: para o ACER.
        mu_buf = np.empty((T, N), dtype=np.float32)
        #: A recompensa é a mesma para todas as políticas, **exceto** no bootstrap do
        #: truncamento por fome: ali entra `γ_i · V_i(s_final)`, que depende do γ e do
        #: crítico de cada uma.
        rew_buf = np.empty((T, N, P), dtype=np.float32)
        done_buf = np.empty((T, N), dtype=np.float32)

        shaping = self.shaping()
        #: `None` = caminho antigo (um shaping só, aplicado dentro do `VecSnake`). Tupla =
        #: o `VecSnake` devolve a recompensa esparsa pura e o shaping é montado aqui, um
        #: por política. Dois caminhos porque o antigo tem que continuar **exato**.
        shaping_i = self.shaping_por_politica()
        gamma_ref = cfg.gamma
        gammas_f = self.gammas.astype(np.float32)
        scores, vitorias = [], 0
        entropias = []

        for t in range(T):
            obs_buf[t], mask_buf[t] = self.obs, self.mask
            logits, _ = self._frente(self.model, tf.convert_to_tensor(self.obs))
            mu = self.espaco.comportamento(logits.numpy(), self.mask,
                                           self.tau, self.omega)

            a = (mu.cumsum(1) > self.rng.random((N, 1))).argmax(1).astype(np.int32)
            act_buf[t] = a
            mu_buf[t] = np.maximum(mu[np.arange(N), a], 1e-8)
            entropias.append(float(-(mu * np.log(mu + 1e-12)).sum(1).mean()))

            self.obs, self.mask, r, d, info = self.env.step(
                a, 0.0 if shaping_i is not None else shaping, gamma_ref)
            self.registra_fim(info)
            if shaping_i is None:
                rew_buf[t] = r[:, None]
            else:
                # `r` é a esparsa pura. Cada política soma o seu shaping, calculado com o
                # **seu** γ: o potencial `γΦ' − Φ` só é invariante para a política cujo γ
                # foi usado, então dar a todas o γ da avaliada enviesaria as outras duas.
                vale = info["shaping_valido"].astype(np.float32)[:, None]        # (N, 1)
                delta = (gammas_f[None, :] * info["phi_new"][:, None]
                         - info["phi_old"][:, None]) * vale                      # (N, P)
                rew_buf[t] = r[:, None] + shaping_i[None, :] * delta
            done_buf[t] = d.astype(np.float32)

            if info["trunc_idx"].size:      # fome é truncamento, não terminação
                _, v_f = self._frente(self.model,
                                      tf.convert_to_tensor(info["final_obs"]))
                v_f = v_f.numpy()
                for i in range(P):
                    rew_buf[t, :, i] = self.bootstrap_truncados(
                        info, rew_buf[t, :, i], v_f[:, i], self.gammas[i])

            # O bandit é atualizado com o retorno **não descontado** do episódio — o
            # score, que é a métrica do contrato. Creditar ao braço que estava no ar no
            # fim do episódio é exato aqui porque o braço só troca quando o episódio
            # acaba: ele esteve no ar durante o episódio inteiro.
            fim = np.nonzero(d)[0]
            if fim.size:
                self.mab.registrar_lote(self.braco[fim], info["scores"])
                self._novo_comportamento(fim)

            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

        self.global_step += T * N
        self.episodes += len(scores)

        lote = {
            "obs": obs_buf.reshape(T * N, b, b, c),
            "mask": mask_buf.reshape(T * N, N_ACTIONS),
            "act": act_buf.reshape(T * N),
            "mu": mu_buf.reshape(T * N),
            "rew": rew_buf,
            "done": done_buf,
            "obs_final": self.obs.copy(),
        }
        stats = {
            "train_score_mean": float(np.mean(scores)) if scores else None,
            "n_episodes": len(scores),
            "wins": vitorias,
            "shaping": shaping if shaping_i is None else float(shaping_i.mean()),
            "entropia_comportamento": float(np.mean(entropias)),
            "tau_medio": float(self.tau.mean()),
            #: Quão longe a mistura está de usar uma política só. 0 é one-hot (o caso
            #: Agent57), `log N` é uniforme. É a medida direta de "o mapeamento é híbrido
            #: ou degenerou?".
            "omega_entropia": float(
                -(self.omega * np.log(self.omega + 1e-12)).sum(1).mean()),
            **self.mab.resumo(),
        }
        return lote, stats

    # ------------------------------------------------------------------ alvos
    def _alvos(self, lote):
        """Recalcula π, V e os alvos V-trace com a rede **atual**.

        Roda uma vez por época, e não uma vez por rollout, porque é exatamente aí que o
        V-trace paga: `μ` está gravado, então cada época pode recorrigir contra a política
        que existe agora. Congelar os alvos na primeira época daria um alvo velho às
        outras três — que é o erro que o clipping do PPO existe para tolerar e que aqui
        não precisa ser tolerado.
        """
        cfg = self.cfg
        T, N, P = cfg.rollout, cfg.num_envs, cfg.n_politicas

        logits, valor = self._frente(self.model, tf.convert_to_tensor(lote["obs"]))
        mask3 = tf.expand_dims(tf.convert_to_tensor(lote["mask"]), 1)
        logits = tf.where(mask3, logits, tf.fill(tf.shape(logits), MASK_NEG))
        logp_all = tf.nn.log_softmax(logits)
        um = tf.one_hot(tf.convert_to_tensor(lote["act"]), N_ACTIONS)[:, None, :]
        logp_all_np = logp_all.numpy()
        logp = tf.reduce_sum(logp_all * um, axis=-1).numpy()          # (T*N, P)
        valor_plano = valor.numpy()                                   # (T*N, P)
        valor = valor_plano.reshape(T, N, P)

        razao = np.exp(logp - np.log(lote["mu"])[:, None]).reshape(T, N, P)
        rho = np.minimum(cfg.rho_barra, razao).astype(np.float32)
        cc = np.minimum(cfg.c_barra, razao).astype(np.float32)

        _, v_final = self._frente(self.model, tf.convert_to_tensor(lote["obs_final"]))
        v_final = v_final.numpy()                                      # (N, P)

        vs = np.empty((T, N, P), dtype=np.float32)
        adv = np.empty((T, N, P), dtype=np.float32)
        for i in range(P):
            vs[:, :, i], adv[:, :, i] = vtrace(
                lote["rew"][:, :, i], valor[:, :, i], lote["done"],
                v_final[:, i], rho[:, :, i], cc[:, :, i], float(self.gammas[i]))

        diag = {
            "razao_media": float(razao.mean()),
            #: Entropia média das políticas alvo, medida fora do grafo de treino. Com o
            #: `τ` já controlando a entropia do *comportamento*, este número diz se a
            #: política **avaliada** ainda tem para onde se mover.
            "entropia_alvo": float(
                -(np.exp(logp_all_np) * np.where(
                    np.isfinite(logp_all_np) & (logp_all_np > -1e8),
                    logp_all_np, 0.0)).sum(-1).mean()),
            **self._diversidade(logp_all_np),
            #: Fração de amostras em que o peso de importância bateu no teto. Perto de 0
            #: o comportamento está colado nas políticas alvo e o V-trace não está
            #: fazendo nada; perto de 1 a correção está saturada e o gradiente vira o de
            #: um on-policy enviesado. Ver `docs/LBC.md`.
            "razao_truncada": float((razao > cfg.rho_barra).mean()),
            "ev": variancia_explicada(valor[:, :, self.indice_alvo].ravel(),
                                      vs[:, :, self.indice_alvo].ravel()),
            #: `V_i` médio da pior cabeça em relação à avaliada. Perto de 1 a população é
            #: de pares; muito abaixo, as outras cabeças não são "diferentes", são piores —
            #: e sob `ω` uniforme elas assinam dois terços do comportamento.
            "valor_relativo_pior": float(
                valor.reshape(-1, P).mean(0).min()
                / (abs(valor.reshape(-1, P).mean(0)[self.indice_alvo]) + 1e-8)),
        }
        return (vs.reshape(T * N, P), adv.reshape(T * N, P),
                logp.astype(np.float32), valor_plano.astype(np.float32), diag)

    @staticmethod
    def _diversidade(logp_all):
        """A população é diversa, ou são `N` cópias caras da mesma política?

        Sem isto, a única forma de responder é carregar o checkpoint e medir por fora —
        que foi como se descobriu, tarde, que a população da primeira execução corrigida
        concordava no argmax em 31,8% dos estados (o acaso, com três ações, é 33%) e que
        duas das três cabeças eram simplesmente piores que a avaliada. É o instrumento que
        faltava para a §2.12 de `docs/LBC.md`.

        Dois números, medidos nos estados do próprio rollout:

        * `divergencia_populacao` — `KL(π_i ‖ π_j)` médio sobre todos os pares. Zero é
          população degenerada (o mapeamento híbrido não tem o que misturar); alto demais
          **não** é bom por si só: significa que as cabeças discordam, e discordar só vale
          quando as duas são competentes — por isso o número seguinte;
        * `acordo_argmax` — fração dos estados em que **todas** concordam na ação de maior
          probabilidade. Perto de 1 é degeneração; perto de `1/|A|` é acaso, e aí a
          população está resolvendo problemas diferentes em vez de resolver o mesmo
          problema de jeitos diferentes. O alvo saudável fica no meio.

        Ambos valem `nan` com população de uma política — não há par a comparar, e escrever
        zero ali afirmaria degeneração onde a pergunta nem existe.
        """
        n, P, _ = logp_all.shape
        if P < 2:
            return {"divergencia_populacao": float("nan"),
                    "acordo_argmax": float("nan")}
        p = np.exp(logp_all)
        kls = [float((p[:, i] * (logp_all[:, i] - logp_all[:, j])).sum(-1).mean())
               for i in range(P) for j in range(P) if i != j]
        am = logp_all.argmax(-1)
        return {"divergencia_populacao": float(np.mean(kls)),
                "acordo_argmax": float((am == am[:, :1]).all(-1).mean())}

    # ----------------------------------------------------------------- update
    @staticmethod
    @tf.function(reduce_retracing=True)
    def _train_step(model, optimizer, obs, mask, act, logp_ref, adv, vs, val_ref,
                    clip_eps, vf_clip, ent_coef, vf_coef, normalizar_vantagem,
                    usar_clip):
        """Um passo de gradiente sobre a **população inteira**, num forward só.

        A perda é a **soma** sobre as políticas de uma perda que é a média sobre o lote —
        e não a média sobre as duas coisas. A diferença não é cosmética: com a média, a
        perda de cada política sairia dividida por `N`, e uma população de três treinaria
        cada cabeça com um terço do gradiente que o PPO dá à dele. A comparação LBC × PPO
        passaria a incluir "e o LBC ainda usa um learning rate efetivo três vezes menor",
        que não é o que se quer medir. Somando, cada política recebe exatamente o gradiente
        que receberia sozinha, e o tronco recebe a soma — que é o que ele de fato deve
        aprender, já que serve às três.

        **A região de confiança.** `adv` já traz o `ρ` do V-trace embutido, então
        `−logπ·adv` é o gradiente do IMPALA. O que se acrescenta aqui é a casca clipada do
        PPO em volta dele, com a razão medida contra `logp_ref` — a política do **início da
        atualização**, e não da época. No primeiro minilote a razão é 1, `min(r·Â, clip(r)·Â)`
        é `Â`, e o gradiente é o do IMPALA sem alteração alguma; o clip só passa a existir
        depois, quando a política já andou. É a diferença entre reusar o lote 128 vezes e
        reusar o lote 128 vezes **sem sair do lugar onde ele foi coletado** — e é o que
        impede a saturação da softmax que matou a `seed0`.

        A vantagem é normalizada **por política** (eixo 0, mantendo a coluna), e não sobre
        o tensor inteiro: cada cabeça tem o seu γ e a sua escala de vantagem, e misturá-las
        faria a cabeça de γ = 0,999 ditar o passo das outras duas.

        As estatísticas devolvidas são **por política** (a média entre elas), para que
        `ent` continue legível na mesma escala dos outros agentes: 1,10 é uniforme sobre
        três ações, aqui como no PPO.
        """
        mask3 = tf.expand_dims(mask, 1)
        um = tf.expand_dims(tf.one_hot(act, N_ACTIONS), 1)
        if normalizar_vantagem:
            media = tf.reduce_mean(adv, axis=0, keepdims=True)
            desvio = tf.math.reduce_std(adv, axis=0, keepdims=True)
            adv = (adv - media) / (desvio + 1e-8)
        with tf.GradientTape() as tape:
            logits, valor = model(obs, training=True)
            logits = tf.where(mask3, logits, tf.fill(tf.shape(logits), MASK_NEG))
            logp_all = tf.nn.log_softmax(logits)
            logp = tf.reduce_sum(logp_all * um, axis=-1)               # (B, P)

            log_razao = logp - logp_ref
            razao = tf.exp(log_razao)
            if usar_clip:
                pg1 = -adv * razao
                pg2 = -adv * tf.clip_by_value(razao, 1.0 - clip_eps, 1.0 + clip_eps)
                pg_por_politica = tf.reduce_mean(tf.maximum(pg1, pg2), axis=0)  # (P,)

                v_clip = val_ref + tf.clip_by_value(valor - val_ref, -vf_clip, vf_clip)
                v_por_politica = 0.5 * tf.reduce_mean(
                    tf.maximum(tf.square(valor - vs), tf.square(v_clip - vs)), axis=0)
            else:
                # ablação: o gradiente do IMPALA cru, sem região de confiança. É o que a
                # `seed0` rodou, e o que colapsou.
                pg_por_politica = -tf.reduce_mean(logp * adv, axis=0)
                v_por_politica = 0.5 * tf.reduce_mean(tf.square(valor - vs), axis=0)

            probs = tf.exp(logp_all)
            seguro = tf.where(mask3, logp_all, tf.zeros_like(logp_all))
            ent_por_politica = -tf.reduce_mean(
                tf.reduce_sum(probs * seguro, axis=-1), axis=0)        # (P,)

            perda = tf.reduce_sum(pg_por_politica
                                  + vf_coef * v_por_politica
                                  - ent_coef * ent_por_politica)

        grads = tape.gradient(perda, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # estimador k3 do KL — não-negativo e de baixa variância, como no PPO
        kl = tf.reduce_mean(tf.exp(log_razao) - 1.0 - log_razao)
        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(razao - 1.0), clip_eps), tf.float32))
        return (tf.reduce_mean(pg_por_politica), tf.reduce_mean(v_por_politica),
                tf.reduce_mean(ent_por_politica), kl, clipfrac)

    def update(self, lote):
        cfg = self.cfg
        self.optimizer.learning_rate.assign(self.lr())
        n = lote["act"].shape[0]
        mb = max(1, n // cfg.minibatches)
        idx = np.arange(n)
        rng = np.random.default_rng(cfg.seed + self.iteration)

        # escalares como tensores — ver a nota em `PPO.update` (§2.6 da revisão)
        ent_c_valor = float(self._ent_coef)
        # `vf_clip <= 0` desliga o clip do valor: 1e9 é "sem teto" de forma exata, sem
        # precisar de um segundo ramo no grafo.
        teto_v = cfg.vf_clip if cfg.vf_clip > 0 else 1e9
        escalares = [tf.constant(v, tf.float32) for v in
                     (cfg.clip_eps, teto_v, ent_c_valor, cfg.vf_coef)]
        tensores = {k: tf.convert_to_tensor(lote[k]) for k in ("obs", "mask", "act")}

        logs = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        atualizacoes = 0
        epocas_feitas = 0
        parar = False
        diag = {}
        t_logp_ref = t_val_ref = None
        for _ in range(cfg.epochs):
            # Os **alvos** são recalculados a cada época (é aí que o V-trace paga: `μ` está
            # gravado, então cada época recorrige contra a política que existe agora). A
            # **referência da região de confiança**, não: ela é fixada na primeira época e
            # vale para a atualização inteira, exatamente como o `old_logp` do PPO. Se ela
            # fosse recalculada por época, o clip só limitaria o passo *dentro* de cada
            # época e a política poderia andar 4 × 0,2 sem nada reclamar — que é justamente
            # o que não pode acontecer.
            vs, adv, logp, valor, diag = self._alvos(lote)
            if t_logp_ref is None:
                t_logp_ref = tf.convert_to_tensor(logp)
                t_val_ref = tf.convert_to_tensor(valor)
            t_vs = tf.convert_to_tensor(vs)
            t_adv = tf.convert_to_tensor(adv)
            rng.shuffle(idx)
            for s_ini in range(0, n, mb):
                sl = tf.convert_to_tensor(idx[s_ini:s_ini + mb])
                pg, vf, e, kl, cf = self._train_step(
                    self.model, self.optimizer,
                    tf.gather(tensores["obs"], sl), tf.gather(tensores["mask"], sl),
                    tf.gather(tensores["act"], sl), tf.gather(t_logp_ref, sl),
                    tf.gather(t_adv, sl), tf.gather(t_vs, sl),
                    tf.gather(t_val_ref, sl), *escalares, cfg.normalizar_vantagem,
                    cfg.clip_eps > 0,
                )
                logs["pg"].append(float(pg))
                logs["vf"].append(float(vf))
                logs["ent"].append(float(e))
                logs["kl"].append(float(kl))
                logs["clipfrac"].append(float(cf))
                atualizacoes += 1
                if cfg.target_kl and float(kl) > cfg.target_kl * 1.5:
                    parar = True
                    break
            epocas_feitas += 1
            if parar:
                break

        saida = {k: float(np.mean(v)) for k, v in logs.items()}
        saida.update(diag)

        # Realimentação do coeficiente de entropia. Roda **depois** do update e vale para a
        # iteração seguinte: o que se mede aqui é a entropia que o passo já produziu.
        if cfg.ent_alvo is not None and logs["ent"]:
            fator = (cfg.ent_ajuste if saida["ent"] < cfg.ent_alvo
                     else 1.0 / cfg.ent_ajuste)
            self._ent_coef = float(np.clip(self._ent_coef * fator,
                                           cfg.ent_coef_min, cfg.ent_coef_max))

        saida["lr"] = float(self.lr())
        saida["ent_coef"] = ent_c_valor
        saida["epochs_done"] = epocas_feitas
        saida["atualizacoes"] = int(atualizacoes)
        return saida

    # ------------------------------------------------------------------- passo
    def iterate(self):
        lote, stats = self.collect()
        stats.update(self.update(lote))
        return stats
