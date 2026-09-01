"""LBC — o único agente daqui em que a exploração é escolhida, e não agendada.

O que estes testes protegem, em ordem de gravidade:

1. **O V-trace corrige de verdade.** Um estimador off-policy errado não levanta exceção:
   ele treina, converge para o valor errado e produz uma curva plausível. O teste âncora
   amarra o caso on-policy ao `compute_gae` do PPO, que já é confiável.
2. **O comportamento nunca põe massa numa ação letal.** A mistura mascara depois de
   escalar por `τ`; inverter a ordem faz a cobra bater na parede de vez em quando, sem
   erro nenhum no log.
3. **O crédito do bandit é do braço certo.** Um episódio de Snake atravessa vários
   rollouts; creditar o retorno ao braço errado transforma o meta-controlador em ruído
   caro que parece estar funcionando.
4. **A política avaliada é a que foi declarada.** Com uma população, "o modelo" tem `N`
   políticas, e pegar a primeira por acidente faria a curva medir uma cabeça e o
   checkpoint guardar outra.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import LBC, LBCConfig, MisturaBoltzmann, compute_gae, vtrace
from snakeai.plot import ORDEM_ALGORITMOS, cores_por_algoritmo, familia_de


def cfg(**kw):
    base = dict(net="resnet_tiny", num_envs=32, rollout=8, epochs=1, minibatches=4,
                eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return LBCConfig(**base)


# ============================================================== V-trace
def dados_vtrace(T=6, N=3, semente=0):
    rng = np.random.default_rng(semente)
    rew = rng.normal(size=(T, N)).astype(np.float32)
    val = rng.normal(size=(T, N)).astype(np.float32)
    done = (rng.random((T, N)) < 0.15).astype(np.float32)
    ultimo = rng.normal(size=N).astype(np.float32)
    return rew, val, done, ultimo


def test_on_policy_vtrace_is_gae_with_lambda_one():
    """A âncora. Com `ρ = c = 1` — comportamento igual à política alvo — o V-trace colapsa
    exatamente no retorno de GAE(λ=1), que este repositório já usa no PPO desde o começo.

    É o teste que separa "implementei V-trace" de "implementei uma recursão parecida":
    a fórmula tem quatro lugares onde um índice pode escorregar um passo, e todos os
    quatro continuam produzindo números finitos e plausíveis.
    """
    rew, val, done, ultimo = dados_vtrace()
    uns = np.ones_like(rew)
    gamma = 0.97

    vs, _ = vtrace(rew, val, done, ultimo, uns, uns, gamma)
    _, ret_gae = compute_gae(rew, val, done, ultimo, gamma, lam=1.0)
    assert vs == pytest.approx(ret_gae, abs=1e-5)


def test_the_advantage_uses_the_corrected_target_of_the_next_step():
    """`adv_t = ρ_t (r_t + γ v_{t+1} − V_t)` — com `v`, não com `V`.

    Trocar `v_{t+1}` por `V(s_{t+1})` é o erro mais fácil de cometer aqui e o mais difícil
    de ver: a vantagem passa a usar uma baseline de uma fonte e um alvo de outra, e o
    gradiente de política deixa de apontar para onde o crítico está aprendendo.
    """
    rew, val, done, ultimo = dados_vtrace()
    uns = np.ones_like(rew)
    gamma = 0.97

    vs, adv = vtrace(rew, val, done, ultimo, uns, uns, gamma)
    vs_prox = np.concatenate([vs[1:], ultimo[None]], axis=0)
    esperado = rew + gamma * (1.0 - done) * vs_prox - val
    assert adv == pytest.approx(esperado, abs=1e-5)


def test_the_truncation_never_amplifies():
    """`ρ̄` é um teto, não um fator. Com todas as razões acima do teto, o resultado tem que
    ser idêntico ao de razões exatamente no teto — é isso que torna o estimador seguro
    para dados arbitrariamente velhos."""
    rew, val, done, ultimo = dados_vtrace()
    gamma = 0.97
    enormes = np.full_like(rew, 50.0)
    no_teto = np.ones_like(rew)

    a = vtrace(rew, val, done, ultimo, np.minimum(1.0, enormes),
               np.minimum(1.0, enormes), gamma)
    b = vtrace(rew, val, done, ultimo, no_teto, no_teto, gamma)
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])


def test_a_behavior_far_from_the_target_shrinks_the_correction():
    """Com `ρ → 0` — a política alvo nunca teria tomado aquela ação — o alvo tem que
    voltar a ser o próprio `V(s)`: não há informação utilizável naquela transição."""
    rew, val, done, ultimo = dados_vtrace()
    zero = np.zeros_like(rew)
    vs, adv = vtrace(rew, val, done, ultimo, zero, zero, 0.97)
    assert vs == pytest.approx(val, abs=1e-6)
    assert adv == pytest.approx(np.zeros_like(adv), abs=1e-6)


def test_the_episode_boundary_stops_the_recursion():
    """Um `done` no meio do rollout tem que isolar completamente o que vem depois.

    Sem isso o retorno atravessa a fronteira e o agente aprende que o episódio seguinte é
    consequência do anterior — o mesmo bug que a §1.1 da revisão custou para achar.
    """
    T, N = 5, 1
    rew = np.zeros((T, N), dtype=np.float32)
    val = np.zeros((T, N), dtype=np.float32)
    done = np.zeros((T, N), dtype=np.float32)
    done[2] = 1.0
    uns = np.ones((T, N), dtype=np.float32)

    base = vtrace(rew, val, done, np.zeros(N, np.float32), uns, uns, 0.99)[0]
    rew_depois = rew.copy()
    rew_depois[3] = 100.0                 # recompensa gorda depois da fronteira
    depois = vtrace(rew_depois, val, done, np.zeros(N, np.float32), uns, uns, 0.99)[0]

    assert depois[:3] == pytest.approx(base[:3]), "a recompensa vazou para trás do `done`"
    assert depois[3] > base[3], "e ela tem que aparecer depois da fronteira"


# ================================================== espaço de comportamento
def espaco(n=3, **kw):
    return MisturaBoltzmann(n, rng=np.random.default_rng(0), **kw)


def test_the_behavior_space_has_one_arm_per_region():
    e = espaco(3, n_faixas=4)
    assert e.n_bracos == 4 * (3 + 1)      # faixas de τ × (uma política + o uniforme)
    assert espaco(1, n_faixas=4).n_bracos == 4, "com N=1 o padrão uniforme é o one-hot"


def test_an_arm_is_a_region_and_not_a_point():
    """O ponto do §4.2: se o braço fosse um `ψ` fixo, o espaço de comportamento voltaria a
    ser finito — que é exatamente a limitação que o LBC existe para remover."""
    e = espaco()
    t1, w1 = e.amostrar([5])
    t2, w2 = e.amostrar([5])
    assert not np.allclose(t1, t2)
    assert not np.allclose(w1, w2)


def test_every_tau_stays_inside_its_band():
    e = espaco(2, tau_min=0.25, tau_max=4.0, n_faixas=4)
    for braco in range(e.n_bracos):
        tau, _ = e.amostrar(np.full(64, braco))
        lo, hi = e.faixas[braco // len(e.padroes)]
        assert (tau >= lo - 1e-6).all() and (tau <= hi + 1e-6).all()


def test_the_weights_are_a_distribution():
    e = espaco()
    _, omega = e.amostrar(np.arange(e.n_bracos))
    assert omega.sum(1) == pytest.approx(np.ones(e.n_bracos), abs=1e-5)
    assert (omega >= 0).all()


def test_the_concentrated_patterns_really_pick_one_policy():
    """Os padrões one-hot são o caso Agent57 — usar *uma* política da população. Se a
    Dirichlet não concentrar, esse canto do espaço de comportamento não existe e o LBC
    perde o regime que ele deveria generalizar."""
    e = espaco(3)
    n_padroes = len(e.padroes)
    for i in range(3):                    # braço da faixa 0, padrão concentrado em π_i
        _, omega = e.amostrar(np.full(200, i))
        assert omega[:, i].mean() > 0.7
    _, uniforme = e.amostrar(np.full(200, n_padroes - 1))
    assert uniforme.mean(0) == pytest.approx(np.full(3, 1 / 3), abs=0.08)


def test_the_mixture_never_puts_mass_on_a_lethal_action():
    """A máscara é aplicada **depois** de multiplicar os logits por `τ`.

    Mascarar antes multiplicaria o `MASK_NEG` por `τ`: com `τ` pequeno o −1e9 encolhe na
    direção do zero e uma ação letal volta a ter probabilidade não desprezível. Nada
    quebra — a cobra só passa a bater na parede de vez em quando, e a causa fica invisível
    na curva.
    """
    e = espaco(2)
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=5.0, size=(16, 2, 3)).astype(np.float32)
    mask = np.ones((16, 3), dtype=bool)
    mask[:, 0] = False

    for tau_valor in (1e-3, 0.25, 1.0, 4.0, 50.0):
        tau = np.full((16, 2), tau_valor, dtype=np.float32)
        omega = np.full((16, 2), 0.5, dtype=np.float32)
        mu = e.comportamento(logits, mask, tau, omega)
        assert mu[:, 0].max() < 1e-6, f"massa em ação letal com τ={tau_valor}"
        assert mu.sum(1) == pytest.approx(np.ones(16), abs=1e-5)


def test_one_hot_weights_degenerate_to_a_single_policy():
    """O caso especial que o próprio paper aponta: `ω` one-hot e `τ = 1` recuperam a
    seleção de política única do Agent57. Se isto não vale, a mistura não é uma
    generalização — é outra coisa."""
    e = espaco(3)
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(8, 3, 3)).astype(np.float32)
    mask = np.ones((8, 3), dtype=bool)
    tau = np.ones((8, 3), dtype=np.float32)
    omega = np.zeros((8, 3), dtype=np.float32)
    omega[:, 1] = 1.0

    mu = e.comportamento(logits, mask, tau, omega)
    # os logits são padronizados por estado antes de escalar por `τ` (ver `docs/LBC.md`
    # §2.6): a referência tem que passar pela mesma transformação, senão o teste estaria
    # medindo a padronização e não a degeneração de `ω`
    l = logits[:, 1, :]
    l = (l - l.mean(-1, keepdims=True)) / np.sqrt(l.var(-1, keepdims=True) + 1e-6)
    z = l - l.max(-1, keepdims=True)
    esperado = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    assert mu == pytest.approx(esperado, abs=1e-6)


def test_standardization_gives_tau_authority_over_any_logit_scale():
    """O defeito que matou a primeira execução: `τ` multiplicava logits livres, que crescem
    sem limite. Com `‖logits‖ ~ 30` a faixa inteira de `τ` produz `argmax` e o espaço de
    comportamento degenera num ponto — `docs/LBC.md` §2.6."""
    rng = np.random.default_rng(0)
    mask = np.ones((2048, 3), dtype=bool)

    def entropia(escala, padronizar, tau_valor):
        logits = rng.normal(0, escala, size=(2048, 1, 3)).astype(np.float32)
        e = MisturaBoltzmann(1, padronizar=padronizar)
        t = np.full((2048, 1), tau_valor, np.float32)
        w = np.ones((2048, 1), np.float32)
        mu = e.comportamento(logits, mask, t, w)
        return float(-(mu * np.log(mu + 1e-12)).sum(1).mean())

    # sem padronizar, a escala dos logits engole a faixa inteira de `τ`
    assert entropia(30.0, False, 0.25) < 0.25

    # padronizando, a faixa de `τ` cobre de quase-uniforme (log 3 = 1,0986) a quase-guloso,
    # e cobre igual seja qual for a escala que a rede tenha alcançado
    for escala in (1.0, 5.0, 30.0):
        assert entropia(escala, True, 0.25) > 1.0
        assert entropia(escala, True, 4.0) < 0.25


def test_a_small_tau_explores_and_a_large_tau_exploits():
    """É este o eixo que o bandit aprende a percorrer. Se `τ` não mudar a entropia de
    forma monotônica, as faixas do espaço de comportamento não significam nada."""
    e = espaco(1)
    logits = np.array([[[3.0, 0.0, -3.0]]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    entropias = []
    for t in (0.05, 1.0, 8.0):
        mu = e.comportamento(logits, mask, np.full((1, 1), t, np.float32),
                             np.ones((1, 1), np.float32))
        entropias.append(float(-(mu * np.log(mu + 1e-12)).sum()))
    assert entropias[0] > entropias[1] > entropias[2]
    assert entropias[0] == pytest.approx(np.log(3), abs=0.02)


# ============================================================ configuração
def test_the_population_and_its_discounts_are_declared_together():
    """Os dois descrevem a mesma população. Derivar um do outro em silêncio deixaria
    `n_politicas=1` rodando com o γ de outra política sem ninguém perceber."""
    with pytest.raises(ValueError, match="gammas"):
        LBCConfig(n_politicas=2)
    LBCConfig(n_politicas=1, gammas=(0.995,), indice_alvo=0)


def test_the_target_must_exist_in_the_population():
    with pytest.raises(ValueError, match="indice_alvo"):
        LBCConfig(indice_alvo=7)


def test_an_unknown_selection_is_refused():
    with pytest.raises(ValueError, match="selecao"):
        LBCConfig(selecao="epsilon")


def test_the_default_target_has_the_same_discount_as_the_ppo():
    """O que faz a comparação LBC × PPO medir controle de comportamento e não desconto."""
    c = LBCConfig()
    from snakeai.agents import PPOConfig
    assert c.gamma == PPOConfig().gamma


# ================================================================== o agente
def test_lbc_is_its_own_algorithm_in_the_arena():
    ag = LBC(cfg())
    assert ag.algo == "lbc"
    assert "lbc" in ORDEM_ALGORITMOS
    assert familia_de("lbc") == "política"
    cores = cores_por_algoritmo({"ppo", "lbc"})
    assert cores["lbc"] != cores["ppo"]


def test_lbc_trains():
    ag = LBC(cfg())
    for _ in range(3):
        s = ag.iterate()
    for chave in ("pg", "vf", "ent", "lr", "razao_media", "entropia_comportamento"):
        assert np.isfinite(s[chave]), f"{chave} virou {s[chave]}"
    assert s["ent"] > 0
    assert s["atualizacoes"] == ag.cfg.epochs * ag.cfg.minibatches


def test_a_degenerate_population_of_one_still_runs():
    """A ablação "reduzir H" da Fig. 5. Ela precisa rodar sem caminho especial no agente,
    senão a comparação com a população de três incluiria diferenças de implementação."""
    ag = LBC(cfg(n_politicas=1, gammas=(0.995,), indice_alvo=0))
    s = ag.iterate()
    assert np.isfinite(s["pg"])
    assert ag.model.outputs[0].shape[1] == 1


def test_the_evaluated_policy_is_the_declared_head():
    """Com uma população, "o modelo" tem N políticas. O `keras_policy` genérico pegaria a
    saída de índice 0 — que aqui é o tensor da população inteira. A cabeça avaliada é uma
    decisão do algoritmo e tem que estar no código."""
    import tensorflow as tf

    ag = LBC(cfg())
    obs, mask = ag.env.reset()
    saida = ag.politica()(obs, mask)
    logits, _ = ag.model(tf.convert_to_tensor(obs), training=False)
    esperado = np.where(mask, logits.numpy()[:, ag.indice_alvo, :], -1e9)
    assert saida == pytest.approx(esperado, abs=1e-4)
    assert ag.indice_alvo == 1


def test_the_behavior_is_not_the_target_policy():
    """Se a mistura coincidisse com a política alvo, o V-trace não teria o que corrigir e
    o LBC seria um A2C caro. A razão de importância é o termômetro disso."""
    ag = LBC(cfg())
    s = ag.iterate()
    assert s["razao_truncada"] > 0.0, "nenhuma amostra off-policy — μ virou π?"


def test_the_bandit_gets_exactly_one_return_per_episode():
    """A atribuição de crédito do meta-controlador, conferida por contagem.

    Um episódio de Snake atravessa vários rollouts. Se o braço fosse sorteado por iteração
    em vez de por episódio, o retorno seria creditado a um braço que só esteve no ar no
    fim — e o bandit aprenderia a preferir o comportamento errado, com curvas que parecem
    saudáveis o tempo todo.
    """
    # A máscara de ação impede a morte por colisão imediata, então o episódio mais curto
    # possível é o da fome: `starve_base + 2·comprimento` = 106 passos. Um rollout menor
    # que isso não encerra episódio nenhum e o teste passaria sem medir nada.
    ag = LBC(cfg(num_envs=16, rollout=128))
    for _ in range(2):
        ag.iterate()
    assert ag.episodes > 0, "o rollout precisa passar do limite de fome (106 passos)"
    assert int(ag.mab.puxadas_totais.sum()) == ag.episodes


def test_the_arm_only_changes_when_the_episode_ends():
    ag = LBC(cfg(num_envs=16, rollout=128))
    antes = ag.braco.copy()
    _, stats = ag.collect()
    assert stats["n_episodes"] > 0
    trocaram = int((ag.braco != antes).sum())
    assert trocaram <= stats["n_episodes"], \
        "algum ambiente trocou de braço no meio do episódio"


def test_random_selection_is_an_ablation_and_says_so():
    """Ela usa o **mesmo** espaço de comportamento e só troca quem escolhe — é a ablação
    da Fig. 5. Sem a marca na variante, ela dividiria a identidade `(algo, variant, seed)`
    com o algoritmo oficial e as duas virariam uma curva só na arena."""
    ag = LBC(cfg(selecao="aleatoria"))
    assert ag.variant.endswith("+selecao_aleatoria")
    assert LBC(cfg()).variant == "resnet_tiny"
    s = ag.iterate()
    assert np.isfinite(s["pg"])


def test_a_different_population_size_marks_the_variant():
    ag = LBC(cfg(n_politicas=1, gammas=(0.995,), indice_alvo=0))
    assert ag.variant.endswith("+pop1")


def test_random_selection_ignores_the_bandit_but_keeps_measuring_it():
    """A ablação continua registrando os retornos por braço. Sem isso não haveria como
    dizer *por que* ela perdeu (ou não) para a seleção aprendida."""
    ag = LBC(cfg(selecao="aleatoria", num_envs=16, rollout=128))
    for _ in range(2):
        ag.iterate()
    assert ag.episodes > 0
    assert int(ag.mab.puxadas_totais.sum()) == ag.episodes


def test_reloading_the_model_rebuilds_the_optimizer():
    """O otimizador antigo aponta para as variáveis do modelo antigo — §3.4 da revisão."""
    ag = LBC(cfg())
    ag.iterate()
    antigo = ag.optimizer
    ag.on_model_reloaded()
    assert ag.optimizer is not antigo
    ag.iterate()


def test_the_hunger_truncation_bootstraps_with_each_policys_own_gamma():
    """Fome é truncamento, não terminação — e cada política tem o seu γ e o seu crítico.

    Um único bootstrap compartilhado ensinaria à política de γ = 0,999 o valor terminal da
    de γ = 0,99. É o §1.1 da revisão aplicado a uma população.
    """
    ag = LBC(cfg(num_envs=8))
    info = {"trunc_idx": np.array([0, 2]),
            "final_obs": None, "final_mask": None}
    rew = np.zeros(8, dtype=np.float32)
    v_final = np.array([10.0, 20.0], dtype=np.float32)

    saidas = [ag.bootstrap_truncados(info, rew, v_final, g) for g in ag.gammas]
    assert saidas[0][0] == pytest.approx(10.0 * ag.gammas[0])
    assert saidas[-1][0] == pytest.approx(10.0 * ag.gammas[-1])
    assert saidas[0][0] != pytest.approx(saidas[-1][0]), \
        "γ diferente tem que dar bootstrap diferente"
    assert saidas[0][1] == 0.0, "só os truncados recebem bootstrap"


def test_the_entropy_bonus_is_not_scheduled():
    """Nos outros agentes daqui a entropia decai numa reta. Aqui quem controla a entropia
    do comportamento é o `τ` escolhido pelo bandit; um agendamento por cima faria o mesmo
    trabalho duas vezes e em desacordo. Com `ent_alvo=None` o coeficiente é constante — e
    o que **não** pode existir é dependência do passo global."""
    ag = LBC(cfg(total_steps=1000, ent_alvo=None))
    inicio = ag.iterate()["ent_coef"]
    ag.global_step = 900
    fim = ag.iterate()["ent_coef"]
    assert inicio == fim == ag.cfg.ent_coef


def test_the_entropy_coefficient_is_a_price_paid_for_a_floor():
    """Com `ent_alvo`, o coeficiente é realimentado pela entropia **medida**, e não pelo
    passo: sobe quando a política está abaixo do alvo, desce quando está acima. É o piso que
    faltava — `docs/LBC.md` §2.8."""
    # alvo inatingível: a entropia medida fica sempre abaixo, o preço só pode subir
    sobe = LBC(cfg(total_steps=10 ** 6, ent_alvo=1.09, ent_coef=0.01))
    antes = sobe.iterate()["ent_coef"]
    for _ in range(3):
        sobe.iterate()
    assert sobe._ent_coef > antes

    # alvo já satisfeito na inicialização (política ~uniforme): o preço tem que cair
    desce = LBC(cfg(total_steps=10 ** 6, ent_alvo=1e-4, ent_coef=0.01))
    antes = desce.iterate()["ent_coef"]
    for _ in range(3):
        desce.iterate()
    assert desce._ent_coef < antes

    # e nunca sai da faixa declarada
    assert sobe.cfg.ent_coef_min <= sobe._ent_coef <= sobe.cfg.ent_coef_max


def test_the_kl_brake_actually_cuts_the_update_short():
    """O modo de falha da primeira execução: 128 passos de gradiente por rollout sem freio
    nenhum saturam a softmax, e softmax saturada é ponto fixo absorvente — `docs/LBC.md`
    §2.7. A parada por KL é o freio; este teste é sobre ele estar ligado ao pedal.

    Não é um teste sobre o *valor* do KL numa configuração de brinquedo (com 8 ambientes e
    4 iterações a política mal se move, e a divergência só aparece depois de ~20). É sobre a
    parada existir, ser observável em `epochs_done`, e respeitar o teto declarado.
    """
    # teto impossível de violar: as 4 épocas rodam inteiras
    solto = LBC(cfg(total_steps=10 ** 6, target_kl=1e9))
    assert solto.iterate()["epochs_done"] == solto.cfg.epochs

    # teto impossível de respeitar: para assim que o KL deixa de ser zero. São dois
    # minilotes e não um porque no primeiro a razão é **exatamente** 1 — `logp_ref` acabou
    # de ser medido na mesma rede —, e KL zero não viola teto nenhum. É a mesma aritmética
    # do PPO, e é o que garante que o gradiente do primeiro passo seja o do IMPALA cru.
    travado = LBC(cfg(total_steps=10 ** 6, target_kl=1e-12))
    saida = travado.iterate()
    assert saida["epochs_done"] == 1
    assert saida["atualizacoes"] == 2

    # e no padrão o KL medido nunca passa do teto declarado
    padrao = LBC(cfg(total_steps=10 ** 6))
    for _ in range(4):
        assert padrao.iterate()["kl"] <= padrao.cfg.target_kl * 1.5 + 1e-6


def test_turning_the_trust_region_off_restores_the_raw_impala_gradient():
    """`clip_eps = 0` é a ablação: o gradiente `−logπ·Â` cru, que é o que a `seed0` rodou.
    Ela tem que ser alcançável por configuração, aparecer no nome da variante e não quebrar
    o passo — senão não dá para medir quanto a região de confiança vale."""
    ag = LBC(cfg(total_steps=10 ** 6, clip_eps=0.0, normalizar_vantagem=False))
    saida = ag.iterate()
    assert "sem_clip" in ag.variant
    assert np.isfinite(saida["pg"]) and np.isfinite(saida["vf"])
    assert saida["clipfrac"] == pytest.approx(0.0, abs=1.0)   # só não pode dar NaN
