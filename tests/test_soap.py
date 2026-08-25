"""SOAP — o agente com um latente discreto que atravessa os passos.

O que estes testes protegem, em ordem de gravidade:

1. **O controle exato.** Com uma opção só, o SOAP tem que *ser* o PPO — mesmo GAE, mesma
   vantagem, mesma responsabilidade, mesma razão. Sem isso a comparação entre as duas
   curvas mediria a implementação, não as opções. É a âncora da suíte inteira.
2. **A crença nunca vaza entre episódios.** `ζ` é a única coisa aqui que sobrevive a um
   passo; deixá-la atravessar a morte da cobra faz o agente começar a partida acreditando
   estar num regime que pertence à anterior — e nada quebra.
3. **A vantagem de opção é centrada.** Se não for, ela desloca o gradiente da
   sub-política em vez de só redistribuir entre as opções, e o SOAP vira um PPO com uma
   baseline errada.
4. **A avaliação usa o marginal, com memória.** Avaliar uma sub-política isolada, ou o
   marginal com `ζ` congelado, dá um número mais baixo por defeito de medição — e a
   conclusão "opções não ajudam" viria daí.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import (SOAP, PoliticaComOpcoes, SOAPConfig, compute_gae,
                            gae_de_opcoes, vantagem_de_opcao)
from snakeai.nets import build_option_actor_critic
from snakeai.plot import ORDEM_ALGORITMOS, cores_por_algoritmo, familia_de


def cfg(**kw):
    base = dict(net="resnet_tiny", num_envs=32, rollout=8, epochs=1, minibatches=4,
                eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return SOAPConfig(**base)


def dados(T=7, N=4, Z=3, semente=0):
    """Um rollout sintético coerente: `π_ψ` normalizado, `ζ` normalizado, `done` esparso."""
    rng = np.random.default_rng(semente)
    rew = rng.normal(size=(T, N, Z)).astype(np.float32)
    val = rng.normal(size=(T, N, Z)).astype(np.float32)
    done = (rng.random((T, N)) < 0.15).astype(np.float32)
    pi_z = rng.random((T, N, Z, Z)).astype(np.float32)
    pi_z /= pi_z.sum(-1, keepdims=True)
    zeta = rng.random((T, N, Z)).astype(np.float32)
    zeta /= zeta.sum(-1, keepdims=True)
    zeta_final = np.full((N, Z), 1.0 / Z, dtype=np.float32)
    ultimo_v = rng.normal(size=(N, Z)).astype(np.float32)
    pi_a = rng.random((T, N, Z)).astype(np.float32)
    alpha = np.einsum("tnz,tnz->tn", zeta, pi_a).astype(np.float32)
    conj = pi_a[:, :, :, None] * pi_z
    return dict(rew=rew, val=val, done=done, pi_z=pi_z, zeta=zeta,
                zeta_final=zeta_final, ultimo_v=ultimo_v, conj=conj, alpha=alpha)


# ==================================================== o controle: uma opção só
def test_with_one_option_the_option_gae_is_the_ppo_gae():
    """A âncora. Com `Z = 1` não há para onde marginalizar: `π_ψ ≡ 1` e a recursão por
    par de opções colapsa, termo a termo, na do `compute_gae` que o PPO usa desde o
    começo do repositório.

    É este teste que separa "implementei o GAE com opções" de "implementei uma recursão
    parecida": há quatro lugares na fórmula onde um índice pode escorregar um passo, e os
    quatro continuam produzindo números finitos e plausíveis.
    """
    d = dados(Z=1)
    gamma, lam = 0.97, 0.9
    a_marg, _ = gae_de_opcoes(d["rew"], d["val"], d["done"], d["pi_z"], d["ultimo_v"],
                              gamma, lam)
    adv, _ = compute_gae(d["rew"][:, :, 0], d["val"][:, :, 0], d["done"],
                         d["ultimo_v"][:, 0], gamma, lam)
    assert a_marg[:, :, 0] == pytest.approx(adv, abs=1e-5)


def test_with_one_option_the_option_advantage_is_the_plain_advantage():
    """Segunda metade do controle: sem opções para redistribuir, o termo de utilidade
    centrada é identicamente zero e `A^GOA` vira `A^GAE`. Junto com o teste anterior, isso
    fecha a afirmação "SOAP com `n_opcoes=1` **é** PPO" — que é o que dá sentido à
    ablação."""
    d = dados(Z=1)
    a_marg, _ = gae_de_opcoes(d["rew"], d["val"], d["done"], d["pi_z"], d["ultimo_v"],
                              0.97, 0.9)
    goa = vantagem_de_opcao(a_marg, d["zeta"], d["zeta_final"], d["done"], d["conj"],
                            d["alpha"])
    assert goa == pytest.approx(a_marg, abs=1e-5)


def test_one_option_makes_the_belief_trivial_and_alpha_the_action_probability():
    """O mesmo controle, agora dentro do agente e não na álgebra: com uma opção, `ζ ≡ 1` e
    `α_t` é exatamente `π(a_t|s_t)` — o denominador da razão do PPO."""
    ag = SOAP(cfg(n_opcoes=1))
    lote, _ = ag.collect()
    assert lote["zeta"] == pytest.approx(np.ones_like(lote["zeta"]))
    assert lote["alpha"] == pytest.approx(lote["pi_a"][:, :, 0], abs=1e-6)
    assert lote["pi_z"] == pytest.approx(np.ones_like(lote["pi_z"]))


# ============================================================ os estimadores
def test_the_option_advantage_is_centred_under_the_next_belief():
    """`A^GOA(z') = base + (1−d)[U(z') − E_ζ U]`, e o termo entre colchetes tem média zero
    sob `ζ_{t+1}` **por construção**.

    Isso importa mais do que parece: a média sob a crença é o que a sub-política enxerga.
    Se ela não fosse zero, o termo de opção deslocaria o gradiente de `π_θ` — e o SOAP
    passaria a treinar a política de ação contra uma baseline enviesada, sem que nada
    além da curva mudasse.
    """
    d = dados()
    a_marg, _ = gae_de_opcoes(d["rew"], d["val"], d["done"], d["pi_z"], d["ultimo_v"],
                              0.97, 0.9)
    goa = vantagem_de_opcao(a_marg, d["zeta"], d["zeta_final"], d["done"], d["conj"],
                            d["alpha"])

    base = np.einsum("tnz,tnz->tn", a_marg, d["zeta"])
    # a crença do passo seguinte: `ζ_{t+1}` é `ζ[t+1]`, e `ζ_final` no último passo
    zeta_prox = np.concatenate([d["zeta"][1:], d["zeta_final"][None]], axis=0)
    media = np.einsum("tnz,tnz->tn", goa, zeta_prox)
    assert media == pytest.approx(base, abs=1e-4)


def test_the_option_signal_dies_at_an_episode_boundary():
    """Num passo terminal não há opção seguinte que valha alguma coisa: `A^GOA` tem que
    ficar constante em `z'`. Sem o `(1−d_t)`, a utilidade do episódio seguinte vazaria
    para dentro deste — o mesmo defeito que a §1.1 da revisão custou para achar, agora no
    eixo das opções."""
    d = dados()
    d["done"][:] = 1.0
    a_marg, _ = gae_de_opcoes(d["rew"], d["val"], d["done"], d["pi_z"], d["ultimo_v"],
                              0.97, 0.9)
    goa = vantagem_de_opcao(a_marg, d["zeta"], d["zeta_final"], d["done"], d["conj"],
                            d["alpha"])
    assert goa.std(axis=-1) == pytest.approx(np.zeros(goa.shape[:2]), abs=1e-6)


def test_a_reward_after_a_done_does_not_leak_backwards():
    T, N, Z = 5, 1, 2
    rew = np.zeros((T, N, Z), dtype=np.float32)
    val = np.zeros((T, N, Z), dtype=np.float32)
    done = np.zeros((T, N), dtype=np.float32)
    done[2] = 1.0
    pi_z = np.full((T, N, Z, Z), 0.5, dtype=np.float32)
    ultimo = np.zeros((N, Z), dtype=np.float32)

    base, _ = gae_de_opcoes(rew, val, done, pi_z, ultimo, 0.99, 0.95)
    depois = rew.copy()
    depois[3] = 100.0
    novo, _ = gae_de_opcoes(depois, val, done, pi_z, ultimo, 0.99, 0.95)

    assert novo[:3] == pytest.approx(base[:3]), "a recompensa vazou para trás do `done`"
    assert novo[3].max() > base[3].max()


def test_the_marginal_gae_follows_the_option_transition():
    """`A_t(z)` é a média de `A_t(z,z')` sob `π_ψ(·|s,a,z)`. Com uma transição
    determinística, ela tem que ser exatamente o par correspondente — se não for, o
    crítico da opção corrente está sendo treinado contra o futuro de outra opção."""
    T, N, Z = 3, 2, 3
    d = dados(T=T, N=N, Z=Z, semente=5)
    pi_z = np.zeros((T, N, Z, Z), dtype=np.float32)
    pi_z[:, :, :, 1] = 1.0                        # toda opção vai para a de índice 1
    a_marg, a_par = gae_de_opcoes(d["rew"], d["val"], d["done"], pi_z, d["ultimo_v"],
                                  0.99, 0.95)
    assert a_marg == pytest.approx(a_par[:, :, :, 1], abs=1e-5)


# ==================================================================== a rede
def test_the_option_network_exposes_the_three_heads():
    m = build_option_actor_critic(10, "resnet_tiny", n_opcoes=4)
    la, lz, v = m.outputs
    assert tuple(la.shape[1:]) == (4, 3)
    assert tuple(lz.shape[1:]) == (4, 3, 4), "π_ψ depende de (s, a, z), não só de s"
    assert tuple(v.shape[1:]) == (4,), "o crítico é condicionado à opção"


def test_the_network_refuses_zero_options():
    with pytest.raises(ValueError):
        build_option_actor_critic(10, "resnet_tiny", n_opcoes=0)


# ================================================================== a crença
def test_the_belief_stays_a_distribution_through_a_rollout():
    ag = SOAP(cfg())
    for _ in range(2):
        lote, _ = ag.collect()
        assert lote["zeta"].sum(-1) == pytest.approx(
            np.ones(lote["zeta"].shape[:2]), abs=1e-4)
        assert (lote["zeta"] >= 0).all()
    assert ag.zeta.sum(-1) == pytest.approx(np.ones(ag.cfg.num_envs), abs=1e-4)


def test_the_belief_resets_when_the_episode_ends():
    """`ζ` é a única coisa que sobrevive a um passo. Deixá-la atravessar a morte da cobra
    faz o agente começar a partida convencido de estar num regime que pertence à
    anterior — e nada quebra, a curva só fica pior."""
    # A máscara de ação impede a colisão imediata, então o episódio mais curto possível é
    # o da fome: `starve_base + 2·comprimento` = 106 passos. Um rollout menor que isso não
    # encerra episódio nenhum e o teste passaria sem medir nada.
    ag = SOAP(cfg(num_envs=16, rollout=128))
    ag.zeta[:] = np.eye(ag.cfg.n_opcoes, dtype=np.float32)[0]
    lote, stats = ag.collect()
    assert stats["n_episodes"] > 0

    uniforme = np.full(ag.cfg.n_opcoes, 1.0 / ag.cfg.n_opcoes, dtype=np.float32)
    assert lote["zeta"][0] == pytest.approx(
        np.tile(np.eye(ag.cfg.n_opcoes, dtype=np.float32)[0], (16, 1))), \
        "o buffer tem que guardar a crença **de antes** do passo"

    # a crença gravada no passo seguinte a cada `done` é a única que precisa ser exata
    t, n = np.nonzero(lote["done"][:-1])
    assert t.size > 0
    assert lote["zeta"][t + 1, n] == pytest.approx(
        np.tile(uniforme, (t.size, 1)), abs=1e-6), \
        "a crença atravessou a morte da cobra"


def test_a_deterministic_transition_moves_the_whole_belief():
    """A atualização de `ζ` é a do filtro para a frente. Com `π_ψ` apontando toda opção
    para a mesma, a crença tem que colapsar nela em um passo."""
    Z = 3
    pol = PoliticaComOpcoes(modelo=None, n_opcoes=Z)
    pol._garante(2)
    pol._pi_a = np.full((2, Z, 3), 1 / 3, dtype=np.float32)
    lz = np.full((2, Z, 3, Z), -10.0, dtype=np.float32)
    lz[:, :, :, 2] = 10.0
    pol._lz = lz
    pol.apos_passo(np.zeros(2, np.int32), np.zeros(2, bool))
    assert pol.zeta[:, 2] == pytest.approx(np.ones(2), abs=1e-3)


# ============================================================ a política avaliada
def test_the_evaluated_policy_is_the_marginal_and_not_a_subpolicy():
    """O protocolo é `argmax`. O argmax do marginal é a ação que a política de fato
    escolhe; o argmax de qualquer `π_θ(·|s,z)` isolada seria outra política — e o número
    publicado seria de um agente que nunca jogou."""
    import tensorflow as tf

    ag = SOAP(cfg())
    obs, mask = ag.env.reset()
    pol = ag.politica()
    saida = pol(obs, mask)

    la, _, _ = ag.model(tf.convert_to_tensor(obs), training=False)
    la = np.where(mask[:, None, :], la.numpy(), -1e9)
    e = np.exp(la - la.max(-1, keepdims=True))
    pi_a = e / e.sum(-1, keepdims=True)
    marginal = np.einsum("nz,nza->na", pol.zeta, pi_a)
    esperado = np.where(mask, np.log(marginal + 1e-12), -1e9)
    assert saida == pytest.approx(esperado, abs=1e-4)


def test_the_evaluation_policy_never_scores_a_masked_action():
    ag = SOAP(cfg())
    obs, mask = ag.env.reset()
    mask[:, 0] = False
    saida = ag.politica()(obs, mask)
    assert (saida[:, 0] <= -1e8).all()


def test_the_evaluation_policy_carries_the_belief_across_steps():
    ag = SOAP(cfg())
    pol = ag.politica()
    obs, mask = ag.env.reset()
    pol(obs, mask)
    antes = pol.zeta.copy()
    pol.apos_passo(np.ones(obs.shape[0], np.int32), np.zeros(obs.shape[0], bool))
    assert not np.allclose(antes, pol.zeta), "ζ não avançou — a política ficou sem memória"


def test_the_best_checkpoint_can_play():
    """Diferente do DreamerV3, aqui a política inteira mora num modelo só — então o
    checkpoint `best` joga sem truque nenhum, e `avaliar_melhor` mede o que promete."""
    ag = SOAP(cfg())
    outro = build_option_actor_critic(10, "resnet_tiny", n_opcoes=ag.cfg.n_opcoes)
    pol = ag.politica_do_modelo(outro)
    obs, mask = ag.env.reset()
    assert pol(obs, mask).shape == (obs.shape[0], 3)


# ==================================================================== o agente
def test_soap_is_its_own_algorithm_in_the_arena():
    ag = SOAP(cfg())
    assert ag.algo == "soap"
    assert "soap" in ORDEM_ALGORITMOS
    assert familia_de("soap") == "política"
    cores = cores_por_algoritmo({"ppo", "soap"})
    assert cores["soap"] != cores["ppo"]


def test_soap_trains():
    ag = SOAP(cfg())
    for _ in range(3):
        s = ag.iterate()
    for chave in ("pg", "vf", "ent", "kl", "lr", "opcao_persistencia",
                  "opcao_divergencia", "opcao_uso_entropia"):
        assert np.isfinite(s[chave]), f"{chave} virou {s[chave]}"
    assert s["ent"] > 0
    assert 0.0 <= s["alpha_medio"] <= 1.0


def test_the_options_start_indistinguishable_and_the_diagnostic_says_so():
    """No início as `Z` sub-políticas são quase iguais — os logits nascem com ganho 0,01.
    O diagnóstico de colapso tem que **reportar isso**, porque é exatamente o estado que
    ele precisa distinguir mais tarde: se `opcao_divergencia` continuar aqui no fim do
    treino, as opções existem no papel e não no comportamento."""
    _, s = SOAP(cfg()).collect()
    assert s["opcao_divergencia"] < 0.05
    assert s["opcao_uso_entropia"] == pytest.approx(s["opcao_uso_max"], abs=1e-3)


def test_the_responsibilities_form_a_distribution():
    """`w(z,z') = ζ(z)·p_velho(a,z'|s,z)/α` soma 1 sobre os pares. É o que torna a perda
    uma média ponderada de perdas de PPO em vez de uma soma de escala arbitrária — e é
    fácil de quebrar trocando `α` por outra normalização."""
    ag = SOAP(cfg())
    lote, _ = ag.collect()
    conj = lote["pi_a"][:, :, :, None] * lote["pi_z"]
    w = lote["zeta"][:, :, :, None] * conj / lote["alpha"][:, :, None, None]
    assert w.sum(axis=(-1, -2)) == pytest.approx(
        np.ones(w.shape[:2]), abs=1e-4)


def test_the_hunger_truncation_bootstraps_each_option_separately():
    """Fome é truncamento, não terminação — e o crítico é por opção, então o valor do
    estado final também é. Um bootstrap único ensinaria a todas as opções o valor terminal
    de uma delas."""
    ag = SOAP(cfg(num_envs=8))
    info = {"trunc_idx": np.array([0, 2]), "final_obs": None, "final_mask": None}
    rew = np.zeros(8, dtype=np.float32)
    v_final = np.array([[10.0, 20.0, 30.0, 40.0],
                        [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    saidas = [ag.bootstrap_truncados(info, rew, v_final[:, z], ag.cfg.gamma)
              for z in range(ag.cfg.n_opcoes)]
    assert saidas[0][0] == pytest.approx(10.0 * ag.cfg.gamma)
    assert saidas[3][0] == pytest.approx(40.0 * ag.cfg.gamma)
    assert saidas[0][1] == 0.0, "só os truncados recebem bootstrap"


def test_a_different_option_count_marks_the_variant():
    """`load_all` agrupa por `(algo, variant, seed)`. Sem a marca, o controle de uma opção
    dividiria identidade com o SOAP oficial e as duas virariam uma curva só."""
    assert SOAP(cfg()).variant == "resnet_tiny"
    assert SOAP(cfg(n_opcoes=1)).variant.endswith("+op1")
    assert SOAP(cfg(ent_opcao_coef=0.01)).variant.endswith("+entz0.01")


def test_reloading_the_model_rebuilds_the_optimizer():
    ag = SOAP(cfg())
    ag.iterate()
    antigo = ag.optimizer
    ag.on_model_reloaded()
    assert ag.optimizer is not antigo
    ag.iterate()


def test_the_gif_advances_a_policy_with_memory():
    """O `quadros_do_episodio` tem que respeitar o mesmo contrato que `snakeai.eval`.

    Sem isto, o GIF era gravado com o estado interno congelado no valor inicial: o agente
    do vídeo não era o agente da curva. Vale para o SOAP e valia, sem que ninguém notasse,
    para o DreamerV3.
    """
    from snakeai.env.render import quadros_do_episodio

    class Espia:
        def __init__(self):
            self.chamadas = 0

        def __call__(self, obs, mask):
            return np.where(mask, 0.0, -1e9).astype(np.float32)

        def apos_passo(self, acoes, done):
            self.chamadas += 1

    espia = Espia()
    quadros, _, _ = quadros_do_episodio(espia, max_steps=20)
    assert espia.chamadas == len(quadros) - 1
