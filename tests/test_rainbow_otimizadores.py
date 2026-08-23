"""Rainbow como algoritmo próprio, e o eixo de otimizadores que sucedeu o K-FAC."""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest

from snakeai.agents import DQN, DQNConfig, Rainbow, RainbowConfig
from snakeai.otimizadores import LR_SUGERIDO, OTIMIZADORES, cria_otimizador
from snakeai.plot import ORDEM_ALGORITMOS, cores_por_algoritmo


def rb(**kw):
    base = dict(net="resnet_tiny", num_envs=8, batch_size=16, memory_size=2000,
                warmup_steps=0, learn_every=2, total_steps=2000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return RainbowConfig(**base)


# ------------------------------------------------------------------- Rainbow
def test_rainbow_turns_on_all_six_components():
    """A composição canônica mora no código, não na cabeça de quem configura."""
    c = RainbowConfig()
    assert Rainbow.componentes(c) == {
        "double": True, "dueling": True, "per": True,
        "noisy": True, "n_steps": True, "c51": True,
    }


def test_rainbow_is_its_own_algorithm_in_the_arena():
    """Como variante de DQN ele pegaria a cor do DQN e viraria um rótulo ilegível."""
    ag = Rainbow(rb())
    assert ag.algo == "rainbow"
    assert ag.variant == "completo"
    assert "rainbow" in ORDEM_ALGORITMOS
    cores = cores_por_algoritmo({"ppo", "dqn", "rainbow"})
    assert cores["rainbow"] != cores["dqn"]


def test_rainbow_trains():
    ag = Rainbow(rb())
    for _ in range(4):
        stats = ag.iterate()
    assert np.isfinite(stats["loss"])
    assert stats["epsilon"] == 0.0, "a exploração do Rainbow vem das noisy nets"


def test_rainbow_without_exploration_is_refused():
    """`noisy=False` e `eps=0` deixa o agente sem exploração nenhuma — erro, não surpresa."""
    with pytest.raises(ValueError, match="não explora"):
        Rainbow(rb(noisy=False))


# --------------------------------------------------------------- suporte do C51
SUPORTES = [(-20.0, 20.0, 51), (-20.0, 20.0, 101), (-10.0, 10.0, 51),
            (-2.0, 60.0, 51), (-1.0, 1.0, 51), (-5.0, 25.0, 201)]


@pytest.mark.parametrize("v_min,v_max,n_atoms", SUPORTES)
def test_c51_projection_stays_inside_the_support(v_min, v_max, n_atoms):
    """A projeção categórica não pode indexar fora do suporte.

    `tz` é preso a `[v_min, v_max]`, mas `delta_z` é float32 e a divisão devolve
    50,0000476 para o átomo do topo em `[-20, 20]` com 51 átomos — `ceil` dá 51 e o
    `np.add.at` estoura. O bug era **latente**: a aritmética de `[-2, 60]` arredondava
    para baixo e escondia o defeito, e trocar o suporte por qualquer um dos canônicos
    (inclusive o `[-10, 10]` do Atari que a docstring de `suporte_c51` cita) derrubava o
    treino com `IndexError`. Este teste varre suportes de propósito.
    """
    ag = Rainbow(rb(v_min=v_min, v_max=v_max, n_atoms=n_atoms))
    n = 8
    lote = {
        "obs": np.zeros((n, 10, 10, 5), np.float32),
        "next_obs": np.zeros((n, 10, 10, 5), np.float32),
        "act": np.zeros(n, np.int64),
        # os extremos são o caso perigoso: alvo exatamente em v_min e em v_max
        "rew": np.array([v_max, v_min, 0.0, 1.0, -1.0, v_max, v_min, 0.5], np.float32),
        "done": np.array([1, 1, 0, 0, 0, 0, 0, 1], np.float32),
        "next_mask": np.ones((n, 3), bool),
    }
    alvo = ag._alvo(lote)
    assert alvo.shape == (n, n_atoms)
    assert np.isfinite(alvo).all()
    # a projeção redistribui massa, nunca cria nem destrói
    np.testing.assert_allclose(alvo.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize("v_min,v_max,n_atoms", SUPORTES)
def test_c51_starts_unbiased(v_min, v_max, n_atoms):
    """O `Q` inicial do C51 é o **ponto médio do suporte**, e isso é uma armadilha.

    Com os logits em ~0 a softmax é uniforme, então `Q = média(suporte)`. Um suporte
    assimétrico faz todo estado nascer valendo o ponto médio — e esse valor é um ponto
    fixo do bootstrap, porque o alvo de uma transição não terminal é `r + γⁿ·Q` que já é
    aproximadamente `Q`. Com `[-2, 60]` isso valia **+29**: medido, o `Q` médio ficou
    preso em +28,6 por 120 mil passos enquanto uma maçã valia +1 sobre essa linha de base.

    O teste exige que o suporte configurado seja simétrico o bastante para nascer perto de
    zero. Se alguém alargar `v_max` "para caber o retorno máximo" sem mexer em `v_min`,
    isto falha aqui em vez de falhar depois de 2 h de GPU.
    """
    if abs(v_min + v_max) > 1e-6:
        pytest.skip("suporte assimétrico: incluído só na varredura de projeção")
    ag = Rainbow(rb(v_min=v_min, v_max=v_max, n_atoms=n_atoms))
    obs = np.zeros((4, 10, 10, 5), np.float32)
    q = float(np.asarray(ag._q_valores(ag.model, keras.ops.convert_to_tensor(obs))).mean())
    assert abs(q) < 0.5, f"Q inicial {q:.2f} — o agente nasce otimista e o bootstrap trava"


def test_the_default_rainbow_support_is_symmetric_and_covers_a_perfect_game():
    """Os dois requisitos do suporte, que puxam em direções opostas.

    Simétrico, para o `Q` inicial nascer em zero; e largo o bastante para o retorno de um
    jogo perfeito — 97 maçãs a ~10 passos cada com γ=0,995 rendem 20,3.
    """
    c = RainbowConfig()
    assert c.v_min + c.v_max == 0, "suporte assimétrico faz o Q inicial nascer viesado"
    retorno_perfeito = sum(c.gamma ** (10 * k) for k in range(97))
    assert c.v_max >= retorno_perfeito, f"v_max={c.v_max} < retorno perfeito {retorno_perfeito:.1f}"
    delta_z = (c.v_max - c.v_min) / (c.n_atoms - 1)
    assert delta_z <= 0.5, (
        f"delta_z={delta_z:.2f}: uma maçã vale {1/delta_z:.1f} átomos. O C51 canônico do "
        "Atari usa 0,4 com recompensas em ±1")


def test_rainbow_is_a_dqn_configuration():
    """Não é algoritmo novo: é a soma dos seis. Herdar deixa isso explícito no código."""
    assert issubclass(Rainbow, DQN)
    assert Rainbow.iterate is DQN.iterate
    assert Rainbow._alvo is DQN._alvo


# -------------------------------------------------------------- otimizadores
@pytest.mark.parametrize("nome", OTIMIZADORES)
def test_every_optimizer_builds(nome):
    opt = cria_otimizador(nome, 1e-3, clipnorm=1.0)
    assert isinstance(opt, keras.optimizers.Optimizer)


def test_unknown_optimizer_raises_with_the_list():
    with pytest.raises(ValueError, match="adamw"):
        cria_otimizador("kfac", 1e-3)


@pytest.mark.parametrize("nome", OTIMIZADORES)
def test_dqn_trains_with_every_optimizer(nome):
    """O eixo que substitui o K-FAC: a pergunta 'o otimizador importa?' agora roda."""
    cfg = DQNConfig(net="resnet_tiny", optimizer=nome, num_envs=8, batch_size=16,
                    memory_size=1000, warmup_steps=0, learn_every=2, total_steps=2000,
                    eval_every_steps=10**9, log_every_steps=10**9,
                    salvar_gif=False, salvar_grafico=False)
    ag = DQN(cfg)
    for _ in range(3):
        stats = ag.iterate()
    assert np.isfinite(stats["loss"])


def test_optimizer_appears_in_the_variant_name_only_when_it_is_not_the_default():
    """Senão toda variante ganharia um `+adam` que não informa nada."""
    assert DQN(DQNConfig(net="resnet_tiny", optimizer="adam")).variant == "base"
    assert DQN(DQNConfig(net="resnet_tiny", optimizer="lion")).variant == "base+lion"


def test_ppo_also_accepts_the_optimizer_axis():
    from snakeai.agents import PPO, PPOConfig

    ag = PPO(PPOConfig(net="resnet_tiny", optimizer="adamw", num_envs=8, rollout=4,
                       minibatches=1, epochs=1, salvar_gif=False, salvar_grafico=False))
    assert isinstance(ag.optimizer, keras.optimizers.AdamW)
    ag.iterate()


def test_lr_suggestions_reflect_the_step_geometry():
    """Comparar otimizadores com o mesmo LR mede quem tolera aquele LR, não o otimizador.

    O Lion dá passos de magnitude constante (só o sinal do momento), então precisa de LR
    muito menor; o SGD, sem escala adaptativa, precisa de muito maior.
    """
    assert LR_SUGERIDO["lion"] < LR_SUGERIDO["adam"] < LR_SUGERIDO["sgd"]
    assert set(LR_SUGERIDO) == set(OTIMIZADORES)


def test_kfac_is_not_in_this_axis_and_the_module_says_where_it_is():
    """O K-FAC existe (`snakeai/kfac.py`), mas não cabe nesta assinatura.

    `cria_otimizador(nome, lr)` não tem como entregar as ativações de entrada e os
    gradientes de pré-ativação de cada camada, que é do que o K-FAC vive. Deixar isso
    escrito no módulo evita que alguém "conserte" a ausência adicionando um
    `optimizer="kfac"` que silenciosamente não pré-condicionaria nada.
    """
    import snakeai.otimizadores as mod

    assert "kfac" not in OTIMIZADORES
    assert "K-FAC" in mod.__doc__ and "snakeai/kfac.py" in mod.__doc__
    assert "ACKTR" in mod.__doc__


def test_rainbow_keeps_epsilon_at_zero_by_default():
    """A composição canônica não muda: sem pedir, o Rainbow explora só por noisy nets."""
    ag = Rainbow(rb())
    assert ag.epsilon() == 0.0


def test_epsilon_under_noisy_needs_an_explicit_opt_in():
    """Pedir ε junto com noisy nets tem de ser possível — e explícito.

    Antes `epsilon()` devolvia zero incondicionalmente com `noisy=True`, então `eps_start`
    era um campo ignorado **em silêncio**: sem erro e sem efeito. Agora o padrão continua
    zero (a composição do paper não muda, e as ablações de DQN com `noisy` seguem sem ε),
    mas `eps_mesmo_com_noisy=True` liga os dois. É o botão que permite medir se a
    exploração é o gargalo restante do Rainbow neste ambiente.
    """
    sem = Rainbow(rb(eps_start=0.5, eps_end=0.05, eps_frac=0.5))
    assert sem.epsilon() == 0.0, "o padrão do Rainbow tem de continuar sem ε"

    com = Rainbow(rb(eps_start=0.5, eps_end=0.05, eps_frac=0.5, eps_mesmo_com_noisy=True))
    assert com.epsilon() == pytest.approx(0.5)
    com.global_step = int(com.cfg.total_steps * 0.5)
    assert com.epsilon() == pytest.approx(0.05)


# --------------------------------------------- a janela de n passos e o truncamento
def test_the_n_step_window_stops_at_the_episode_boundary():
    """A morte por fome é truncamento: `done=0` para o alvo, mas o episódio acabou.

    O buffer usava `done` como única marca de fim. Com `done=0` a fila não era esvaziada e
    as janelas seguintes somavam recompensas do episódio **seguinte**, com um `next_obs` de
    outra trajetória. `n_steps=1` é imune — cada janela é um passo — e por isso o DQN base
    nunca sentiu; o Rainbow usa `n_steps=3` e 90% dos episódios deste ambiente acabam por
    fome, então duas de cada três janelas de cada fronteira saíam contaminadas.
    """
    import numpy as np
    from snakeai.memory.replay import ReplayBuffer

    g, n = 0.9, 3
    recs = [1.0, 2.0, 4.0, 100.0, 200.0, 400.0]   # episódios A(0-2, fome no 2) e B(3-5)
    fim = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    buf = ReplayBuffer(100, (2, 2, 1), n_actions=3, n_steps=n, gamma=g, num_envs=1,
                       rng=np.random.default_rng(0))
    for k, r in enumerate(recs):
        buf.add_batch(np.full((1, 2, 2, 1), k, np.float32), np.zeros(1, np.int64),
                      np.array([r], np.float32), np.full((1, 2, 2, 1), k + 1, np.float32),
                      np.zeros(1, np.float32),          # done=0 sempre: é truncamento
                      np.ones((1, 3), bool), fim=np.array([fim[k]], np.float32))

    guardado = {int(buf.obs[i][0, 0, 0]): (float(buf.rew[i]), int(buf.n_real[i]))
                for i in range(len(buf))}
    assert guardado[0] == pytest.approx((1.0 + g * 2.0 + g * g * 4.0, 3))
    assert guardado[1] == pytest.approx((2.0 + g * 4.0, 2)), "a janela atravessou a fronteira"
    assert guardado[2] == pytest.approx((4.0, 1)), "a janela atravessou a fronteira"
    assert guardado[3][0] == pytest.approx(100.0 + g * 200.0 + g * g * 400.0)


def test_a_short_window_is_discounted_by_its_real_length():
    """`γ**n_real`, não `γ**n_steps`.

    As janelas esvaziadas na fronteira são mais curtas, e como a fome bootstrapa de
    verdade (`done=0`) o desconto errado desloca o alvo em vez de ser anulado pelo `done`.
    """
    import numpy as np
    from snakeai.agents import Rainbow

    ag = Rainbow(rb(n_steps=3, gamma=0.9))
    nn = 4
    lote = {"obs": np.zeros((nn, 10, 10, 5), np.float32),
            "next_obs": np.zeros((nn, 10, 10, 5), np.float32),
            "act": np.zeros(nn, np.int64), "rew": np.zeros(nn, np.float32),
            "done": np.zeros(nn, np.float32), "next_mask": np.ones((nn, 3), bool),
            "n_real": np.array([3, 2, 1, 3], np.int32)}
    alvo_curto = ag._alvo(lote)
    lote["n_real"] = np.array([3, 3, 3, 3], np.int32)
    alvo_cheio = ag._alvo(lote)
    # as linhas 1 e 2 têm janela curta: o alvo tem de ser diferente
    assert not np.allclose(alvo_curto[1], alvo_cheio[1])
    assert not np.allclose(alvo_curto[2], alvo_cheio[2])
    # a linha 0 tem janela cheia: idêntica
    np.testing.assert_allclose(alvo_curto[0], alvo_cheio[0], atol=1e-6)


def test_the_best_checkpoint_can_actually_play(tmp_path):
    """`avaliar_melhor()` recarrega o `best` e joga com ele — os dois passos têm de valer.

    Duas falhas moravam aqui, as duas só visíveis **no fim do treino**: o `Lambda` que
    impedia recarregar (§2.14) e o `keras_policy` genérico, que assume saída
    `(lote, ações)` e quebra com os `(lote, ações, átomos)` do C51 (§2.17). Este teste faz
    o caminho inteiro — salvar, recarregar, montar a política, jogar um lote.
    """
    import numpy as np
    import keras

    ag = Rainbow(rb())
    caminho = str(tmp_path / "best.keras")
    ag.model.save(caminho)
    recarregado = keras.models.load_model(caminho)          # §2.14
    pol = ag.politica_do_modelo(recarregado)                # §2.17
    obs = np.zeros((7, 10, 10, 5), np.float32)
    mask = np.ones((7, 3), bool)
    logits = np.asarray(pol(obs, mask), dtype=np.float32)
    assert logits.shape == (7, 3), f"a política devolveu {logits.shape}, não (lote, ações)"
    assert np.isfinite(logits).all()
