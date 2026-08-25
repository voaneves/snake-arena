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
    # a fila de n passos só emite depois de `n_steps` passos por ambiente, e o padrão
    # agora é 20 (§2.25) — quatro iterações não enchiam nem a fila
    for _ in range(2 + ag.cfg.n_steps // max(1, ag.cfg.learn_every)):
        stats = ag.iterate()
    assert np.isfinite(stats["loss"])
    assert stats["epsilon"] == 0.0, "a exploração do Rainbow vem das noisy nets"


def test_a_deviation_from_the_canonical_composition_marks_the_variant():
    """A composição canônica mora no código; a **identidade da execução** também tem de.

    Sem isto, a execução de `n_steps=3` só se distinguia se quem a rodou lembrasse de
    passar `variant=` na mão. Esquecer faria as duas dividirem `(algo, variant, seed)` e
    virarem uma curva só na arena — com a de 0,57 arrastando a de 65,43 sem deixar rastro.
    """
    assert Rainbow(rb()).variant == "completo"
    assert Rainbow(rb(n_steps=3)).variant == "completo+n3"
    assert Rainbow(rb(per=False)).variant == "completo+sem_per"
    assert Rainbow(rb(n_steps=3, dueling=False)).variant == "completo+n3+sem_dueling"


def test_an_explicit_variant_still_wins():
    """Os nomes que as execuções de agosto receberam à mão continuam valendo — o histórico
    não se move quando a marcação automática entra."""
    assert Rainbow(rb(), variant="completo+n3").variant == "completo+n3"


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


# ------------------------------------------- o eixo de atualizações (§2.18-2.20)
def test_the_update_counter_is_not_double_counted(tmp_path):
    """`meta["atualizacoes"]` tem de ser o número real de passos de gradiente.

    O `DQN` mantinha um contador próprio chamado `_atualizacoes` — o mesmo nome do
    contador do `AgentBase` —, e como `AgentBase.train` também soma o `atualizacoes` que
    `iterate()` devolve, o atributo era incrementado **duas vezes por iteração**. Toda
    execução de DQN e Rainbow gravou exatamente **2,00×** o número real; o PPO, que não tem
    contador próprio, gravou 1,00×. O metadado é o eixo do `ORCAMENTO_DE_GRADIENTE.md`, e
    o viés valia para uma família só.
    """
    from snakeai.agents import DQN, DQNConfig

    ag = DQN(DQNConfig(net="resnet_tiny", num_envs=8, batch_size=16, memory_size=2000,
                       warmup_steps=0, learn_every=2, total_steps=2000,
                       eval_every_steps=10 ** 9, eval_episodes=20, eval_envs=10,
                       log_every_steps=10 ** 9, salvar_gif=False, salvar_grafico=False,
                       runs_dir=str(tmp_path), ckpt_dir=str(tmp_path)))
    reais = []
    original = ag._passo_treino
    ag._passo_treino = lambda *a, **kw: (reais.append(1), original(*a, **kw))[1]
    registro = ag.train(verbose=False)
    gravado = registro.record.meta["atualizacoes"]
    assert gravado == len(reais), f"gravado {gravado} para {len(reais)} passos reais"


def test_per_priority_is_the_kl_not_the_cross_entropy():
    """A prioridade da PER é a surpresa, e no C51 a surpresa é a KL — não a CE.

    `CE = KL(alvo‖pred) + H(alvo)`, e com 121 átomos `H` fica preso perto de `ln 121`.
    Medido antes da correção: `corr(CE, KL) = −0,9066`, massa dos 10% maiores em 0,100
    (uniforme), e a amostra de **maior** erro do lote recebendo prioridade **menor** que a
    de menor erro. A PER não estava só inerte — estava invertida.
    """
    import numpy as np

    ag = Rainbow(rb())
    n = 64
    alvo = np.zeros((n, ag.cfg.n_atoms), np.float64)
    rng = np.random.default_rng(0)
    # metade dos alvos concentrados (H baixo), metade difusos (H alto)
    for i in range(n):
        if i % 2:
            alvo[i, rng.integers(ag.cfg.n_atoms)] = 1.0
        else:
            alvo[i] = 1.0 / ag.cfg.n_atoms
    H = -(alvo * np.log(np.clip(alvo, 1e-12, None))).sum(-1)
    kl_verdadeiro = np.linspace(0.0, 3.0, n)
    ce = kl_verdadeiro + H                       # é o que o grafo devolvia

    prio = ag._prioridades(ce, alvo)
    np.testing.assert_allclose(prio, kl_verdadeiro, atol=1e-8)
    assert np.corrcoef(prio, kl_verdadeiro)[0, 1] > 0.999
    assert (prio >= 0).all(), "prioridade negativa quebra a sum-tree"
    # e a CE crua estaria anticorrelacionada, que é o defeito que isto conserta
    assert np.corrcoef(ce, kl_verdadeiro)[0, 1] < np.corrcoef(prio, kl_verdadeiro)[0, 1]


def test_per_priority_in_the_scalar_branch_is_the_absolute_error():
    """Fora do C51 a prioridade é `|δ|`, não a perda de Huber.

    `(δ²/2)**α ∝ |δ|**2α`, então usar a perda dobrava o expoente efetivo da PER na região
    quadrática — `per_alpha=0,6` virava 1,2 — e a ablação media um `α` que não era o do
    `config`. Não afeta o Rainbow, afeta toda variante `per=True, n_atoms=0`.
    """
    import numpy as np
    import tensorflow as tf
    from snakeai.agents import DQN, DQNConfig

    ag = DQN(DQNConfig(net="resnet_tiny", num_envs=4, batch_size=8, memory_size=500,
                       n_atoms=0, per=True, dueling=False, noisy=False,
                       warmup_steps=0, total_steps=1000))
    n = 8
    obs = np.zeros((n, 10, 10, 5), np.float32)
    act = np.zeros(n, np.int32)
    q = np.asarray(ag._q_valores(ag.model, tf.convert_to_tensor(obs)))[:, 0]
    alvo = (q + np.linspace(-2.0, 2.0, n)).astype(np.float32)
    _, surpresa = ag._passo_treino(
        tf.convert_to_tensor(obs), tf.convert_to_tensor(act),
        tf.convert_to_tensor(alvo), tf.convert_to_tensor(np.ones(n, np.float32)), False)
    esperado = np.abs(alvo - q)
    np.testing.assert_allclose(np.asarray(surpresa), esperado, atol=1e-4)


def test_the_target_network_syncs_often_enough():
    """`target_update` contado no orçamento real, não no dobrado.

    5 M passos compram ~18.500 atualizações reais (`num_envs × learn_every = 256` passos
    por atualização). Com `target_update=1.000` isso dava 18,6 sincronizações no treino
    inteiro — o DQN da Nature faz ~1.250. O piso aqui é o do DQN base do repositório, que
    decola aos 750 k.
    """
    c = RainbowConfig()
    por_atualizacao = c.num_envs * c.learn_every
    atualizacoes = c.total_steps / por_atualizacao
    sincronizacoes = atualizacoes / c.target_update
    assert sincronizacoes >= 50, (
        f"{sincronizacoes:.0f} sincronizações em {atualizacoes:,.0f} atualizações reais — "
        "o valor propaga vezes demais poucas")


def test_the_entry_priority_does_not_ratchet_forever():
    """`max_prioridade` é um máximo **recente**, não histórico.

    A referência usa o máximo histórico, e no Atari isso é inofensivo porque a recompensa é
    cortada em ±1 e o erro de TD tem teto. Aqui a prioridade é a KL do C51 (§2.19) e não tem
    teto: um pico isolado fixaria o piso de toda transição nova para sempre. Medido antes do
    decaimento, `max_prioridade` subia de 4,21 para 4,90 em 250 iterações sem nunca voltar.
    """
    import numpy as np
    from snakeai.memory.replay import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(64, (2, 2, 1), n_actions=3, num_envs=1,
                                  rng=np.random.default_rng(0))
    for k in range(8):
        buf.add_batch(np.zeros((1, 2, 2, 1), np.float32), np.zeros(1, np.int64),
                      np.zeros(1, np.float32), np.zeros((1, 2, 2, 1), np.float32),
                      np.zeros(1, np.float32), np.ones((1, 3), bool))

    buf.update_priorities([0], [50.0])              # um pico isolado
    assert buf.max_prioridade >= 50.0
    for _ in range(400):                            # regime normal depois dele
        buf.update_priorities([1], [0.05])
    assert buf.max_prioridade < 1.0, (
        f"o pico de 50 ainda sustenta o máximo em {buf.max_prioridade:.2f} — "
        "toda transição nova entraria fixada nele")
    # e um regime de erro alto de verdade continua sustentando
    for _ in range(50):
        buf.update_priorities([2], [8.0])
    assert buf.max_prioridade >= 8.0


def test_the_multistep_horizon_reaches_the_reward():
    """`n_steps` tem de alcançar a recompensa, e 3 não alcança neste ambiente.

    O agente gasta ~12 passos por maçã. Com uma janela de 3, a decisão que o levou até a
    comida sai do retorno antes de a recompensa entrar, e a atribuição de crédito passa a
    depender só do bootstrap — que depende das sincronias do alvo, dezenas num treino
    inteiro. Medido: com `n_steps=3` o agente ficava um milhão de passos em 100% de morte
    por fome e decolava aos ~1,85 M; com 20, decola aos ~700 k.
    """
    c = RainbowConfig()
    passos_por_maca = 12                    # medido nas execuções que aprenderam
    assert c.n_steps >= passos_por_maca, (
        f"n_steps={c.n_steps} não alcança a maçã, que chega ~{passos_por_maca} passos depois")
    # e o suporte precisa cobrir o retorno acumulado da janela
    retorno_max = c.n_steps * 1.0 + 2.0     # comer a cada passo, mais o bônus de vitória
    assert c.v_max >= retorno_max, (
        f"v_max={c.v_max} < retorno máximo de {c.n_steps} passos ({retorno_max})")


def test_the_defaults_are_the_configuration_that_was_measured():
    """Os padrões são a execução que funcionou, não a que a referência sugere.

    `learn_every=1` (reamostragem 8,0, igual ao `Kaixhin`) e `target_update=1000` são mais
    fiéis ao canônico, e chegaram a ser o padrão. Voltaram para 4 e 250 porque a execução
    que decolou aos 700 k rodou assim, e trocar as duas junto com `n_steps=20` mediria a
    soma. As duas continuam registradas como a próxima ablação em §2.23.
    """
    c = RainbowConfig()
    assert (c.learn_every, c.target_update) == (4, 250)
    reamostragem = c.batch_size / (c.num_envs * c.learn_every)
    assert reamostragem == 2.0, "a reamostragem mudou sem a ablação que a justifica"
