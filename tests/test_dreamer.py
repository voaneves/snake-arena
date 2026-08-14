"""DreamerV3 — o modelo do mundo, as transformações, e a avaliação com memória.

Um agente baseado em modelo tem uma classe de falha que os outros não têm: o sonho pode
divergir do jogo sem que nada quebre. O ator então aprende uma política ótima para um mundo
que não existe, a curva fica achatada, e a conclusão errada ("modelo do mundo não funciona
neste ambiente") parece medida. Os testes daqui atacam essa classe: cada peça é conferida
contra a forma fechada, e a avaliação é conferida contra o esquecimento do latente.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from snakeai.agents import DreamerV3, DreamerV3Config
from snakeai.agents.dreamerv3 import PoliticaRecorrente, _percentil
from snakeai.memory.sequencia import SequenceBuffer
from snakeai.nets.dreamer import (PRESETS_DREAMER, CelulaRecorrente,
                                  amostra_straight_through, bins_simetricos,
                                  build_decoder, build_encoder, de_two_hot, symexp,
                                  symlog, two_hot, unimix)
from snakeai.plot import ORDEM_ALGORITMOS, familia_de


def cfg(**kw):
    base = dict(preset="dreamer_tiny", num_envs=8, batch_size=4, seq_len=8,
                memory_size=200, warmup_steps=0, horizonte=5, collect_steps=8,
                eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return DreamerV3Config(**base)


# ------------------------------------------------------------------- symlog
def test_symlog_roundtrips():
    x = tf.constant([-1e6, -50.0, -1.0, 0.0, 0.3, 1.0, 97.0, 1e6])
    np.testing.assert_allclose(symexp(symlog(x)).numpy(), x.numpy(), rtol=1e-4)


def test_symlog_compresses_the_range_that_matters_here():
    """O retorno em Snake vai de ~0 a ~50 durante o treino. `symlog` põe os dois regimes
    na mesma ordem de grandeza, que é o que dispensa reajustar o learning rate."""
    assert float(symlog(tf.constant(50.0))) / float(symlog(tf.constant(1.0))) < 6.0


# ------------------------------------------------------------------ two-hot
def test_two_hot_is_a_distribution():
    bins = bins_simetricos(41)
    t = two_hot(tf.constant([-3.7, 0.0, 2.4, 19.9]), bins)
    np.testing.assert_allclose(tf.reduce_sum(t, axis=-1).numpy(), 1.0, atol=1e-5)
    assert int(tf.reduce_sum(tf.cast(t > 0, tf.int32))) <= 8, "no máximo dois bins por valor"


def test_two_hot_recovers_the_value():
    """A ida e a volta têm que ser exatas; senão a recompensa prevista vem enviesada."""
    bins = bins_simetricos(41)
    x = tf.constant([-11.3, -1.0, 0.0, 0.5, 7.7, 15.2])
    logits = tf.math.log(two_hot(x, bins) + 1e-12)
    np.testing.assert_allclose(de_two_hot(logits, bins).numpy(), x.numpy(), atol=1e-3)


def test_two_hot_on_a_bin_is_one_hot():
    bins = bins_simetricos(41)
    t = two_hot(tf.gather(bins, [7, 20, 33]), bins).numpy()
    assert (t.max(axis=-1) > 0.999).all()


def test_two_hot_clips_instead_of_wrapping():
    """Um valor fora da grade tem que saturar no bin extremo. Se ele vazasse, o alvo do
    crítico apontaria para o outro lado da grade — erro grande e silencioso."""
    bins = bins_simetricos(41)
    t = two_hot(tf.constant([1e4]), bins).numpy()[0]
    assert t.argmax() == 40


# ----------------------------------------------------------------- unimix / ST
def test_unimix_keeps_the_shape_and_kills_minus_inf():
    lg = tf.constant([[-1e5, 0.0, 1e5, 0.0] * 2])
    out = unimix(lg, grupos=2, classes=4, mistura=0.01)
    assert out.shape == lg.shape
    assert np.isfinite(out.numpy()).all()
    p = tf.nn.softmax(tf.reshape(out, [1, 2, 4]), axis=-1).numpy()
    assert p.min() >= 0.01 / 4 * 0.99, "toda classe fica com pelo menos ~mistura/K"


def test_straight_through_is_one_hot_forward_and_differentiable_backward():
    """As duas metades do truque, medidas separadamente."""
    lg = tf.constant(np.random.randn(6, 4 * 5), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(lg)
        z, _ = amostra_straight_through(lg, grupos=4, classes=5)
        alvo = tf.reduce_sum(z * 2.0)
    zz = z.numpy().reshape(6, 4, 5)
    np.testing.assert_allclose(zz.sum(axis=-1), 1.0, atol=1e-5)
    assert set(np.unique(zz).round(5)) <= {0.0, 1.0}, "o forward tem que ser exatamente one-hot"
    g = tape.gradient(alvo, lg)
    assert g is not None and np.abs(g.numpy()).sum() > 0, "sem gradiente, o encoder não treina"


# --------------------------------------------------------------------- redes
def test_encoder_and_decoder_round_trip_the_shape():
    enc = build_encoder(10, canais=24)
    dim = enc.output_shape[-1]
    dec = build_decoder(dim, 10, canais=24)
    x = tf.zeros([3, 10, 10, 5])
    assert dec(enc(x)).shape == (3, 10, 10, 5)


def test_recurrent_cell_is_serializable_and_keeps_the_width():
    c = CelulaRecorrente(16)
    h = c(tf.zeros([4, 7]), tf.zeros([4, 16]))
    assert h.shape == (4, 16)
    assert CelulaRecorrente.from_config(c.get_config()).unidades == 16


@pytest.mark.parametrize("preset", sorted(PRESETS_DREAMER))
def test_every_preset_builds_and_trains(preset):
    ag = DreamerV3(cfg(preset=preset, batch_size=2, seq_len=6, horizonte=3))
    ag.iterate()
    assert np.isfinite(ag.iterate()["modelo"])


# ------------------------------------------------------------------- memória
def test_sequence_buffer_shapes():
    b = SequenceBuffer(4, 20, (10, 10, 5), 3)
    for _ in range(12):
        b.add(np.zeros((4, 10, 10, 5), np.float32), np.zeros(4, np.int32),
              np.zeros(4, np.float32), np.ones(4, np.float32),
              np.zeros(4, bool), np.ones((4, 3), bool))
    lote = b.sample(5, 6)
    assert lote["obs"].shape == (5, 6, 10, 10, 5)
    assert lote["act"].shape == (5, 6) and lote["mask"].shape == (5, 6, 3)


def test_sampled_window_never_crosses_the_ring_head():
    """A janela que atravessa a cabeça do anel cola o passo mais novo no mais velho.

    O modelo aprende essa descontinuidade como se fosse física do jogo — e nada avisa.
    O teste marca cada passo com um número crescente e exige que toda janela amostrada
    seja uma sequência contígua desses números.
    """
    b = SequenceBuffer(2, 10, (1,), 3, seed=0)
    for t in range(37):  # dá a volta no anel três vezes e sobra
        b.add(np.full((2, 1), t, np.float32), np.zeros(2, np.int32),
              np.zeros(2, np.float32), np.ones(2, np.float32),
              np.zeros(2, bool), np.ones((2, 3), bool))
    for _ in range(200):
        seq = b.sample(8, 5)["obs"][:, :, 0]
        d = np.diff(seq, axis=1)
        assert (d == 1).all(), f"janela descontínua: {seq[np.where(d != 1)[0][0]]}"


def test_buffer_refuses_a_sequence_longer_than_it_has():
    b = SequenceBuffer(2, 50, (1,), 3)
    b.add(np.zeros((2, 1), np.float32), np.zeros(2, np.int32), np.zeros(2, np.float32),
          np.ones(2, np.float32), np.zeros(2, bool), np.ones((2, 3), bool))
    assert not b.pronto(8)
    with pytest.raises(ValueError, match="sequência pede"):
        b.sample(2, 8)


def test_episode_boundaries_are_kept_not_avoided():
    """Cruzar fim de episódio é permitido — `first` é que marca onde zerar o latente."""
    b = SequenceBuffer(1, 20, (1,), 3, seed=1)
    for t in range(20):
        b.add(np.zeros((1, 1), np.float32), np.zeros(1, np.int32), np.zeros(1, np.float32),
              np.array([0.0 if t == 9 else 1.0], np.float32),
              np.array([t == 10]), np.ones((1, 3), bool))
    achou = any(b.sample(16, 6)["first"].any() for _ in range(20))
    assert achou, "as janelas nunca incluíram um começo de episódio"


# ------------------------------------------------------------------- retornos
def test_lambda_returns_reduce_to_td_zero_when_lambda_is_zero():
    ag = DreamerV3(cfg(lam=0.0, gamma=0.9))
    rew = tf.constant([[1.0], [2.0], [3.0]])
    cont = tf.ones([3, 1])
    v = tf.constant([[10.0], [20.0], [30.0], [40.0]])
    R = ag._retornos_lambda(rew, cont, v).numpy()[:, 0]
    np.testing.assert_allclose(R, [1 + .9 * 20, 2 + .9 * 30, 3 + .9 * 40], rtol=1e-5)


def test_lambda_returns_stop_at_termination():
    """`cont=0` tem que cortar o retorno ali. Sem isso o ator aprende que morrer é neutro."""
    ag = DreamerV3(cfg(lam=1.0, gamma=0.9))
    rew = tf.constant([[1.0], [5.0]])
    cont = tf.constant([[0.0], [1.0]])
    v = tf.constant([[10.0], [20.0], [30.0]])
    R = ag._retornos_lambda(rew, cont, v).numpy()[:, 0]
    assert R[0] == pytest.approx(1.0), "com cont=0 o retorno é só a recompensa"


def test_percentile_matches_numpy():
    x = tf.constant(np.random.default_rng(0).normal(size=(7, 11)), tf.float32)
    for q in (5.0, 50.0, 95.0):
        assert float(_percentil(x, q)) == pytest.approx(
            float(np.percentile(x.numpy(), q, method="lower")), abs=0.05)


# ---------------------------------------------------------------- a avaliação
def test_the_evaluation_policy_actually_remembers():
    """Se o latente não avançasse, a mesma observação daria sempre os mesmos logits — e o
    Dreamer seria medido como uma rede sem memória, ou seja, para baixo."""
    ag = DreamerV3(cfg())
    pol = ag.politica()
    obs, mask = ag.env.reset()
    primeiro = pol(obs, mask)
    pol.apos_passo(np.zeros(len(obs), np.int32), np.zeros(len(obs), bool))
    segundo = pol(obs, mask)
    assert not np.allclose(primeiro, segundo), "o estado recorrente não avançou"


def test_apos_passo_resets_the_latent_on_death():
    """Depois de `done`, a parte determinística do latente tem que voltar ao início.

    A comparação é em `h`, não nos logits: `z` é amostrado, então dois estados idênticos
    dão amostras diferentes. `h` é determinístico e é o que carrega a história.
    """
    ag = DreamerV3(cfg())
    pol = ag.politica()
    obs, mask = ag.env.reset()
    n = len(obs)
    pol(obs, mask)
    h_inicial = pol.h.numpy().copy()

    for _ in range(4):
        pol.apos_passo(np.zeros(n, np.int32), np.zeros(n, bool))
        pol(obs, mask)
    assert not np.allclose(pol.h.numpy(), h_inicial), "sem morte, `h` tem que ter andado"

    pol.apos_passo(np.zeros(n, np.int32), np.ones(n, bool))
    pol(obs, mask)
    np.testing.assert_allclose(pol.h.numpy(), h_inicial, atol=1e-5)


def test_evaluate_calls_the_hook_when_the_policy_has_one():
    from snakeai.eval import evaluate

    ag = DreamerV3(cfg())
    pol = ag.politica()
    chamadas = []
    original = pol.apos_passo

    def espia(acoes, done):
        chamadas.append(int(done.sum()))
        original(acoes, done)

    pol.apos_passo = espia
    stats, _ = evaluate(pol, episodes=20, num_envs=10, max_steps=2000)
    assert chamadas, "`evaluate` não chamou `apos_passo`"
    assert stats["episodes"] == 20


def test_evaluate_still_works_for_memoryless_policies():
    """A extensão do `evaluate` não pode exigir o gancho de quem não tem estado."""
    from snakeai.eval import evaluate

    def sem_memoria(obs, mask):
        return np.where(mask, 0.0, -1e9).astype(np.float32)

    assert not hasattr(sem_memoria, "apos_passo")
    stats, _ = evaluate(sem_memoria, episodes=20, num_envs=10, max_steps=2000)
    assert stats["episodes"] == 20


# ------------------------------------------------------------------- integração
def test_dreamerv3_trains_and_reports_the_pieces():
    ag = DreamerV3(cfg())
    for _ in range(3):
        s = ag.iterate()
    for chave in ("modelo", "ator", "critico", "recon", "kl_dyn", "kl_rep",
                  "rew_sonho", "train_ratio"):
        assert chave in s and np.isfinite(s[chave]), chave


def test_the_recorded_model_is_the_actor():
    """Um modelo do mundo excelente com um ator ruim vale zero na arena. O checkpoint tem
    que guardar o que a arena mede."""
    ag = DreamerV3(cfg())
    assert ag.model is ag.ator


def test_dreamerv3_belongs_to_the_world_model_family():
    assert "dreamerv3" in ORDEM_ALGORITMOS
    assert familia_de("dreamerv3") == familia_de("muzero") == "modelo"


def test_the_dream_always_leaves_one_action_available():
    """Um estado imaginado onde a máscara prevista zera as três ações é artefato do modelo.
    Se o softmax recebesse três `-inf`, viraria NaN e o treino morreria sem explicação."""
    ag = DreamerV3(cfg())
    ag.iterate()
    estado0 = tf.zeros([6, ag.dim_estado])
    estados, logps, ent = ag._sonha(estado0)
    assert np.isfinite(logps.numpy()).all() and np.isfinite(ent.numpy()).all()
    assert estados.shape[0] == ag.cfg.horizonte + 1


# ------------------------------------------- o que fazia a GPU ficar parada
def test_the_training_step_runs_as_a_graph_not_in_eager():
    """O desenrolamento do RSSM é um laço Python de `seq_len` passos e o sonho é outro de
    `horizonte`. Em modo eager isso vira milhares de kernels minúsculos e sequenciais, e
    numa GPU o custo é latência de lançamento, não cálculo — a placa fica ociosa esperando
    o Python. Medido: 1.910 ms → 95 ms na perda do modelo, 20×.
    """
    ag = DreamerV3(cfg())
    assert isinstance(ag._grafo, tf.types.experimental.GenericFunction), \
        "o passo de gradiente tem que estar dentro de um `tf.function`"


def test_the_return_scale_survives_graph_mode():
    """`self._escala_ret` como `float` seria atualizado só na traçagem e congelaria ali.

    Nada quebraria: o ator seguiria dividindo a vantagem por uma escala da primeira
    iteração, e a normalização por percentis — que existe justamente porque o retorno vai
    de ~1 a ~50 — deixaria de normalizar. Silencioso, e cara.
    """
    ag = DreamerV3(cfg())
    assert isinstance(ag._escala_ret, tf.Variable)
    antes = float(ag._escala_ret)
    for _ in range(4):
        ag.iterate()
    assert float(ag._escala_ret) != antes, "a escala não está sendo atualizada em grafo"


# ------------------------------------------------------------- o train ratio
def test_train_ratio_is_the_knob_and_train_steps_is_derived():
    """Expor a razão, e não o número de passos, é o que a mantém significando o mesmo
    quando `num_envs` muda — com `train_steps` fixo, dobrar os ambientes metade o
    aprendizado sem que nada avise."""
    for envs, coleta in ((64, 16), (16, 8), (128, 4)):
        c = DreamerV3Config(train_ratio=4.0, num_envs=envs, collect_steps=coleta)
        real = c.train_steps * c.batch_size * c.seq_len / (coleta * envs)
        assert real == pytest.approx(4.0, rel=0.25), \
            f"{envs} ambientes × {coleta}: razão real {real:.2f}"


def test_train_steps_can_still_be_pinned_for_an_ablation():
    c = DreamerV3Config(train_steps=3)
    assert c.train_steps == 3


def test_the_default_ratio_is_high_enough_to_actually_learn():
    """O padrão anterior era 0,5 — meia transição revisitada por passo de ambiente.

    Com ele, 400 mil passos deixavam o agente no piso aleatório, porque o modelo do mundo
    quase não era treinado. O Dreamer é um algoritmo que troca computação por amostras;
    com razão abaixo de 1 ele perde as duas coisas.
    """
    assert DreamerV3Config().train_ratio >= 2.0
    assert DreamerV3Config().train_steps >= 4


def test_the_collect_step_also_runs_as_a_graph():
    """A coleta era 99% overhead de despacho: 7,7 ms de ambiente contra 603 ms de modelo.

    Cada passo chama encoder, posterior, GRU e ator. Em eager são quatro despachos
    separados sobre um lote pequeno; numa GPU cada um espera o Python, e como o treino já
    está em grafo é a **coleta** que passa a segurar a placa ociosa.
    """
    ag = DreamerV3(cfg())
    assert isinstance(ag._grafo_politica, tf.types.experimental.GenericFunction)


def test_the_policy_step_is_pure_so_it_can_be_traced():
    """`(h, z)` entram e saem como tensores em vez de virar atributo.

    Atribuir a `self` dentro de um `tf.function` acontece **só na traçagem**: o latente
    congelaria no da primeira iteração e o agente agiria para sempre com o estado inicial,
    sem erro nenhum.
    """
    ag = DreamerV3(cfg())
    ag.collect()          # sem isto, `primeiro` é todo True e `h` sai zerado por definição
    obs, mask = ag.obs, ag.mask
    h0, z0 = ag._h, ag._z
    a, h, z = ag._passo_de_politica(
        h0, z0, tf.one_hot(ag._ultima_acao, 3), tf.convert_to_tensor(ag._primeiro),
        tf.convert_to_tensor(obs, tf.float32), tf.convert_to_tensor(mask))
    assert a.shape == (ag.cfg.num_envs,)
    assert not np.allclose(h.numpy(), h0.numpy()), "o estado devolvido tem que avançar"
    assert ag._h is h0, "e a função não pode ter trocado o atributo de `self`"


def test_the_latent_keeps_advancing_across_collect_calls():
    """O contrapeso do teste acima: puro não pode virar sem memória."""
    ag = DreamerV3(cfg())
    ag.collect()
    h1 = ag._h.numpy().copy()
    ag.collect()
    assert not np.allclose(ag._h.numpy(), h1)


def test_the_evaluation_policy_uses_the_same_graph():
    """A avaliação são 1.000 episódios de centenas de passos. Em eager, o protocolo
    oficial custaria mais que um pedaço do treino."""
    pol = DreamerV3(cfg()).politica()
    assert isinstance(pol._grafo, tf.types.experimental.GenericFunction)
