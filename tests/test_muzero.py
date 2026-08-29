"""MuZero.

O teste central deste arquivo é `test_muzero_and_alphazero_share_the_same_search`: se a
busca não for literalmente o mesmo objeto, a comparação entre os dois deixa de medir "o que
custa não ter o simulador" e passa a medir duas implementações diferentes de MCTS.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from snakeai.agents import AlphaZero, AlphaZeroConfig, MuZero, MuZeroConfig
from snakeai.env.vec_snake import N_ACTIONS, N_CHANNELS, VecSnake
from snakeai.eval import MASK_NEG
from snakeai.nets.muzero import (
    build_dinamica,
    build_predicao,
    build_representacao,
    escala_gradiente,
    normaliza_oculto,
)
from snakeai.search import MCTS, DinamicaAprendida, DinamicaReal

LARGURA_TINY = 32


def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=6, unroll=2, num_simulations=4,
                batch_size=16, memory_size=2000, total_steps=1000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return MuZeroConfig(**base)


# ------------------------------------------------------------------- as redes
def test_the_three_networks_have_the_right_shapes():
    h = build_representacao(10, "resnet_tiny")
    g = build_dinamica(10, "resnet_tiny")
    f = build_predicao(10, "resnet_tiny")

    obs = np.zeros((4, 10, 10, N_CHANNELS), np.float32)
    s = np.asarray(h(obs, training=False))
    assert s.shape == (4, 10, 10, LARGURA_TINY)

    # os dois tensores da dinâmica têm que ser do MESMO tipo: o Keras 3 recusa uma lista
    # que mistura tf.Tensor e ndarray com "you cannot mix tensors and non-tensors"
    planos = np.zeros((4, 10, 10, N_ACTIONS), np.float32)
    s2, r = g([s, planos], training=False)
    assert tuple(s2.shape) == tuple(s.shape)
    assert tuple(r.shape) == (4, 1)

    logits, v = f(s, training=False)
    assert tuple(logits.shape) == (4, N_ACTIONS) and tuple(v.shape) == (4, 1)


def test_hidden_state_is_normalized_to_zero_one():
    """Sem isso a escala do estado cresce a cada `g` e o desenrolar de K passos explode."""
    x = np.random.default_rng(0).normal(0, 50, size=(4, 10, 10, 8)).astype(np.float32)
    y = np.asarray(normaliza_oculto(ops.convert_to_tensor(x)))
    assert y.min() >= -1e-5 and y.max() <= 1 + 1e-5
    for i in range(4):
        assert y[i].min() == pytest.approx(0.0, abs=1e-4)
        assert y[i].max() == pytest.approx(1.0, abs=1e-4)


def test_representation_and_dynamics_stay_bounded_over_a_long_unroll():
    h = build_representacao(10, "resnet_tiny")
    g = build_dinamica(10, "resnet_tiny")
    s = np.asarray(h(np.zeros((2, 10, 10, N_CHANNELS), np.float32), training=False))
    planos = np.zeros((2, 10, 10, N_ACTIONS), np.float32)
    for _ in range(20):
        s, _ = g([s, planos], training=False)
        s = np.asarray(s)
    assert np.isfinite(s).all() and s.max() <= 1 + 1e-4


def test_gradient_scale_keeps_the_value_and_shrinks_the_gradient():
    x = tf.Variable([2.0])
    with tf.GradientTape() as t:
        y = escala_gradiente(x * 1.0, 0.5)
    assert float(y[0]) == pytest.approx(2.0)
    assert float(t.gradient(y, x)[0]) == pytest.approx(0.5)


# ------------------------------------------------------------------ a dinâmica
def test_muzero_and_alphazero_share_the_same_search():
    """A afirmação arquitetural do repositório, verificada.

    Se a busca não for o MESMO objeto, a diferença entre os dois algoritmos deixa de medir
    "quanto custa não ter o simulador" e passa a medir duas implementações de MCTS.
    """
    az = AlphaZero(AlphaZeroConfig(net="resnet_tiny", num_envs=4, rollout=2,
                                   num_simulations=3, batch_size=8,
                                   salvar_gif=False, salvar_grafico=False))
    mz = MuZero(cfg_min(num_envs=4, rollout=2, num_simulations=3))
    assert type(az.mcts) is type(mz.mcts) is MCTS
    assert isinstance(az.mcts.dinamica, DinamicaReal)
    assert isinstance(mz.mcts.dinamica, DinamicaAprendida)
    # e o código da busca é o mesmo método, não uma cópia
    assert az.mcts.run.__func__ is mz.mcts.run.__func__


def test_real_dynamics_matches_the_environment():
    din = DinamicaReal(10)
    env = VecSnake(4, 10, rng=np.random.default_rng(0))
    env.reset()
    estado = env.get_state()
    a = np.array([0, 1, 2, 1], np.int32)
    _, obs, mask, r, d = din.passo(estado, a)
    env.set_state(estado)
    obs2, mask2, r2, d2, _ = env.step(a)
    assert np.array_equal(r, r2) and np.array_equal(d, d2)


def test_learned_dynamics_reports_no_terminal_and_no_mask():
    """Documenta a limitação: o modelo não prevê fim de episódio nem ação ilegal."""
    mz = MuZero(cfg_min())
    din = mz.mcts.dinamica
    assert din.usa_mascara is False
    s = mz.h(np.zeros((3, 10, 10, N_CHANNELS), np.float32), training=False).numpy()
    novo, obs, mask, r, d = din.passo(s, np.array([0, 1, 2], np.int32))
    assert novo.shape == s.shape
    assert mask.all() and not d.any()


# ------------------------------------------------------------------ o agente
def test_collect_stores_the_unroll_targets_aligned():
    """Cada amostra guarda os K passos que vêm DEPOIS dela — desalinhar isso ensina errado."""
    cfg = cfg_min(rollout=6, unroll=2, num_envs=4)
    ag = MuZero(cfg)
    ag.collect()
    n = ag._cheio
    # todas as `T` linhas viram amostra: o que não tem futuro dentro da janela é
    # **mascarado**, não descartado (ver `test_every_collected_step_becomes_a_sample`)
    assert n == cfg.rollout * cfg.num_envs
    assert ag._buf_pi[:n].shape[1] == cfg.unroll + 1
    assert ag._buf_act[:n].shape[1] == cfg.unroll
    assert ag._buf_r[:n].shape[1] == cfg.unroll
    assert np.allclose(ag._buf_pi[:n].sum(-1), 1.0, atol=1e-4)


def test_training_reports_the_three_losses():
    ag = MuZero(cfg_min(batch_size=16))
    ag.iterate()
    stats = ag.iterate()
    for chave in ("perda_pi", "perda_v", "perda_r"):
        assert chave in stats and np.isfinite(stats[chave])


def test_reward_loss_anchors_the_model_to_the_world():
    """A perda de recompensa é a única âncora: sem ela a dinâmica inventa a física."""
    ag = MuZero(cfg_min(batch_size=16))
    for _ in range(4):
        stats = ag.iterate()
    assert stats["perda_r"] >= 0
    # a rede de dinâmica recebe gradiente
    antes = [w.numpy().copy() for w in ag.g.trainable_variables]
    ag.iterate()
    depois = [w.numpy() for w in ag.g.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))


def test_representation_receives_gradient_through_the_unroll():
    ag = MuZero(cfg_min(batch_size=16, unroll=3))
    ag.iterate()
    antes = [w.numpy().copy() for w in ag.h.trainable_variables]
    ag.iterate()
    depois = [w.numpy() for w in ag.h.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))


def test_official_policy_has_no_search():
    ag = MuZero(cfg_min())
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a, b = fn(obs, mask), fn(obs, mask)
    assert np.array_equal(a, b)
    assert (a[~mask] == MASK_NEG).all()


def test_search_runs_over_hidden_states():
    ag = MuZero(cfg_min(num_simulations=6))
    obs, mask = ag.env.reset()
    visitas, valores = ag._busca(obs, mask)
    assert visitas.shape == (ag.cfg.num_envs, N_ACTIONS)
    assert (visitas.sum(1) > 0).all()
    assert (visitas[~mask] == 0).all(), "a máscara vale na raiz, onde o estado é real"


def test_temperature_decays():
    """O agendamento por fração do **treino** — hoje só com `temp_passos=0`.

    O padrão passou a ser o do paper: τ alto nos primeiros lances de cada **episódio**,
    frio no resto (`test_per_move_temperature_replaces_the_training_schedule` cobre esse).
    Os dois não se somam: `temp_passos > 0` substitui este por completo.
    """
    ag = MuZero(cfg_min(total_steps=1000, temp_inicio=1.0, temp_fim=0.25, temp_frac=0.5,
                        temp_passos=0))
    assert ag.temperatura() == pytest.approx(1.0)
    ag.global_step = 500
    assert ag.temperatura() == pytest.approx(0.25)


def test_per_move_temperature_replaces_the_training_schedule():
    ag = MuZero(cfg_min(temp_passos=5, temp_inicio=1.0, temp_fim=0.05))
    ag.env.steps[:] = 0
    assert np.allclose(ag.temperatura(), 1.0)
    ag.env.steps[:] = 99
    assert np.allclose(ag.temperatura(), 0.05)
    ag.env.steps[: ag.cfg.num_envs // 2] = 0
    t = ag.temperatura()
    assert t.shape == (ag.cfg.num_envs,) and set(np.unique(t)) == {1.0, 0.05}


def test_checkpoint_saves_all_three_networks(tmp_path):
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = MuZero(cfg)
    ag.iterate()
    ag.salvar("last")
    for nome in ("h", "g", "f"):
        assert (tmp_path / f"muzero_last_{nome}.keras").exists()

    outro = MuZero(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    x = np.zeros((2, 10, 10, N_CHANNELS), np.float32)
    assert np.allclose(np.asarray(ag.h(x, training=False)),
                       np.asarray(outro.h(x, training=False)), atol=1e-5)
    outro.iterate()


# --------------------------------------- os consertos herdados do AlphaZero (§2.27-§2.29)
def test_the_alphazero_fixes_are_the_default_here_too():
    """O `MCTS` é o mesmo objeto, então os defeitos eram os mesmos.

    E aqui não havia execução de controle a preservar — o MuZero nunca rodou sob o
    contrato —, então os consertos já nascem ligados. Cada um continua desligável, senão
    não haveria como medir quanto valeram.
    """
    c = MuZeroConfig()
    assert (c.fpu, c.q_normalizado, c.desempate) == ("pai", True, "aleatorio")
    assert c.valor_symlog is True and c.bootstrap_fim_janela is True
    assert (c.temp_alvo, c.temp_passos) == (1.0, 30)
    assert (c.epochs_por_iter, c.lr_final, c.dirichlet_alpha) == (8, 5e-5, 1.0)

    velho = MuZeroConfig(fpu="zero", q_normalizado=False, valor_symlog=False,
                         temp_alvo=0.0, temp_passos=0, bootstrap_fim_janela=False,
                         desempate="ordem")
    assert velho.fpu == "zero" and velho.valor_symlog is False


def test_the_search_config_reaches_the_tree():
    """As flags não podem ficar no config sem chegar no `MCTS` — seria melhoria de mentira."""
    ag = MuZero(cfg_min())
    assert ag.mcts.fpu == "pai" and ag.mcts.q_normalizado is True
    assert ag.mcts.desempate == "aleatorio"
    outro = MuZero(cfg_min(fpu="zero", q_normalizado=False, desempate="ordem"))
    assert outro.mcts.fpu == "zero" and outro.mcts.q_normalizado is False


def test_symlog_keeps_the_tree_on_the_real_value_scale():
    """O backup do MCTS soma `recompensa + γ·valor`, e a recompensa é a que `g` prevê, na
    escala do mundo. Se a leitura devolvesse o valor comprimido, a árvore compararia
    recompensas com logaritmos."""
    ag = MuZero(cfg_min(valor_symlog=True))
    cru = MuZero(cfg_min(valor_symlog=False))
    for a, b in ((cru.h, ag.h), (cru.f, ag.f), (cru.g, ag.g)):
        a.set_weights(b.get_weights())
    obs, mask = ag.env.reset()
    _, _, v_sym = ag._repr_predicao(tf.convert_to_tensor(obs), tf.convert_to_tensor(mask))
    _, _, v_cru = cru._repr_predicao(tf.convert_to_tensor(obs), tf.convert_to_tensor(mask))
    esperado = np.sign(v_cru.numpy()) * np.expm1(np.abs(v_cru.numpy()))
    assert np.allclose(v_sym.numpy(), esperado, atol=1e-4)

    teto = float(np.expm1(MuZero.LIMITE_SYMLOG))
    for x in (0.0, 4.6, 40.0, -40.0):
        assert abs(float(MuZero._symexp(np.float32(x)).numpy())) <= teto + 1e-3


def test_search_evaluation_follows_the_same_protocol():
    """A coluna separada do contrato: mesmo protocolo, escolhendo com a busca.

    O MuZero existe para buscar sobre um modelo aprendido; publicá-lo medido só sem buscar
    seria meia medição. Ver `docs/BUSCA_DEGENERADA.md`.
    """
    ag = MuZero(cfg_min())
    st = ag.avaliar_com_busca(episodes=32, num_simulations=3)
    assert st["episodes"] == 32 and st["num_simulations"] == 3
    assert 0.0 <= st["score_mean"] <= 97
    assert set(st) >= {"score_mean", "score_median", "score_max", "win_rate", "completo"}


def test_the_policy_target_is_the_raw_visit_count_by_default():
    ag = MuZero(cfg_min(temp_inicio=0.05, temp_fim=0.05))
    ag.collect()
    n = ag._cheio
    pi = ag._buf_pi[:n].reshape(-1, ag._buf_pi.shape[-1])
    assert np.allclose(pi.sum(1), 1.0, atol=1e-5)
    assert pi.max(1).mean() < 0.99      # τ=0,05 no alvo teria dado quase-argmax


def test_the_learning_rate_decays(): 
    ag = MuZero(cfg_min(total_steps=1000, lr=3e-4, lr_final=5e-5))
    cedo = ag.iterate()
    ag.global_step = 1000
    tarde = ag.iterate()
    assert tarde["lr"] < cedo["lr"] and tarde["lr"] == pytest.approx(5e-5, rel=1e-3)
    assert tarde["atualizacoes"] == ag.cfg.epochs_por_iter


def test_the_unroll_never_crosses_an_episode_boundary():
    """O `VecSnake` reseta sozinho: sem máscara, o desenrolar treinava `g` a prever a
    recompensa de uma partida **nova**, sorteada, que o estado oculto não tem como
    conhecer — e a perda de recompensa é a única âncora do latente no mundo.

    Força mortes cedo pondo a fome a um passo do limite: aí toda janela atravessa pelo
    menos uma fronteira, e a máscara tem que zerar tudo depois dela.
    """
    cfg = cfg_min(rollout=6, unroll=2)
    ag = MuZero(cfg)
    # fome a um passo do limite: todos os ambientes terminam em t=0, e o episódio que a
    # linha 0 enxerga a partir de t=1 é outro jogo
    ag.env.hunger[:] = ag.env.starve_base + 2 * ag.env.length - 1
    ag.collect()
    vivo = ag._buf_vivo[: ag._cheio].reshape(cfg.rollout, cfg.num_envs, cfg.unroll + 1)

    assert (vivo[..., 0] == 1.0).all(), "o passo 0 é sempre real"
    assert vivo[0, :, 1:].sum() == 0.0, \
        "a linha que atravessa a morte deixou passo do desenrolar marcado como real"
    # a máscara é monotônica: uma vez fora, não volta
    assert (np.diff(vivo, axis=-1) <= 0).all()


def test_every_collected_step_becomes_a_sample():
    """Antes, `validos = T - K` descartava as `K` últimas linhas da janela — 31% dos passos
    coletados nunca viravam amostra, e continuavam contados no orçamento de 5 M."""
    cfg = cfg_min(rollout=6, unroll=2)
    ag = MuZero(cfg)
    ag.collect()
    assert ag._cheio == cfg.rollout * cfg.num_envs
    # os passos que cairiam fora da janela estão mascarados, não descartados
    vivo = ag._buf_vivo[: ag._cheio].reshape(cfg.rollout, cfg.num_envs, cfg.unroll + 1)
    assert vivo[cfg.rollout - 1, :, 1:].sum() == 0.0, "a última linha não tem futuro"


def test_masked_unroll_does_not_shrink_the_loss():
    """A média é sobre os passos reais, não sobre o lote inteiro — senão a perda cairia só
    porque a janela atravessou uma morte, e o gradiente encolheria junto."""
    x = tf.constant([[1.0, 1.0, 1.0, 1.0]])
    cheia = MuZero._media_mascarada(x, tf.ones_like(x))
    metade = MuZero._media_mascarada(x, tf.constant([[1.0, 1.0, 0.0, 0.0]]))
    assert float(cheia) == pytest.approx(1.0)
    assert float(metade) == pytest.approx(1.0)
    nenhuma = MuZero._media_mascarada(x, tf.zeros_like(x))
    assert float(nenhuma) == pytest.approx(0.0)      # e não NaN


def test_search_column_scores_the_last_apple_and_counts_wins_from_the_sample():
    """As duas armadilhas que `snakeai/eval.py` documenta, agora compartilhadas.

    Ler `env.score` antes do passo perde um ponto em todo episódio que termina comendo —
    isto é, em **toda vitória por tabuleiro cheio** —, e contar vitórias num contador do
    laço soma os ambientes que já cumpriram a cota. As duas subestimam ou inflam justamente
    o regime em que um agente bom vive.
    """
    ag = MuZero(cfg_min())
    st = ag.avaliar_com_busca(episodes=32, num_simulations=3)
    assert st["episodes"] == 32 and st["num_simulations"] == 3
    # o mesmo conjunto de chaves que a linha da política pura, para as duas caberem na
    # mesma tabela do `format_verdict`
    for chave in ("score_mean", "score_median", "score_std", "score_max", "score_p95",
                  "win_rate", "perfect_possible", "fim_fome", "fim_colisao",
                  "fim_tabuleiro_cheio"):
        assert chave in st, chave
    assert 0.0 <= st["win_rate"] <= 1.0
    assert abs(sum(st[f"fim_{k}"] for k in ("fome", "colisao", "tabuleiro_cheio")) - 1.0) < 1e-6
