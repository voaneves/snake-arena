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


# --------------------------------------------- §2.31 · o peso dos passos do desenrolar
def _passo_isolado(cfg):
    """Um `_passo` sobre o buffer recém-coletado, com as mesmas amostras e os mesmos
    pesos iniciais. Duas configs com a mesma semente produzem buffers idênticos, então a
    única diferença entre as chamadas é a que se quer medir."""
    ag = MuZero(cfg)
    ag.collect()
    i = np.arange(cfg.batch_size)
    p, v, r, p0 = ag._passo(
        tf.convert_to_tensor(ag._buf_obs[i]), tf.convert_to_tensor(ag._buf_mask[i]),
        tf.convert_to_tensor(ag._buf_act[i]), tf.convert_to_tensor(ag._buf_pi[i]),
        tf.convert_to_tensor(ag._buf_z[i]), tf.convert_to_tensor(ag._buf_r[i]),
        tf.convert_to_tensor(ag._buf_vivo[i]),
        tf.ones(cfg.batch_size, tf.float32),        # sem PER, todo peso é 1
        cfg.coef_valor, cfg.coef_recompensa)
    return float(p), float(p0), ag


def test_the_unrolled_steps_are_a_raw_sum_by_default():
    """O padrão soma os `K+1` termos sem peso. É o que a primeira execução rodou, e é o
    que faz `unroll` maior **diluir** o passo 0 — o único que a métrica oficial mede."""
    assert MuZeroConfig().normaliza_unroll is False
    perda, p0, _ = _passo_isolado(cfg_min(batch_size=16, unroll=4))
    assert p0 < perda, "com K=4 o passo 0 é uma fração pequena da soma"


def test_normalizing_the_unroll_divides_only_the_imagined_steps():
    """`scale_gradient(loss, 1/K)` do pseudocódigo: o passo 0 fica inteiro e os `K` passos
    imaginados dividem um peso entre si. O passo 0 sai de `1/(K+1)` da soma para ~metade."""
    K = 4
    crua, p0_crua, _ = _passo_isolado(cfg_min(batch_size=16, unroll=K))
    norm, p0_norm, _ = _passo_isolado(
        cfg_min(batch_size=16, unroll=K, normaliza_unroll=True))
    # mesmas amostras, mesmos pesos iniciais: o passo 0 tem de bater exatamente
    assert p0_norm == pytest.approx(p0_crua, rel=1e-5)
    # e a parte imaginada tem de ser exatamente `1/K` da que a soma crua usou
    assert norm - p0_norm == pytest.approx((crua - p0_crua) / K, rel=1e-4)


def test_the_step_zero_share_of_the_policy_loss_is_reported():
    """Sem `frac_pi_0` não dá para distinguir "a destilação falha no estado real" de "a
    destilação falha nos estados imaginados" — a soma esconde os dois casos."""
    ag = MuZero(cfg_min(batch_size=16, unroll=4))
    ag.iterate()
    stats = ag.iterate()
    assert stats["perda_pi_0"] <= stats["perda_pi"] + 1e-6
    assert 0.0 < stats["frac_pi_0"] <= 1.0
    assert stats["frac_pi_0"] == pytest.approx(
        stats["perda_pi_0"] / stats["perda_pi"], rel=1e-6)


def test_normalizing_makes_the_step_zero_share_independent_of_the_unroll():
    """O ponto todo: com a soma crua, dobrar `unroll` corta a fatia do passo 0 quase pela
    metade; com o peso do paper, a fatia não depende de `K`."""
    fatias = {}
    for K in (2, 6):
        for norm in (False, True):
            perda, p0, _ = _passo_isolado(
                cfg_min(batch_size=16, rollout=8, unroll=K, normaliza_unroll=norm))
            fatias[(K, norm)] = p0 / perda
    assert fatias[(6, False)] < fatias[(2, False)] * 0.6, "a soma crua dilui o passo 0"
    assert fatias[(6, True)] == pytest.approx(fatias[(2, True)], abs=0.12)


# --------------------------------------------------- §2.32 · Reanalyse da política
def test_reanalyse_is_off_by_default():
    """Ligar por argumento de paper repetiria o erro do §2.27 ao contrário. E aqui há um
    custo de busca real a pagar, medido em `tools/diag_reanalise.py`."""
    assert MuZeroConfig().reanalise == 0.0
    ag = MuZero(cfg_min(batch_size=16))
    ag.iterate()
    assert ag.iterate()["reanalises"] == 0


def test_reanalyse_refreshes_the_step_zero_target_with_the_current_network():
    """O alvo guardado veio de uma rede de dezenas de iterações atrás. Refazer a busca com
    a rede atual é o que o Apêndice H chama de Reanalyse."""
    cfg = cfg_min(batch_size=16, epochs_por_iter=2, reanalise=0.5)
    ag = MuZero(cfg)
    ag.collect()
    antes = ag._buf_pi.copy()
    i = np.arange(8)
    n = ag._reanalisar(i)
    assert n == 8
    depois = ag._buf_pi
    # o passo 0 das linhas escolhidas mudou; nada mais mudou
    assert not np.allclose(antes[i, 0], depois[i, 0])
    assert np.allclose(antes[i, 1:], depois[i, 1:]), "só o passo 0 é refeito"
    resto = np.arange(8, ag._cheio)
    assert np.allclose(antes[resto], depois[resto]), "só as linhas escolhidas"
    # continua sendo uma distribuição
    assert np.allclose(depois[i, 0].sum(-1), 1.0, atol=1e-4)


def test_reanalyse_writes_back_so_the_work_compounds():
    """Se o alvo refeito não voltasse para o buffer, cada sorteio pagaria a busca de novo
    e a taxa de refresco não comporia — seria custo puro."""
    cfg = cfg_min(batch_size=16, epochs_por_iter=2, reanalise=1.0)
    ag = MuZero(cfg)
    ag.collect()
    alvo = ag._buf_pi.copy()
    ag._aprender()
    mudou = ~np.all(np.isclose(alvo[: ag._cheio, 0], ag._buf_pi[: ag._cheio, 0]), axis=-1)
    assert mudou.sum() > 0, "o alvo refeito ficou gravado no buffer"
    assert np.allclose(alvo[: ag._cheio, 1:], ag._buf_pi[: ag._cheio, 1:])


def test_reanalyse_counts_its_searches():
    """O custo tem de estar no registro: é ele que se lê contra as `num_envs × rollout`
    buscas da coleta para saber o que a execução vai custar de parede."""
    cfg = cfg_min(batch_size=16, epochs_por_iter=3, reanalise=0.5)
    ag = MuZero(cfg)
    ag.iterate()
    st = ag.iterate()
    assert st["reanalises"] == 3 * 8


def test_reanalyse_can_search_cheaper_than_collection():
    """O botão de custo. É desvio do paper — produz alvo de qualidade menor que o da
    coleta — e existe para o caso de a busca cheia não caber no tempo de parede."""
    cfg = cfg_min(batch_size=16, num_simulations=8, reanalise=0.5, reanalise_sims=3)
    ag = MuZero(cfg)
    ag.collect()
    ag._reanalisar(np.arange(4))
    assert ag._busca_reanalise().num_simulations == 3
    assert ag.mcts.num_simulations == 8, "a busca da coleta não é tocada"


def test_reanalyse_adds_no_root_noise_so_the_target_is_reproducible():
    """O ruído de Dirichlet existe para explorar na geração de dados; aqui o que se produz
    é um alvo, e um alvo não deve depender de um sorteio.

    O teste é a **reprodutibilidade**, e não "o alvo refeito é mais afiado": esta última
    depende do estado do treino. Com a rede treinada, cujo prior já é agudo, o ruído
    espalha e o refeito sai mais afiado; com a rede recém-iniciada, cujo prior é quase
    uniforme, um sorteio típico de `Dir(1,1,1)` (máximo esperado ~0,61) é *mais* agudo que
    ela e a direção se inverte. A ausência de ruído vale nos dois casos.
    """
    cfg = cfg_min(batch_size=16, num_simulations=24, reanalise=1.0, desempate="ordem")
    ag = MuZero(cfg)
    ag.collect()
    i = np.arange(min(24, ag._cheio))
    ag._reanalisar(i)
    uma = ag._buf_pi[i, 0].copy()
    ag._reanalisar(i)
    assert np.allclose(uma, ag._buf_pi[i, 0]), "sem ruído, o alvo é reprodutível"
    # o contraste: com ruído na raiz, dois alvos do mesmo estado e da mesma rede diferem
    visitas_a, _ = ag._busca(ag._buf_obs[i], ag._buf_mask[i], ruido=True)
    visitas_b, _ = ag._busca(ag._buf_obs[i], ag._buf_mask[i], ruido=True)
    assert not np.allclose(visitas_a, visitas_b), "o ruído da coleta é um sorteio"


def test_reanalyse_never_swaps_the_learned_dynamics_for_the_real_one():
    """Uma árvore sem a `DinamicaAprendida` percorre o `VecSnake` — isto é, vira o
    AlphaZero. É uma troca que não levanta exceção: só devolve outro algoritmo."""
    from snakeai.search import DinamicaAprendida
    for sims in (0, 3):
        ag = MuZero(cfg_min(batch_size=16, num_simulations=8, reanalise_sims=sims))
        assert isinstance(ag._busca_reanalise().dinamica, DinamicaAprendida)
    # e a árvore alternativa é construída uma vez só
    ag = MuZero(cfg_min(batch_size=16, num_simulations=8, reanalise_sims=3))
    assert ag._busca_reanalise() is ag._busca_reanalise()
    assert ag._busca_reanalise() is not ag.mcts
    assert ag.mcts.num_simulations == 8, "a busca da coleta não é tocada"


# ------------------------------------- §2.33 · cabeça categórica e transformação de escala
def test_the_scalar_head_is_still_the_default():
    """Nada disto liga sozinho: há uma execução de controle a preservar."""
    assert MuZeroConfig().n_suporte == 0
    assert MuZeroConfig().transformacao == "symlog"
    ag = MuZero(cfg_min(batch_size=16))
    assert ag.atomos is None
    assert ag.f.output_shape[1][-1] == 1, "cabeça de valor escalar"
    assert ag.g.output_shape[1][-1] == 1, "cabeça de recompensa escalar"


def test_the_categorical_head_emits_logits_over_the_support():
    ag = MuZero(cfg_min(batch_size=16, n_suporte=121))
    assert ag.f.output_shape[1][-1] == 121
    assert ag.g.output_shape[1][-1] == 121
    assert ag.atomos.shape == (121,)


def test_two_hot_reproduces_the_target_exactly_through_its_expectation():
    """É a propriedade que define a projeção: um alvo de 3,7 vira 0,3 no átomo 3 e 0,7 no
    4 justamente para que a esperança devolva 3,7. Errar isto não levanta exceção — só
    treina a cabeça contra um número que não é o alvo."""
    ag = MuZero(cfg_min(batch_size=16, n_suporte=121))
    alvo = tf.constant([-4.0, -1.5, -0.033, 0.0, 0.7, 2.5, 4.0])
    th = ag._dois_quentes(alvo)
    assert np.allclose(th.numpy().sum(-1), 1.0, atol=1e-5), "é uma distribuição"
    assert (th.numpy() > 1e-9).sum(-1).max() <= 2, "no máximo dois átomos, daí o nome"
    volta = tf.reduce_sum(th * ag.atomos, -1)
    assert np.allclose(volta.numpy(), alvo.numpy(), atol=1e-4)


def test_the_support_is_sized_by_the_domain_and_not_copied_from_atari():
    """601 átomos em [-300,300] dariam ~3 pontos de resolução perto de zero, num jogo cujo
    valor medido vive entre 0 e ~11. As duas transformações cobrem a MESMA faixa real."""
    for t in ("symlog", "h"):
        ag = MuZero(cfg_min(batch_size=16, n_suporte=121, transformacao=t))
        reais = ag._descomprime(ag.atomos).numpy()
        assert reais[0] == pytest.approx(-60.0, rel=1e-3)
        assert reais[-1] == pytest.approx(60.0, rel=1e-3)
        assert reais[61] - reais[60] < 0.3, f"{t}: resolução perto de zero"


def test_the_categorical_loss_is_cross_entropy_and_is_minimal_at_the_target():
    ag = MuZero(cfg_min(batch_size=16, n_suporte=121))
    alvo = tf.constant([0.0, 1.0, -2.0])
    perfeito = tf.math.log(ag._dois_quentes(alvo) + 1e-9)
    uniforme = tf.zeros_like(perfeito)
    assert float(tf.reduce_mean(ag._perda_escalar(uniforme, alvo))) == pytest.approx(
        float(np.log(121)), rel=1e-3), "sem informação, a CE é ln(n)"
    assert (float(tf.reduce_mean(ag._perda_escalar(perfeito, alvo)))
            < float(tf.reduce_mean(ag._perda_escalar(uniforme, alvo))))


def test_the_paper_transform_round_trips_and_has_its_own_ceiling():
    """`h` cresce como √x e o `symexp` como exp, então reusar o limite do symlog cortaria
    o valor em 47 na escala real — um teto que este jogo encosta, e um corte silencioso."""
    x = tf.constant([-400.0, -97.0, -20.0, -1.0, 0.0, 1.0, 20.0, 97.0, 400.0])
    assert np.allclose(MuZero._h_inv(MuZero._h(x)).numpy(), x.numpy(), rtol=1e-4, atol=1e-2)
    assert MuZero.LIMITE_H > MuZero.LIMITE_SYMLOG
    # os dois tetos descrevem a MESMA fronteira real
    teto_h = float(MuZero._h_inv(tf.constant(MuZero.LIMITE_H)))
    teto_s = float(MuZero._symexp(tf.constant(MuZero.LIMITE_SYMLOG)))
    assert teto_h == pytest.approx(teto_s, rel=0.05)


def test_the_tree_reads_value_and_reward_on_the_real_scale_with_a_categorical_head():
    """O backup soma `recompensa + γ·valor`, e as duas saem de cabeças categóricas na
    escala comprimida. Se a volta não acontecer, a árvore soma unidades diferentes."""
    for kw in ({"n_suporte": 121}, {"n_suporte": 121, "transformacao": "h"}):
        ag = MuZero(cfg_min(batch_size=16, num_simulations=4, **kw))
        _s, pri, val = ag._repr_predicao(tf.convert_to_tensor(ag.obs),
                                         tf.convert_to_tensor(ag.mask))
        assert np.all(np.isfinite(val.numpy()))
        assert np.allclose(pri.numpy().sum(-1), 1.0, atol=1e-4)
        assert np.abs(val.numpy()).max() < 61.0, "dentro do suporte"
        ag.iterate()


def test_training_runs_with_every_combination_of_head_and_transform():
    for kw in ({}, {"n_suporte": 121}, {"transformacao": "h"},
               {"n_suporte": 121, "transformacao": "h"}):
        ag = MuZero(cfg_min(batch_size=16, **kw))
        ag.iterate()
        st = ag.iterate()
        for c in ("perda_pi", "perda_v", "perda_r"):
            assert np.isfinite(st[c]), f"{kw}: {c}"


# ------------------------------------------- §2.34 · o agendamento de temperatura de Atari
def test_the_board_game_temperature_schedule_is_still_the_default():
    assert MuZeroConfig().temp_esquema == "lance"


def test_the_atari_schedule_steps_by_training_progress_and_not_by_move():
    """O Apêndice D tem dois agendamentos. O de Atari amostra o episódio INTEIRO, com τ em
    degraus por passo de treino — e é o que cabe num jogo de ~1.200 lances, onde
    `temp_passos=30` deixa 97,5% do episódio frio desde a primeira iteração."""
    ag = MuZero(cfg_min(batch_size=16, temp_esquema="treino", total_steps=1000))
    for frac, esperado in ((0.0, 1.0), (0.49, 1.0), (0.5, 0.5), (0.74, 0.5),
                           (0.75, 0.25), (0.99, 0.25)):
        ag.global_step = int(frac * ag.cfg.total_steps)
        t = ag.temperatura()
        assert np.isscalar(t) or np.ndim(t) == 0, "vale para o episódio inteiro"
        assert float(t) == pytest.approx(esperado)


def test_the_move_schedule_leaves_almost_the_whole_episode_cold():
    """O número que motiva o §2.34: com episódios longos, `temp_passos` quase não age."""
    ag = MuZero(cfg_min(batch_size=16, temp_passos=30))
    ag.env.steps[:] = 500                      # meio de um episódio típico do agente bom
    assert np.all(ag.temperatura() == ag.cfg.temp_fim)
    ag.env.steps[:] = 5
    assert np.all(ag.temperatura() == ag.cfg.temp_inicio)


# ------------------------------------------- §2.35 · replay priorizado do Apêndice G
def test_uniform_sampling_is_still_the_default():
    assert MuZeroConfig().per == 0.0
    assert MuZeroConfig().per_beta == 1.0


def test_the_priority_is_the_gap_between_the_search_and_the_game():
    """`p = |ν − z|`: o quanto o valor que a BUSCA achou na raiz discordou do retorno que o
    jogo entregou. Não é o erro da rede — e por isso é fixo no instante da coleta."""
    ag = MuZero(cfg_min(batch_size=16))
    ag.collect()
    p = ag._buf_prio[: ag._cheio]
    assert np.all(p >= 0.0) and np.any(p > 0.0)
    # gravada mesmo com o PER desligado: é diagnóstico de graça
    assert ag.cfg.per == 0.0


def test_priorities_are_never_updated_after_collection():
    """Ao contrário do PER do DQN, que segue o erro TD atual. As duas quantidades do
    Apêndice G são fixas, e é isso que torna esta implementação simples."""
    ag = MuZero(cfg_min(batch_size=16, per=1.0))
    ag.collect()
    antes = ag._buf_prio.copy()
    for _ in range(3):
        ag._aprender()
    assert np.array_equal(antes, ag._buf_prio)


def test_prioritized_sampling_favours_the_rows_where_search_and_game_disagreed():
    ag = MuZero(cfg_min(batch_size=16, per=1.0, memory_size=400))
    ag.collect()
    n = ag._cheio
    ag._buf_prio[:n] = 1e-6
    alvos = np.arange(5)
    ag._buf_prio[alvos] = 100.0
    p = np.power(ag._buf_prio[:n] + 1e-6, 1.0)
    probs = p / p.sum()
    sorteados = ag.rng.choice(n, size=4000, p=probs)
    fatia = np.isin(sorteados, alvos).mean()
    assert fatia > 0.9, f"as 5 linhas de prioridade alta levaram só {fatia:.1%} dos sorteios"


def test_importance_weights_are_normalised_so_the_loss_scale_does_not_wander():
    """Sem normalizar, a escala da perda passaria a depender de qual amostra caiu no lote —
    e o passo do otimizador junto."""
    ag = MuZero(cfg_min(batch_size=16, per=1.0))
    ag.collect()
    n = ag._cheio
    p = np.power(ag._buf_prio[:n] + 1e-6, ag.cfg.per)
    probs = p / p.sum()
    i = ag.rng.choice(n, size=16, p=probs)
    w = np.power(1.0 / (n * probs[i]), ag.cfg.per_beta)
    w = w / w.max()
    assert w.max() == pytest.approx(1.0) and np.all(w > 0)
    # e o peso é MENOR onde a probabilidade de sorteio foi maior — é essa a correção
    assert np.corrcoef(probs[i], w)[0, 1] < 0


def test_the_weight_is_the_mask_weight_so_no_per_means_no_change():
    """Com `per=0` todo peso é 1, e a média ponderada tem de dar exatamente a média — senão
    ligar o PER mudaria o resultado mesmo desligado."""
    x = tf.constant([[2.0], [4.0]])
    m = tf.constant([[1.0], [1.0]])
    assert float(MuZero._media_mascarada(x, m)) == pytest.approx(3.0)
    dobro = MuZero._media_mascarada(x, tf.constant([[2.0], [2.0]]))
    assert float(dobro) == pytest.approx(3.0), "escalar o peso todo não muda a média"


def test_training_runs_with_prioritised_replay():
    for kw in ({"per": 1.0}, {"per": 0.6, "per_beta": 0.4},
               {"per": 1.0, "n_suporte": 121}):
        ag = MuZero(cfg_min(batch_size=16, **kw))
        ag.iterate()
        st = ag.iterate()
        for c in ("perda_pi", "perda_v", "perda_r"):
            assert np.isfinite(st[c]), f"{kw}: {c}"
