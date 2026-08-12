"""ACER.

O algoritmo que o repositório antigo tentou três vezes e nunca fez rodar. Os dois primeiros
testes existem para que as duas formas de morte não voltem: a dimensão de tempo perdida e a
matemática montada dentro do grafo funcional do Keras.

Depois vêm as propriedades do Retrace — que é a parte onde um erro de sinal ou de
deslocamento no tempo produz um algoritmo que treina, não reclama, e aprende a coisa
errada.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import ACER, ACERConfig, retrace
from snakeai.env.vec_snake import N_ACTIONS
from snakeai.eval import MASK_NEG
from snakeai.memory import TrajectoryBuffer


def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=4, total_steps=2000,
                warmup_segments=2, replay_ratio=1, memory_size=8,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return ACERConfig(**base)


# ------------------------------------------------- os dois bugs que não voltam
def test_stored_segment_keeps_the_time_axis():
    """O `ValueError: expected shape=(None, 256, 100), found (None, 100)` do ACER legado.

    A dimensão de tempo tinha sumido entre a coleta e o update. Aqui o buffer recusa
    qualquer coisa que não venha `(T, N, ...)`.
    """
    buf = TrajectoryBuffer(4)
    T, N = 5, 3
    ok = dict(
        obs_final=np.zeros((N, 10, 10, 5), np.float32),
        mask_final=np.ones((N, 3), bool),
        obs=np.zeros((T, N, 10, 10, 5), np.float32),
        mask=np.ones((T, N, 3), bool),
        act=np.zeros((T, N), np.int32),
        mu=np.ones((T, N, 3), np.float32) / 3,
        rew=np.zeros((T, N), np.float32),
        done=np.zeros((T, N), np.float32),
    )
    seg = buf.add(**ok)
    assert seg["obs"].shape[:2] == (T, N)
    assert seg["mu"].shape == (T, N, 3)

    achatado = dict(ok, obs=np.zeros((T * N, 10, 10, 5), np.float32))
    with pytest.raises(ValueError, match="eixo de tempo"):
        buf.add(**achatado)


def test_segment_stores_its_own_final_state():
    """O bootstrap do Retrace tem que sair do fim DAQUELA trajetória.

    Usar o estado atual do ambiente para fazer bootstrap de um segmento guardado há mil
    iterações não levanta exceção nenhuma — só ensina o valor errado, em silêncio.
    """
    ag = ACER(cfg_min())
    seg, _ = ag.collect()
    assert "obs_final" in seg and "mask_final" in seg
    assert seg["obs_final"].shape == (ag.cfg.num_envs, 10, 10, 5)
    guardado = seg["obs_final"].copy()

    for _ in range(3):                      # o ambiente anda; o segmento guardado não
        ag.collect()
    assert np.array_equal(ag.memoria.dados[0]["obs_final"], guardado)
    assert not np.array_equal(ag.obs, guardado), "o ambiente deveria ter avançado"


def test_the_model_is_only_input_to_outputs():
    """O `TypeError` do ACER legado: matemática montada dentro do grafo funcional.

    O modelo tem que ser só `entrada -> [logits, Q]`. Toda a lógica do ACER acontece em
    `tf.function` sobre tensores concretos, onde `tf.cond`, `tf.gather` e companhia
    funcionam.
    """
    ag = ACER(cfg_min())
    assert len(ag.model.outputs) == 2
    logits, q = ag.model.outputs
    assert logits.shape[-1] == N_ACTIONS
    assert q.shape[-1] == N_ACTIONS, "o crítico do ACER é Q(s,·), não um escalar"


# ------------------------------------------------------------------- Retrace
def test_retrace_reduces_to_the_monte_carlo_return_when_rho_is_one():
    """Com ρ̄ = 1 e Q = V, o Retrace vira o retorno descontado puro — conferível à mão."""
    T, N = 3, 1
    rew = np.array([[1.0], [1.0], [1.0]], np.float32)
    done = np.zeros((T, N), np.float32)
    v = np.zeros((T, N), np.float32)
    q_a = np.zeros((T, N), np.float32)
    rho = np.ones((T, N), np.float32)
    g = 0.5

    ret = retrace(rew, done, q_a, v, rho, np.zeros(N, np.float32), g)
    assert ret[2, 0] == pytest.approx(1.0)
    assert ret[1, 0] == pytest.approx(1.0 + g * 1.0)
    assert ret[0, 0] == pytest.approx(1.0 + g * (1.0 + g * 1.0))


def test_retrace_stops_at_terminal_states():
    """Terminação corta a recursão: o retorno do episódio seguinte não pode vazar."""
    T, N = 3, 1
    rew = np.array([[1.0], [5.0], [100.0]], np.float32)
    done = np.array([[0.0], [1.0], [0.0]], np.float32)
    z = np.zeros((T, N), np.float32)
    ret = retrace(rew, done, z, z, np.ones((T, N), np.float32),
                  np.zeros(N, np.float32), 0.9)
    assert ret[1, 0] == pytest.approx(5.0), "o passo terminal só carrega a própria recompensa"


def test_retrace_truncation_shrinks_the_correction():
    """ρ̄ = min(1, ρ) nunca amplifica — é o que torna dados velhos seguros de usar."""
    T, N = 4, 1
    rew = np.zeros((T, N), np.float32)
    done = np.zeros((T, N), np.float32)
    v = np.full((T, N), 1.0, np.float32)
    q_a = np.zeros((T, N), np.float32)            # discrepância Q^ret − Q é grande
    ultimo = np.ones(N, np.float32)

    forte = retrace(rew, done, q_a, v, np.ones((T, N), np.float32), ultimo, 0.99)
    fraco = retrace(rew, done, q_a, v, np.full((T, N), 0.1, np.float32), ultimo, 0.99)
    assert abs(forte[0, 0]) > abs(fraco[0, 0]), \
        "ρ̄ menor deveria encolher a propagação da correção"


def test_retrace_shapes():
    T, N = 6, 4
    z = np.zeros((T, N), np.float32)
    ret = retrace(z, z, z, z, np.ones((T, N), np.float32), np.zeros(N, np.float32), .99)
    assert ret.shape == (T, N) and ret.dtype == np.float32


# ------------------------------------------------------------------- rollout
def test_collect_records_the_behavior_policy():
    """Sem μ gravado, a razão π/μ seria π/π = 1 e o ACER viraria um A2C caro."""
    ag = ACER(cfg_min())
    seg, stats = ag.collect()
    assert seg["mu"].shape == (ag.cfg.rollout, ag.cfg.num_envs, N_ACTIONS)
    assert np.allclose(seg["mu"].sum(-1), 1.0, atol=1e-4), "μ tem que ser distribuição"
    escolhidas = seg["mu"][np.indices(seg["act"].shape)[0],
                           np.indices(seg["act"].shape)[1], seg["act"]]
    assert (escolhidas > 0).all(), "ação com probabilidade zero foi amostrada"


def test_collect_never_takes_a_masked_action():
    ag = ACER(cfg_min())
    seg, _ = ag.collect()
    T, N = seg["act"].shape
    it, iN = np.indices((T, N))
    assert seg["mask"][it, iN, seg["act"]].all()


def test_segments_accumulate_in_memory():
    ag = ACER(cfg_min(memory_size=3))
    for _ in range(5):
        ag.collect()
    assert len(ag.memoria) == 3, "o buffer é circular"


# -------------------------------------------------------------------- update
def test_iterate_does_on_policy_plus_replay_updates():
    """O replay ratio é a razão de existir do ACER — 1 update on-policy + k off-policy."""
    ag = ACER(cfg_min(replay_ratio=3, warmup_segments=1))
    ag.iterate()                       # enche a memória
    stats = ag.iterate()
    assert stats["updates"] == 4, "1 on-policy + 3 do replay"


def test_no_replay_before_warmup():
    ag = ACER(cfg_min(replay_ratio=3, warmup_segments=99))
    stats = ag.iterate()
    assert stats["updates"] == 1


def test_update_changes_weights_and_reports_finite_metrics():
    ag = ACER(cfg_min())
    antes = [w.numpy().copy() for w in ag.model.trainable_variables]
    stats = ag.iterate()
    depois = [w.numpy() for w in ag.model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))
    for chave in ("loss_q", "entropia", "rho_medio"):
        assert np.isfinite(stats[chave]), f"{chave} virou NaN"


def test_importance_ratio_starts_near_one():
    """No update on-policy, π e μ são a mesma política — ρ tem que valer ~1."""
    ag = ACER(cfg_min(replay_ratio=0, warmup_segments=99))
    stats = ag.iterate()
    assert 0.8 < stats["rho_medio"] < 1.25


@pytest.mark.parametrize("trust_region", [False, True])
def test_trust_region_toggle_both_train(trust_region):
    ag = ACER(cfg_min(trust_region=trust_region))
    stats = ag.iterate()
    assert np.isfinite(stats["loss_q"])


def test_anchor_policy_lags_behind_the_online_one():
    """A média de Polyak é a âncora: se acompanhasse na hora, não restringiria nada."""
    ag = ACER(cfg_min(polyak=0.99))
    for _ in range(3):
        ag.iterate()
    difs = [not np.allclose(a, b)
            for a, b in zip(ag.model.get_weights(), ag.media.get_weights())]
    assert any(difs)


def test_anchor_is_not_updated_without_trust_region():
    ag = ACER(cfg_min(trust_region=False))
    antes = [w.copy() for w in ag.media.get_weights()]
    ag.iterate()
    assert all(np.allclose(a, b) for a, b in zip(antes, ag.media.get_weights()))


# ------------------------------------------------------------------ avaliação
def test_greedy_policy_is_deterministic_and_masked():
    ag = ACER(cfg_min())
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a, b = fn(obs, mask), fn(obs, mask)
    assert np.array_equal(a, b)
    assert (a[~mask] == MASK_NEG).all()


def test_checkpoint_roundtrip_rebuilds_the_anchor(tmp_path):
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = ACER(cfg); ag.iterate(); ag.salvar("last")
    outro = ACER(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    assert outro.media is not None
    outro.iterate()


def test_full_train_writes_a_record(tmp_path):
    cfg = cfg_min(total_steps=300, ckpt_dir=str(tmp_path / "ck"),
                  runs_dir=str(tmp_path / "runs"))
    rec = ACER(cfg).train(verbose=False)
    assert rec.record.algo == "acer"
    assert (tmp_path / "runs" / "acer" / "resnet_tiny" / "seed0" / "history.json").exists()
