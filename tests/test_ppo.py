"""PPO.

Os testes daqui miram os detalhes que fazem um PPO treinar sem erro e aprender errado —
que é a categoria de bug mais cara em RL, porque a curva sobe um pouco e ninguém
desconfia. Em especial: a máscara aplicada de forma inconsistente entre rollout e update,
e o bootstrap do truncamento por fome.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
import tensorflow as tf

from snakeai.agents import PPO, PPOConfig, compute_gae
from snakeai.agents.ppo import policy_forward
from snakeai.eval import MASK_NEG
from snakeai.record import CONTRATO, validate


def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=16, rollout=8, total_steps=256,
                eval_every_steps=10**9, eval_episodes=50, eval_envs=25,
                minibatches=2, epochs=1, log_every_steps=10**9)
    base.update(kw)
    return PPOConfig(**base)


# ---------------------------------------------------------------------- GAE
def test_gae_matches_a_hand_computation():
    """Dois passos, sem terminação: dá para conferir com lápis."""
    rew = np.array([[1.0], [2.0]], dtype=np.float32)
    val = np.array([[0.5], [1.0]], dtype=np.float32)
    done = np.zeros((2, 1), dtype=np.float32)
    ultimo = np.array([3.0], dtype=np.float32)
    g, lam = 0.9, 0.8

    adv, ret = compute_gae(rew, val, done, ultimo, g, lam)
    d1 = 2.0 + g * 3.0 - 1.0
    d0 = 1.0 + g * 1.0 - 0.5
    assert adv[1, 0] == pytest.approx(d1, rel=1e-5)
    assert adv[0, 0] == pytest.approx(d0 + g * lam * d1, rel=1e-5)
    assert ret == pytest.approx(adv + val, rel=1e-5)


def test_gae_does_not_leak_across_episode_boundaries():
    """Se o valor do episódio seguinte vazasse, o agente aprenderia que durar é ruim."""
    rew = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    val = np.zeros((3, 1), dtype=np.float32)
    done = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)   # termina em t=1
    adv, _ = compute_gae(rew, val, done, np.zeros(1, dtype=np.float32), 0.99, 0.95)
    # a vantagem em t=1 é só a recompensa daquele passo — nada do t=2 entra
    assert adv[1, 0] == pytest.approx(1.0, rel=1e-6)


def test_gae_shape_and_dtype():
    rew = np.zeros((5, 4), dtype=np.float32)
    adv, ret = compute_gae(rew, rew.copy(), rew.copy(), np.zeros(4, np.float32), .99, .95)
    assert adv.shape == ret.shape == (5, 4)
    assert adv.dtype == np.float32


# ------------------------------------------------------------ máscara e razão
def test_update_reproduces_the_rollout_log_prob():
    """O detalhe que silenciosamente destrói um PPO com máscara.

    Se o update não reaplicar a máscara aos logits, o `log_prob` calculado lá não bate com
    o que gerou a ação; a razão do PPO deixa de valer 1 na primeira passada e o algoritmo
    passa a otimizar uma coisa que não existe. Aqui recalculamos o `log_prob` pelo caminho
    do update, com o modelo intacto, e exigimos que bata com o do rollout.
    """
    ag = PPO(cfg_min())
    lote, _ = ag.collect()

    logits, _ = ag.model(lote["obs"], training=False)
    logits = tf.where(lote["mask"], logits,
                      tf.fill(tf.shape(logits), MASK_NEG))
    logp_all = tf.nn.log_softmax(logits)
    logp = tf.gather(logp_all, lote["act"], batch_dims=1).numpy()

    assert np.allclose(logp, lote["logp"], atol=1e-4), \
        "log_prob do update não bate com o do rollout — a razão do PPO vira ruído"
    razao = np.exp(logp - lote["logp"])
    assert np.allclose(razao, 1.0, atol=1e-3)


def test_masked_actions_are_never_sampled():
    ag = PPO(cfg_min())
    lote, _ = ag.collect()
    permitida = lote["mask"][np.arange(len(lote["act"])), lote["act"]]
    assert permitida.all(), "o rollout escolheu uma ação que a máscara proibia"


def test_policy_forward_kills_masked_logits():
    ag = PPO(cfg_min())
    logits, valor = policy_forward(ag.model, tf.convert_to_tensor(ag.obs),
                                   tf.convert_to_tensor(ag.mask))
    logits = logits.numpy()
    assert (logits[~ag.mask] == MASK_NEG).all()
    assert tuple(valor.shape) == (ag.cfg.num_envs,)


# ----------------------------------------------------------------- truncamento
def test_starvation_gets_a_value_bootstrap():
    """Fome é truncamento: o episódio continuaria, então o valor final entra na conta.

    Sem o bootstrap, a recompensa do passo truncado seria só `-0,5`, e o agente aprenderia
    que ficar vivo muito tempo é ruim.
    """
    cfg = cfg_min(num_envs=8, rollout=200)
    ag = PPO(cfg)
    ag.env.starve_base = 6          # força fome cedo
    ag.obs, ag.mask = ag.env.reset()
    lote, stats = ag.collect()
    # com fome tão curta, é certo que houve truncamento no rollout
    assert stats["n_episodes"] > 0


def test_collect_returns_a_coherent_batch():
    cfg = cfg_min(num_envs=8, rollout=5)
    ag = PPO(cfg)
    lote, stats = ag.collect()
    n = cfg.num_envs * cfg.rollout
    assert lote["obs"].shape == (n, 10, 10, 5)
    assert lote["mask"].shape == (n, 3)
    assert lote["act"].shape == lote["logp"].shape == (n,)
    assert lote["adv"].shape == lote["ret"].shape == lote["val"].shape == (n,)
    for k, v in lote.items():
        if v.dtype != bool:
            assert np.isfinite(v).all(), f"{k} tem NaN ou inf"
    assert ag.global_step == n


# ------------------------------------------------------------- agendamentos
def test_schedules_move_from_start_to_end():
    cfg = cfg_min(total_steps=1000, lr_start=1e-3, lr_end=1e-4,
                  ent_coef_start=0.02, ent_coef_end=0.002)
    ag = PPO(cfg)
    assert ag.lr() == pytest.approx(1e-3)
    assert ag.ent_coef() == pytest.approx(0.02)
    ag.global_step = 1000
    assert ag.lr() == pytest.approx(1e-4)
    assert ag.ent_coef() == pytest.approx(0.002)


def test_shaping_decays_to_zero_and_stays_there():
    """Decair a zero é o que garante que a política ótima final seja a do jogo real."""
    cfg = cfg_min(total_steps=1000, shaping_start=0.5, shaping_frac=0.25)
    ag = PPO(cfg)
    assert ag.shaping() == pytest.approx(0.5)
    ag.global_step = 125
    assert ag.shaping() == pytest.approx(0.25)
    ag.global_step = 250
    assert ag.shaping() == pytest.approx(0.0)
    ag.global_step = 900
    assert ag.shaping() == 0.0


# ------------------------------------------------------------------- update
def test_update_changes_the_weights_and_reports_metrics():
    ag = PPO(cfg_min())
    antes = [w.numpy().copy() for w in ag.model.trainable_variables]
    lote, _ = ag.collect()
    logs = ag.update(lote)
    depois = [w.numpy() for w in ag.model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))
    for chave in ("pg", "vf", "ent", "kl", "clipfrac", "lr", "ent_coef"):
        assert chave in logs and np.isfinite(logs[chave])


def test_approx_kl_is_non_negative():
    """O estimador k3 é não-negativo por construção; um KL negativo denunciaria bug."""
    ag = PPO(cfg_min())
    lote, _ = ag.collect()
    logs = ag.update(lote)
    assert logs["kl"] >= -1e-6


def test_kl_early_stop_cuts_the_epochs():
    """Com `target_kl` praticamente zero, o update tem que parar na primeira passada."""
    ag = PPO(cfg_min(epochs=4, target_kl=1e-9, lr_start=1e-2, lr_end=1e-2))
    lote, _ = ag.collect()
    logs = ag.update(lote)
    assert logs["epochs_done"] == 1


# ---------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip(tmp_path):
    """O Colab derruba a sessão — é questão de quando, não de se."""
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = PPO(cfg)
    ag.iterate()
    ag.salvar("last")

    outro = PPO(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    assert outro.global_step == ag.global_step
    assert outro.episodes == ag.episodes
    x = ag.obs[:4]
    m = ag.mask[:4]
    a = policy_forward(ag.model, tf.convert_to_tensor(x), tf.convert_to_tensor(m))[0]
    b = policy_forward(outro.model, tf.convert_to_tensor(x), tf.convert_to_tensor(m))[0]
    assert np.allclose(a.numpy(), b.numpy(), atol=1e-5)


def test_resume_rebuilds_the_optimizer(tmp_path):
    """O otimizador antigo aponta para as variáveis do modelo antigo — e explode."""
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = PPO(cfg); ag.iterate(); ag.salvar("last")
    outro = PPO(cfg_min(ckpt_dir=str(tmp_path)))
    outro.retomar("last")
    outro.iterate()          # se o otimizador não fosse reconstruído, quebraria aqui
    assert outro.global_step > ag.global_step


def test_resume_on_empty_dir_returns_false(tmp_path):
    assert PPO(cfg_min(ckpt_dir=str(tmp_path))).retomar() is False


# ------------------------------------------------------------------ contrato
def test_board_size_outside_the_contract_is_refused():
    with pytest.raises(ValueError, match="contrato"):
        PPOConfig(board_size=20)


def test_a_short_run_produces_a_record_that_fails_the_contract(tmp_path):
    """Execução de fumaça não pode virar resultado oficial — o portão tem que barrar."""
    cfg = cfg_min(total_steps=256, ckpt_dir=str(tmp_path), runs_dir=str(tmp_path),
                  eval_episodes=50, eval_envs=25)
    rec = PPO(cfg).train(verbose=False)
    problemas = validate(rec.record)
    assert any("episódios" in p for p in problemas)
    assert rec.record.algo == "ppo"
    assert rec.record.params > 0
    assert rec.record.env_spec == CONTRATO


def test_seed_makes_the_run_reproducible():
    a = PPO(cfg_min(seed=7)); a.iterate()
    b = PPO(cfg_min(seed=7)); b.iterate()
    assert a.global_step == b.global_step
    xa = [w.numpy() for w in a.model.trainable_variables]
    xb = [w.numpy() for w in b.model.trainable_variables]
    assert all(np.allclose(p, q, atol=1e-5) for p, q in zip(xa, xb))


# --------------------------------------------- o log que parecia instabilidade
def test_training_log_is_a_moving_average_not_the_last_iteration():
    """`2,50 · 10,00 · — · — · 2,00 · 11,00` não era o algoritmo oscilando.

    Era o log imprimindo a média dos episódios que por acaso terminaram **naquela**
    iteração — amostra de tamanho 0 a 3. O `—` é "nenhum episódio acabou agora". Quanto
    melhor o agente fica, mais longos os episódios e mais frequente o traço, então o
    defeito piora exatamente quando o treino está indo bem.
    """
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, salvar_gif=False, salvar_grafico=False))
    assert ag.media_movel() is None, "sem episódio nenhum, não há média"

    ag._registra_episodios(10.0, 1)     # uma iteração com 1 episódio de score 10
    ag._registra_episodios(2.0, 20)     # outra com 20 episódios de score 2
    # ponderado pela quantidade: (10 + 40) / 21, e não a média das médias (6,0)
    assert ag.media_movel() == pytest.approx(50 / 21)


def test_the_curve_records_the_moving_average_and_keeps_the_raw_value():
    """A curva precisa do número estável; o valor cru continua disponível para quem
    quiser ver a variância entre iterações."""
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=16, rollout=4, minibatches=1,
                       epochs=1, total_steps=400, eval_every_steps=10 ** 9,
                       log_every_steps=1, salvar_gif=False, salvar_grafico=False))
    ag.train(verbose=False)
    pontos = [p for p in ag.history if "train_score_mean" in p]
    assert pontos, "nada foi registrado"
    assert any("train_score_iter" in p for p in pontos)


def test_the_run_folder_carries_the_models(tmp_path):
    """A pasta da execução tem que ser autossuficiente.

    Os checkpoints vivem em `ckpt_dir`, que a execução seguinte sobrescreve. Sem a cópia,
    o `history.json` afirma um score que ninguém consegue reproduzir — e num repositório
    que existe para tornar resultados comparáveis, um número sem o modelo que o produziu
    é exatamente o que não serve.
    """
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, total_steps=200, eval_episodes=20, eval_envs=10,
                       eval_every_steps=100, log_every_steps=10 ** 9,
                       ckpt_dir=str(tmp_path / "ckpt"), runs_dir=str(tmp_path / "runs"),
                       salvar_gif=False, salvar_grafico=False))
    rec = ag.train(verbose=False)

    pasta = tmp_path / "runs" / "ppo" / ag.variant / "seed0" / "modelos"
    assert (pasta / "last.keras").exists()
    assert (pasta / "best.keras").exists(), "o melhor checkpoint também vai junto"


def test_the_record_carries_both_results(tmp_path):
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, total_steps=200, eval_episodes=20, eval_envs=10,
                       eval_every_steps=100, log_every_steps=10 ** 9,
                       ckpt_dir=str(tmp_path / "ckpt"), runs_dir=str(tmp_path / "runs"),
                       salvar_gif=False, salvar_grafico=False))
    rec = ag.train(verbose=False)
    assert "score_mean" in rec.record.final
    assert "score_mean" in rec.record.melhor
    assert "global_step" in rec.record.melhor, "sem o passo, o número fica sem endereço"


def test_the_window_is_measured_in_episodes_not_in_iterations():
    """A correção do defeito que a média móvel tinha na primeira versão.

    A janela era de 200 **iterações**. Numa iteração de PPO cabem 512 × 96 = 49.152 passos
    e ~200 episódios, então 200 iterações cobriam ~10 M passos: a execução inteira. A
    "média móvel" era, na prática, **média acumulada** — e média acumulada é arrastada
    para sempre pelos episódios ruins do começo. Foi por isso que um PPO com avaliação em
    32,2 aparecia no log com "treino 15,76": o 15,76 incluía o score 1 dos primeiros
    episódios.
    """
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, salvar_gif=False, salvar_grafico=False))
    ag.JANELA_EPISODIOS = 500

    for _ in range(10):                       # 10 iterações × 200 episódios ruins
        ag._registra_episodios(1.0, 200)
    for _ in range(3):                        # e agora o agente aprendeu
        ag._registra_episodios(30.0, 200)

    assert ag.episodios_na_janela() <= 800, "a janela não pode crescer sem limite"
    assert ag.media_movel() > 25, \
        f"a média móvel ficou em {ag.media_movel():.1f} — está arrastando o passado"


def test_the_window_never_empties_even_with_huge_iterations():
    """Se uma única iteração já produz mais episódios que a janela, a janela é ela.

    Descartar mesmo assim deixaria a média sem nenhum episódio para calcular.
    """
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, salvar_gif=False, salvar_grafico=False))
    ag.JANELA_EPISODIOS = 100
    ag._registra_episodios(5.0, 5000)
    ag._registra_episodios(9.0, 5000)

    assert len(ag._janela) == 1
    assert ag.media_movel() == pytest.approx(9.0), "a janela é a última iteração"


def test_a_slow_algorithm_still_accumulates_a_meaningful_window():
    """O outro extremo: uma iteração de DQN termina 2 ou 3 episódios. A janela precisa
    juntar muitas iterações para o número não pular a cada log."""
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, minibatches=1,
                       epochs=1, salvar_gif=False, salvar_grafico=False))
    ag.JANELA_EPISODIOS = 500
    for _ in range(400):
        ag._registra_episodios(3.0, 2)
    assert 500 <= ag.episodios_na_janela() <= 502
