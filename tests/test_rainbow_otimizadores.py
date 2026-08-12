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


def test_kfac_is_gone_and_documented():
    """O K-FAC não volta: `tensorflow.contrib` não existe desde o TF2."""
    import snakeai.otimizadores as mod

    assert "kfac" not in OTIMIZADORES
    assert "K-FAC" in mod.__doc__ and "contrib" in mod.__doc__
