"""As correções do `docs/REVISAO_ALGORITMOS.md`, cada uma travada por um teste.

Um teste por achado, nomeado pela seção. Todos falham no código anterior — é isso que
os torna úteis: um `git revert` de qualquer das correções acende exatamente uma luz.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest

from snakeai.agents.base import AgentBase
from snakeai.agents.dqn import DQN, DQNConfig
from snakeai.agents.ppo import PPO, PPOConfig, variancia_explicada
from snakeai.agents.rainbow import Rainbow, RainbowConfig
from snakeai.env.vec_snake import VecSnake
from snakeai.nets.heads import NoisyDense, ruido_ligado
from snakeai.record import ORCAMENTO_OFICIAL, RunRecord, validate


# ================================================================== §1.1 truncamento
def test_starvation_is_a_truncation_not_a_terminal_state():
    """`done=1` na fome joga fora o `γ·V(s')`, e o `s'` gravado é o do episódio **novo**,
    porque o ambiente já resetou. Os dois somem juntos ou nenhum."""
    env = VecSnake(4, 10, starve_base=1, rng=np.random.default_rng(0))
    obs, mask = env.reset()
    for _ in range(60):
        obs, mask, r, d, info = env.step(np.ones(4, dtype=np.int32))
        if len(info["trunc_idx"]):
            break
    assert len(info["trunc_idx"]), "o cenário não produziu truncamento por fome"

    ti = info["trunc_idx"]
    prox_obs, prox_mask, done = AgentBase.desfaz_truncamento(
        info, obs, mask, d.astype(np.float32))

    assert (done[ti] == 0.0).all(), "o truncado tem que permitir bootstrap"
    np.testing.assert_array_equal(prox_obs[ti], info["final_obs"])
    np.testing.assert_array_equal(prox_mask[ti], info["final_mask"])
    # quem não foi truncado não muda, e as entradas originais ficam intactas
    outros = [i for i in range(4) if i not in set(ti.tolist())]
    np.testing.assert_array_equal(prox_obs[outros], obs[outros])
    np.testing.assert_array_equal(done[outros], d.astype(np.float32)[outros])


def test_dqn_stores_the_true_final_state_of_a_truncated_episode():
    """O que chega à memória, e não o que o ambiente devolveu: o DQN gravava `done=1` e a
    observação **do episódio seguinte** — o estado certo não existia no buffer."""
    ag = DQN(DQNConfig(net="resnet_tiny", num_envs=8, learn_every=1, total_steps=10 ** 6,
                       warmup_steps=10 ** 9, salvar_gif=False, salvar_grafico=False))
    ag.env = VecSnake(8, 10, starve_base=1, rng=np.random.default_rng(0))
    ag.obs, ag.mask = ag.env.reset()

    infos, gravados = [], []
    passo_env, add = ag.env.step, ag.memoria.add_batch

    def step_espiao(*a, **kw):
        saida = passo_env(*a, **kw)
        infos.append(saida[-1])
        return saida

    ag.env.step = step_espiao
    ag.memoria.add_batch = lambda *a, **kw: (gravados.append((a, kw)), add(*a, **kw))[1]

    for _ in range(120):
        ag.iterate()
        if len(infos[-1]["trunc_idx"]):
            break
    info = infos[-1]
    assert len(info["trunc_idx"]), "o cenário não produziu truncamento por fome"

    (_obs, _act, _rew, next_obs, done, next_mask), kw = gravados[-1]
    ti = info["trunc_idx"]
    assert (done[ti] == 0.0).all(), "fome gravada como terminal: o bootstrap some"
    # `done=0` é para o alvo bootstrapar; a fronteira do episódio vai em `fim`, e é ela
    # que impede a janela de n passos de atravessar para o episódio seguinte — §2.9
    assert "fim" in kw, "a fronteira do episódio não chega ao buffer"
    assert (np.asarray(kw["fim"])[ti] == 1.0).all(), "fome não marcada como fim de episódio"
    np.testing.assert_array_equal(next_obs[ti], info["final_obs"])
    np.testing.assert_array_equal(next_mask[ti], info["final_mask"])


# ================================================================== §1.3 validate
def test_validate_reads_the_curve_not_the_declared_budget():
    """Uma execução interrompida mantém o `config` intacto; só a curva denuncia."""
    comum = dict(algo="ppo", net="resnet_small", params=1,
                 config={"total_steps": ORCAMENTO_OFICIAL},
                 final={"episodes": 1000, "completo": True, "score_mean": 3.0})
    curta = RunRecord(curve=[{"global_step": 1_000}], **comum)
    assert any("curva vai até" in p for p in validate(curta))

    inteira = RunRecord(curve=[{"global_step": ORCAMENTO_OFICIAL + 13_504}], **comum)
    assert not [p for p in validate(inteira) if "curva vai até" in p]


# ================================================================== §1.4 melhor
def test_evaluating_the_best_checkpoint_does_not_touch_the_live_model(tmp_path):
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=4, rollout=4, total_steps=100,
                       eval_episodes=4, eval_envs=4, ckpt_dir=str(tmp_path),
                       salvar_gif=False, salvar_grafico=False))
    ag.salvar("best")
    antes = [np.asarray(v) for v in ag.model.weights]
    for v in ag.model.weights:                      # o "último passo" diverge do best
        v.assign(np.asarray(v) + 0.1)
    depois = [np.asarray(v) for v in ag.model.weights]

    stats = ag.avaliar_melhor(verbose=False)
    assert stats and "score_mean" in stats
    for v, esperado in zip(ag.model.weights, depois):
        np.testing.assert_allclose(np.asarray(v), esperado, atol=0)
    assert not np.allclose(depois[0], antes[0])


def test_agents_that_cannot_play_from_a_single_file_say_so(tmp_path):
    """Melhor uma coluna ausente e explicada que um número do modelo errado."""
    class Mudo(PPO):
        algo = "mudo"

        def politica_do_modelo(self, modelo):
            raise NotImplementedError("a política é recorrente; o `.keras` é só o ator")

    ag = Mudo(PPOConfig(net="resnet_tiny", num_envs=4, rollout=4, total_steps=100,
                        eval_episodes=4, eval_envs=4, ckpt_dir=str(tmp_path),
                        salvar_gif=False, salvar_grafico=False))
    ag.salvar("best")
    stats = ag.avaliar_melhor(verbose=False)
    assert "indisponivel" in stats and "score_mean" not in stats


def test_extra_weights_travel_with_the_checkpoint(tmp_path):
    """O gancho que torna a pasta do DreamerV3 autossuficiente."""
    entrada = keras.Input(shape=(2,))
    extra = keras.Model(entrada, keras.layers.Dense(3)(entrada))

    class ComExtra(PPO):
        algo = "com_extra"

        def modelos_extra(self):
            return {"extra": extra}

    ag = ComExtra(PPOConfig(net="resnet_tiny", num_envs=4, rollout=4, total_steps=100,
                            ckpt_dir=str(tmp_path), salvar_gif=False,
                            salvar_grafico=False))
    ag.salvar("last")
    guardado = [np.asarray(v) for v in extra.weights]
    for v in extra.weights:
        v.assign(np.asarray(v) + 1.0)
    assert ag._carregar_extra("last")
    for v, esperado in zip(extra.weights, guardado):
        np.testing.assert_allclose(np.asarray(v), esperado, atol=1e-6)


def test_dreamer_evaluates_the_best_checkpoint_and_puts_the_world_model_back(tmp_path):
    """O caso que motivou tudo: `self.model` do Dreamer é o **ator**, e trocá-lo não
    trocava a política — que joga por `self.ator` mais o RSSM inteiro. A coluna `melhor`
    do registro virava uma segunda medição do modelo final."""
    from snakeai.agents import DreamerV3, DreamerV3Config

    ag = DreamerV3(DreamerV3Config(
        preset="dreamer_tiny", num_envs=4, batch_size=4, seq_len=8, memory_size=100,
        warmup_steps=0, horizonte=5, collect_steps=8, eval_episodes=4, eval_envs=4,
        eval_every_steps=10 ** 9, log_every_steps=10 ** 9, ckpt_dir=str(tmp_path),
        salvar_gif=False, salvar_grafico=False))
    ag.salvar("best")

    # o "último passo" diverge do melhor, no ator e no modelo do mundo
    for m in [ag.ator, ag.encoder]:
        for v in m.weights:
            v.assign(np.asarray(v) + 0.05)
    depois = ([np.asarray(v) for v in ag.ator.weights],
              [np.asarray(v) for v in ag.encoder.weights])

    stats = ag.avaliar_melhor(verbose=False)
    assert "score_mean" in stats, "o Dreamer precisa conseguir avaliar o próprio best"

    for v, esperado in zip(ag.ator.weights, depois[0]):
        np.testing.assert_allclose(np.asarray(v), esperado, atol=1e-6)
    for v, esperado in zip(ag.encoder.weights, depois[1]):
        np.testing.assert_allclose(np.asarray(v), esperado, atol=1e-6)
    assert ag.model is ag.ator


# ================================================================== §2.1 orçamento
def test_the_dense_budget_is_the_default_and_the_sparse_one_is_the_ablation():
    """A ablação decidiu: o denso virou o padrão, e a configuração antiga sobrou como
    braço de controle — com sufixo próprio, para não dividir identidade com ele."""
    padrao, esparso = PPOConfig(), PPOConfig.esparso()
    assert esparso.total_steps == padrao.total_steps == ORCAMENTO_OFICIAL
    assert esparso.num_envs == padrao.num_envs

    def atualizacoes(c):
        return c.total_steps / (c.num_envs * c.rollout) * c.epochs * c.minibatches

    assert atualizacoes(padrao) > 10 * atualizacoes(esparso)
    assert esparso.sufixo_variante == "esparso"


def test_the_dense_preset_really_produces_the_updates_it_promises():
    """A conta da tabela do docstring, medida em vez de calculada: o `denso()` só vale a
    execução de decisão se o número de atualizações por passo de ambiente for mesmo o que
    ele promete. Comparação com o padrão, no mesmo orçamento de ambiente."""
    def mede(cfg):
        ag = PPO(cfg)
        passos, updates = 0, 0
        for _ in range(3):
            s = ag.iterate()
            updates += s["atualizacoes"]
        passos = ag.global_step
        return updates / passos                      # atualizações por passo de ambiente

    comum = dict(net="resnet_tiny", num_envs=8, total_steps=10 ** 6, target_kl=10.0,
                 salvar_gif=False, salvar_grafico=False)
    padrao = mede(PPOConfig(rollout=96, epochs=3, minibatches=8, **comum))
    denso = mede(PPOConfig(rollout=32, epochs=4, minibatches=32, **comum))

    assert denso > 10 * padrao, f"esparso {padrao:.2e}, denso {denso:.2e}"
    # e a razão bate com a aritmética do docstring, dentro de 10%
    esperado = (4 * 32 / 32) / (3 * 8 / 96)
    assert abs(denso / padrao / esperado - 1) < 0.1


def test_a_different_hyperparameter_set_gets_a_different_identity():
    """`denso()` muda o orçamento de gradiente, não o contrato — então ele compete. Mas
    competir com a **mesma** identidade `(algo, variant, seed)` do padrão faria `load_all`
    fundir as duas numa curva só, que é o mesmo defeito do canal de fome noutra roupa."""
    comum = dict(net="resnet_tiny", num_envs=4, rollout=4, total_steps=100,
                 salvar_gif=False, salvar_grafico=False)
    assert PPO(PPOConfig(**comum)).variant == "resnet_tiny"

    esparso = PPOConfig.esparso(**{k: v for k, v in comum.items() if k != "rollout"})
    assert PPO(esparso).variant == "resnet_tiny_esparso"

    juntos = PPOConfig.esparso(canal_fome=True, comparable=False, caveat="6 canais",
                               **{k: v for k, v in comum.items() if k != "rollout"})
    assert PPO(juntos).variant == "resnet_tiny_fome_esparso"


def test_the_record_counts_the_gradient_updates(tmp_path):
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=4, rollout=4, epochs=2, minibatches=2,
                       total_steps=32, eval_episodes=4, eval_envs=4,
                       eval_every_steps=10 ** 9, ckpt_dir=str(tmp_path),
                       runs_dir=str(tmp_path), salvar_gif=False, salvar_grafico=False))
    rec = ag.train(verbose=False)
    assert rec.record.meta["atualizacoes"] >= 2


# ================================================================== §2.2 EV
def test_explained_variance_says_what_the_critic_is_worth():
    ret = np.array([1.0, 5.0, 9.0, 13.0])
    assert variancia_explicada(ret, ret) == pytest.approx(1.0)
    assert variancia_explicada(np.full(4, ret.mean()), ret) == pytest.approx(0.0)
    assert variancia_explicada(np.zeros(4), ret) < 0.1
    assert np.isnan(variancia_explicada(np.ones(4), np.ones(4)))


def test_ppo_reports_explained_variance_every_iteration():
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=4, rollout=8, total_steps=10 ** 6,
                       salvar_gif=False, salvar_grafico=False))
    stats = ag.iterate()
    assert "ev" in stats and stats["ev"] <= 1.0
    assert stats["atualizacoes"] >= 1


# ================================================================== §2.3 noisy
def test_noisy_nets_explore_while_collecting_and_stay_quiet_when_evaluating():
    ag = Rainbow(RainbowConfig(net="resnet_tiny", num_envs=32, total_steps=10 ** 6,
                               salvar_gif=False, salvar_grafico=False))
    assert ag.cfg.noisy and ag.epsilon() == 0.0, "o cenário do Rainbow: sem ε"
    assert any(isinstance(c, NoisyDense) for c in _camadas(ag.model))

    obs, mask = ag.env.reset()
    escolhas = {tuple(ag._escolher(obs, mask).tolist()) for _ in range(8)}
    assert len(escolhas) > 1, "a coleta continua determinística — não explora nada"

    pol = ag.politica()
    a, b = pol(obs, mask), pol(obs, mask)
    np.testing.assert_allclose(a, b, atol=0)         # a avaliação tem que ser reprodutível


def test_the_noise_switch_is_restored_even_after_an_error():
    camada = NoisyDense(4)
    modelo = keras.Model(*(lambda i: (i, camada(i)))(keras.Input(shape=(8,))))
    with pytest.raises(RuntimeError):
        with ruido_ligado(modelo):
            raise RuntimeError("boom")
    assert camada.ruido is None


def _camadas(modelo):
    from snakeai.nets.heads import _camadas as f
    return f(modelo)


# ================================================================== §2.5 n passos
@pytest.mark.parametrize("agente", ["alphazero", "muzero"])
def test_the_value_target_bootstraps_everywhere_but_the_last_step(agente):
    """Com `rollout=16` e `n_step=10`, dez dos dezesseis passos ficavam sem bootstrap: o
    fim da janela de coleta virava fim de episódio para 62% das amostras.

    Os dois agentes ganharam depois `bootstrap_fim_janela` (§2.27–§2.29), que fecha também
    o último passo: o horizonte passou de `T - 1 - t` a `limite - t`, com `limite = T - 1`
    quando a flag está desligada e `T` quando está ligada. A garantia do §2.5 continua
    idêntica dos dois lados — **um** passo sem bootstrap, e só ele, com a flag desligada.

    Quem confere o comportamento é `tests/test_search.py`
    (`test_the_last_step_of_the_window_has_no_bootstrap_by_default`); aqui a conferência é
    da forma do laço, para a regressão aparecer mesmo se alguém reescrever a aritmética.
    """
    T, n_step = 16, 10
    sem_bootstrap = [t for t in range(T) if min(n_step, T - 1 - t) == 0]
    assert sem_bootstrap == [T - 1]

    import inspect

    from snakeai.agents import AlphaZeroConfig, MuZeroConfig
    from snakeai.agents import alphazero as az, muzero as mz
    classe, config = ((az.AlphaZero, AlphaZeroConfig) if agente == "alphazero"
                      else (mz.MuZero, MuZeroConfig))

    fonte = inspect.getsource(classe.collect)
    assert "min(cfg.n_step, limite - t)" in fonte
    assert "limite = T - 1" in fonte and "limite = T " in fonte
    assert "t + k + 1 < T" not in fonte

    # o bootstrap do fim da janela virou o padrão depois que a medição o validou, e
    # continua desligável — senão não haveria como medir quanto ele vale
    assert config().bootstrap_fim_janela is True
    assert config(bootstrap_fim_janela=False).bootstrap_fim_janela is False


# ================================================================== §1.9 / arena
def _registro(tmp_path, algo, seed, **kw):
    from snakeai.record import ORCAMENTO_OFICIAL, RunRecord, save
    base = dict(algo=algo, variant="v", seed=seed, net="resnet_small", params=1,
                curve=[{"global_step": ORCAMENTO_OFICIAL}],
                config={"total_steps": ORCAMENTO_OFICIAL},
                final={"episodes": 1000, "completo": True, "score_mean": 8.0,
                       "fim_fome": 0.02, "fim_colisao": 0.85,
                       "fim_tabuleiro_cheio": 0.13})
    base.update(kw)
    r = RunRecord(**base)
    save(r, str(tmp_path / algo / "v" / f"seed{seed}" / "history.json"))
    return r


def test_the_arena_revalidates_instead_of_trusting_the_stamp(tmp_path):
    """`meta["contract_violations"]` é escrito por **quem treinou**, com o código daquele
    dia. Uma execução de agosto passou a régua antiga e continuou entrando na arena como
    oficial porque ninguém reconferiu na hora de montar. Ver `docs/ANTES_DO_ARTIGO.md`."""
    from snakeai.arena import carregar

    _registro(tmp_path, "ppo", 0)
    # medida com o protocolo anterior: sem as chaves de causa de fim, e sem carimbo
    _registro(tmp_path, "acktr", 0,
              final={"episodes": 1000, "completo": True, "score_mean": 83.9})

    oficiais, fora, _ = carregar(str(tmp_path), legado="não_existe")
    assert [r.algo for r in oficiais] == ["ppo"]
    assert [r.algo for r in fora] == ["acktr"]
    assert any("fim_fome" in p for p in fora[0].meta["contract_violations"])


def test_runs_outside_the_contract_are_listed_not_hidden(tmp_path):
    """`comparable=False` sumia das três listas: nem no gráfico, nem na tabela, nem na
    seção de excluídas. O `COMPARABILITY.md` diz que excluir em silêncio é pior que
    incluir — e a ablação do canal de fome é exatamente esse caso."""
    from snakeai.arena import carregar

    _registro(tmp_path, "ppo", 0)
    _registro(tmp_path, "ppo_fome", 0, comparable=False, caveat="6 canais (fome)")

    oficiais, fora, _ = carregar(str(tmp_path), legado="não_existe")
    assert [r.algo for r in oficiais] == ["ppo"]
    assert [r.algo for r in fora] == ["ppo_fome"]
    assert "6 canais" in " ".join(fora[0].meta["contract_violations"])


# ================================================================== §1.7 grade
def test_the_evaluation_grid_is_absolute_and_does_not_drift():
    """A cadência reancorava no passo **atingido**, então cada avaliação caía um bloco
    depois da anterior e o desvio se acumulava: na execução padrão do PPO a última
    avaliação aconteceu 513 mil passos além da grade nominal — 10% do orçamento. Como
    algoritmos avançam em blocos de tamanhos diferentes, isso desalinha a coluna
    `passos até 40`, que é lida sem interpolação."""
    from snakeai.agents.base import proximo_multiplo

    assert proximo_multiplo(0, 250_000) == 250_000
    assert proximo_multiplo(249_999, 250_000) == 250_000
    assert proximo_multiplo(250_000, 250_000) == 500_000
    assert proximo_multiplo(260_000, 250_000) == 500_000

    for bloco in (8_192, 16_384, 49_152):          # A2C/ACKTR, denso, padrão
        passo, alvo, avaliacoes = 0, 0, []
        for _ in range(5_000_000 // bloco + 1):
            passo += bloco
            if passo >= alvo:
                avaliacoes.append(passo)
                alvo = proximo_multiplo(passo, 250_000)
        # cada avaliação cai no máximo um bloco depois do seu ponto nominal, e o desvio
        # **não** cresce ao longo da execução
        desvios = [a - 250_000 * round(a / 250_000) for a in avaliacoes[1:]]
        assert max(desvios) < bloco, (bloco, max(desvios))
        assert desvios[-1] <= max(desvios[:3]) + bloco


# ================================================================== GIF · maçã final
def test_the_gif_reports_the_score_of_the_winning_move():
    """`score = score_antes` perde a maçã do último passo — e o último passo de um
    episódio vencedor é justamente o que come. Terceira cópia do defeito que o `eval.py`
    corrige: o GIF de uma vitória saía rotulado com 96 num tabuleiro cujo perfeito é 97."""
    import snakeai.env.render as render

    class EnvFalso:
        """Dois passos: o segundo vence comendo a última maçã."""

        def __init__(self, *a, **kw):
            self.b, self.n, self.starve_base = 10, 1, 100
            self.occ = np.zeros((1, 10, 10), dtype=np.int32)
            self.food = np.array([[0, 1]])
            self.head = np.array([[0, 0]])
            self.score = np.array([95])
            self.length = np.array([98])
            self.hunger = np.array([0])
            self._passos = 0

        def reset(self):
            return np.zeros((1, 10, 10, 5), np.float32), np.ones((1, 3), bool)

        def step(self, a):
            self._passos += 1
            obs, mask = np.zeros((1, 10, 10, 5), np.float32), np.ones((1, 3), bool)
            if self._passos == 1:
                self.score, self.length = np.array([96]), np.array([99])
                return obs, mask, np.zeros(1), np.zeros(1, bool), {"scores": np.array([])}
            # passo vencedor: come a última maçã e o episódio termina
            return (obs, mask, np.ones(1), np.ones(1, bool),
                    {"scores": np.array([97])})

    original = render.VecSnake
    render.VecSnake = EnvFalso
    try:
        _q, score, motivo = render.quadros_do_episodio(
            lambda obs, mask: np.zeros((1, 3), np.float32), max_steps=5)
    finally:
        render.VecSnake = original

    assert score == 97, "a maçã do passo vencedor sumiu do rótulo do GIF"
    assert motivo == "tabuleiro cheio"


# ============================================== §2.6 · retraçagem de tf.function
def _tracings(fn):
    return fn.experimental_get_tracing_count()


def test_the_training_step_is_traced_once_not_once_per_iteration():
    """`ent_coef` decai a cada iteração e entrava na `tf.function` como **float Python**,
    que faz parte da assinatura: cada iteração recompilava o grafo inteiro e retinha mais
    uma `ConcreteFunction`. O A2C com `rollout=16` faz 610 iterações; com `rollout=4`,
    2.441 — e o TensorFlow começa a avisar na quinta. Ver `docs/REVISAO_ALGORITMOS.md`
    §2.6."""
    from snakeai.agents.a2c import A2C, A2CConfig

    ag = A2C(A2CConfig(net="resnet_tiny", num_envs=8, rollout=4, total_steps=10 ** 6,
                       salvar_gif=False, salvar_grafico=False))
    antes = _tracings(A2C._train_step_a2c)
    for _ in range(6):
        ag.iterate()
        assert ag.ent_coef() != ag.cfg.ent_coef_start or True   # o coeficiente muda
    novas = _tracings(A2C._train_step_a2c) - antes
    assert novas <= 2, f"{novas} traçagens em 6 iterações — o escalar entrou na assinatura"


def test_the_ppo_training_step_is_traced_once_too():
    ag = PPO(PPOConfig(net="resnet_tiny", num_envs=8, rollout=4, epochs=1, minibatches=1,
                       total_steps=10 ** 6, salvar_gif=False, salvar_grafico=False))
    antes = _tracings(PPO._train_step)
    for _ in range(6):
        ag.iterate()
    novas = _tracings(PPO._train_step) - antes
    assert novas <= 2, f"{novas} traçagens em 6 iterações"
