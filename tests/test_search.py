"""MCTS e AlphaZero.

O teste mais importante deste arquivo é `test_search_beats_random_with_an_informative_value`:
uma busca que não joga melhor que o acaso, quando lhe dão um valor informativo, está
quebrada — e quebra de forma silenciosa, porque o código roda, os números saem, e só o
score denuncia. Foi assim que dois bugs desta implementação apareceram.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.agents import AlphaZero, AlphaZeroConfig
from snakeai.env.vec_snake import N_ACTIONS, VecSnake
from snakeai.eval import MASK_NEG
from snakeai.search import MCTS, MinMax, No


def uniforme(obs, mask):
    p = np.asarray(mask, dtype=np.float64)
    p /= p.sum(1, keepdims=True)
    return p, np.zeros(len(obs), dtype=np.float32)


def heuristica(obs, mask):
    """Valor = −distância de Manhattan até a comida, lida da própria observação."""
    n, b = obs.shape[0], obs.shape[1]
    cab = obs[..., 1].reshape(n, -1).argmax(1)
    com = obs[..., 3].reshape(n, -1).argmax(1)
    d = np.abs(cab // b - com // b) + np.abs(cab % b - com % b)
    p = np.asarray(mask, dtype=np.float64)
    p /= p.sum(1, keepdims=True)
    return p, (-d / (2.0 * b)).astype(np.float32)


def heuristica_deslocada(obs, mask):
    """A MESMA heurística, somada de uma constante.

    Ordena os estados exatamente igual — nenhuma decisão *deveria* mudar. E é a forma que
    uma cabeça de valor treinada neste jogo assume: a recompensa é `+1` por maçã, a cabeça
    é linear e com `γ = 0,997` o ponto fixo do valor é `1/(1 − γ**12) ≈ 28`, positivo em
    toda parte. Ver `docs/BUSCA_DEGENERADA.md`.
    """
    p, v = heuristica(obs, mask)
    return p, (v + 1.0).astype(np.float32)


def joga(avaliador, sims, n=32, alvo=32, seed=123, **kw):
    env = VecSnake(n, 10, rng=np.random.default_rng(seed))
    obs, mask = env.reset()
    mcts = MCTS(avaliador, num_simulations=sims, gamma=0.997,
                rng=np.random.default_rng(0), **kw)
    scores = []
    while len(scores) < alvo:
        visitas, _ = mcts.run(env.get_state(), mask, obs)
        obs, mask, r, d, info = env.step(visitas.argmax(1).astype(np.int32))
        scores.extend(info["scores"].tolist())
    return float(np.mean(scores[:alvo]))


# ------------------------------------------------------- estado do ambiente
def test_state_roundtrip_is_exact():
    """A busca inteira depende disto: restaurar um nó tem que devolver o jogo idêntico."""
    env = VecSnake(8, 10, rng=np.random.default_rng(0))
    env.reset()
    for _ in range(30):
        env.step(np.ones(8, np.int32))
    estado = env.get_state()
    obs_a, mask_a = env.obs(), env.action_mask()

    for _ in range(50):
        env.step(np.zeros(8, np.int32))
    env.set_state(estado)
    env.check_invariants()

    assert np.array_equal(env.obs(), obs_a)
    assert np.array_equal(env.action_mask(), mask_a)


def test_restored_state_produces_the_same_transition():
    env = VecSnake(4, 10, rng=np.random.default_rng(1))
    env.reset()
    estado = env.get_state()
    a = np.array([0, 1, 2, 1], np.int32)
    _, _, r1, d1, _ = env.step(a)
    env.set_state(estado)
    _, _, r2, d2, _ = env.step(a)
    assert np.array_equal(r1, r2) and np.array_equal(d1, d2)


# ----------------------------------------------------------------- a busca
def test_run_returns_visit_counts_and_values():
    env = VecSnake(6, 10, rng=np.random.default_rng(0))
    obs, mask = env.reset()
    mcts = MCTS(uniforme, num_simulations=8)
    visitas, valores = mcts.run(env.get_state(), mask, obs)
    assert visitas.shape == (6, N_ACTIONS)
    assert valores.shape == (6,)
    assert (visitas.sum(1) > 0).all()


def test_search_never_visits_a_masked_action():
    """A máscara vale dentro da árvore também — senão a busca planeja morrer."""
    env = VecSnake(32, 10, rng=np.random.default_rng(2))
    obs, mask = env.reset()
    mcts = MCTS(uniforme, num_simulations=16)
    for _ in range(15):
        visitas, _ = mcts.run(env.get_state(), mask, obs)
        assert (visitas[~mask] == 0).all(), "a busca visitou uma ação proibida"
        obs, mask, *_ = env.step(visitas.argmax(1).astype(np.int32))


def test_search_beats_random_with_an_informative_value():
    """O teste que pega bug silencioso na busca.

    Com um valor que sabe onde está a comida, o MCTS tem que jogar MUITO melhor que o
    acaso. A primeira versão deste módulo jogava *pior*, porque o PUCT usava só o valor do
    filho e ignorava a recompensa de chegar até ele — um filho alcançado morrendo tinha
    valor 0 e parecia tão bom quanto um seguro.
    """
    env = VecSnake(32, 10, rng=np.random.default_rng(123))
    obs, mask = env.reset()
    rng = np.random.default_rng(0)
    scores = []
    while len(scores) < 32:
        p = mask.astype(np.float64); p /= p.sum(1, keepdims=True)
        a = (p.cumsum(1) > rng.random((32, 1))).argmax(1).astype(np.int32)
        obs, mask, r, d, info = env.step(a)
        scores.extend(info["scores"].tolist())
    piso = float(np.mean(scores[:32]))

    com_busca = joga(heuristica, sims=12)
    assert com_busca > 5 * max(piso, 0.5), \
        f"busca ({com_busca:.2f}) não superou o acaso ({piso:.2f}) com valor informativo"


def test_more_simulations_do_not_hurt():
    """Mais computação não pode piorar o jogo. Se piorar, o backup está errado."""
    poucas = joga(heuristica, sims=6)
    muitas = joga(heuristica, sims=24)
    assert muitas > poucas * 0.75, f"{muitas:.2f} contra {poucas:.2f} com 4x mais busca"


def test_terminal_children_are_never_expanded():
    """O `VecSnake` reseta sozinho ao terminar.

    Se o nó terminal guardasse o estado devolvido pelo `step`, a árvore teria uma partida
    NOVA e aleatória enxertada dentro dela — e a busca planejaria sobre um jogo que não
    existe. Nó terminal não tem estado nem filhos: vale 0 e acabou.

    Para forçar o caso, colocamos a fome a um passo do limite: qualquer ação termina o
    episódio. Num tabuleiro novo a máscara evita morte por vários níveis de profundidade, e
    o caso nunca apareceria.
    """
    env = VecSnake(8, 10, starve_base=10, rng=np.random.default_rng(5))
    obs, mask = env.reset()
    env.hunger[:] = env.starve_base + 2 * env.length - 1     # o próximo passo é fome
    mcts = MCTS(uniforme, num_simulations=12, starve_base=env.starve_base,
                rng=np.random.default_rng(0))
    mcts.run(env.get_state(), mask, obs)

    def varrer(no, achados):
        for filho in no.filhos.values():
            achados.append(filho)
            varrer(filho, achados)
        return achados

    terminais = 0
    for raiz in mcts._ultimas_raizes:
        for no in varrer(raiz, []):
            if no.terminal:
                terminais += 1
                assert no.estado is None, "nó terminal guardou o estado pós-reset"
                assert not no.filhos, "nó terminal foi expandido"
    assert terminais > 0, "o cenário deveria produzir nós terminais"


def test_search_env_inherits_the_training_env_config():
    """A árvore tem que simular o MESMO jogo. Regra de fome diferente = mundo diferente."""
    mcts = MCTS(uniforme, board_size=10, starve_base=7)
    assert mcts._ambiente(4).starve_base == 7
    padrao = MCTS(uniforme, board_size=10)
    assert padrao._ambiente(4).starve_base == 100


def test_visit_policy_temperature():
    visitas = np.array([[1.0, 9.0, 0.0]])
    quente = MCTS.politica_das_visitas(visitas, 1.0)
    fria = MCTS.politica_das_visitas(visitas, 1e-9)
    assert quente[0, 1] == pytest.approx(0.9)
    assert fria[0, 1] == pytest.approx(1.0)
    assert np.allclose(quente.sum(1), 1.0) and np.allclose(fria.sum(1), 1.0)


def test_dirichlet_noise_changes_the_root_priors():
    env = VecSnake(16, 10, rng=np.random.default_rng(0))
    obs, mask = env.reset()
    limpo = MCTS(uniforme, num_simulations=16, rng=np.random.default_rng(0))
    ruid = MCTS(uniforme, num_simulations=16, rng=np.random.default_rng(0))
    va, _ = limpo.run(env.get_state(), mask, obs, adicionar_ruido=False)
    vb, _ = ruid.run(env.get_state(), mask, obs, adicionar_ruido=True)
    assert not np.array_equal(va, vb)


# ---------------------------------------------- escala do valor dentro do PUCT
def test_the_fixes_are_the_default_and_can_all_be_turned_off():
    """Os onze consertos do §2.27–§2.29 são o padrão desde que a medição os validou.

    Duas metades, e as duas importam. A primeira: o padrão do `AlphaZeroConfig` **é** o
    agente consertado — quem roda `06_alphazero` sem tocar em nada roda a versão boa. A
    segunda: cada conserto continua desligável, senão o `93_alphazero_ablacoes` não teria
    como medir quanto cada um valeu, e uma melhoria que não dá para desligar é uma
    afirmação sem controle.

    O `MCTS` continua nascendo com a convenção do paper (`fpu="zero"`, sem normalização):
    a escala do valor é propriedade do **agente**, não da busca, e o MuZero ainda não foi
    tocado. Quem decide é quem sabe a escala.
    """
    cfg = AlphaZeroConfig()
    assert (cfg.fpu, cfg.q_normalizado) == ("pai", True)
    assert (cfg.valor_symlog, cfg.vf_coef) == (True, 0.5)
    assert (cfg.temp_alvo, cfg.temp_passos) == (1.0, 30)
    assert (cfg.epochs_por_iter, cfg.lr_final) == (8, 5e-5)
    assert (cfg.desempate, cfg.bootstrap_fim_janela) == ("aleatorio", True)
    assert cfg.dirichlet_alpha == 1.0

    velho = AlphaZeroConfig(fpu="zero", q_normalizado=False, valor_symlog=False,
                            vf_coef=1.0, epochs_por_iter=1, lr_final=0.0, temp_alvo=0.0,
                            temp_passos=0, dirichlet_alpha=0.5, desempate="ordem",
                            bootstrap_fim_janela=False)
    assert velho.fpu == "zero" and velho.bootstrap_fim_janela is False

    m = MCTS(uniforme)
    assert m.fpu == "zero" and m.q_normalizado is False and m.desempate == "ordem"


def test_search_collapses_when_the_value_is_positive_and_q_is_unnormalized():
    """A caracterização do bug, para que ele não volte sem ninguém notar.

    Somar uma constante ao valor não muda o ranking de estado nenhum, mas muda tudo no
    PUCT: um filho não visitado vale `0` por convenção, e contra irmãos que valem `V > 0`
    ele nunca é escolhido — o bônus de exploração `c_puct·P·√N` não cobre a diferença. A
    busca colapsa no primeiro filho que tocou e a cobra anda em círculo até morrer de fome.

    `test_search_beats_random_with_an_informative_value` não pega isto porque a heurística
    dele é **negativa**: ali o `0` é otimista e força exploração.
    """
    inteiro = joga(heuristica, sims=8, alvo=12)
    deslocado = joga(heuristica_deslocada, sims=8, alvo=12)
    assert deslocado < 0.25 * inteiro, (
        f"o deslocamento deixou de degradar a busca ({deslocado:.2f} contra "
        f"{inteiro:.2f}) — se foi de propósito, este teste virou obsoleto")


@pytest.mark.parametrize("conserto", [{"q_normalizado": True}, {"fpu": "pai"}])
def test_search_is_invariant_to_a_constant_shift_in_the_value(conserto):
    """O que o conserto tem que entregar: o mesmo jogo com o valor deslocado.

    Os dois consertos atacam o mesmo problema por lados diferentes — `q_normalizado`
    devolve o Q à faixa [0, 1] em que `c_puct` foi calibrado (MuZero, Apêndice B), `fpu`
    troca o palpite do filho virgem de `0` para o valor do pai.
    """
    inteiro = joga(heuristica, sims=8, alvo=12, **conserto)
    deslocado = joga(heuristica_deslocada, sims=8, alvo=12, **conserto)
    assert deslocado > 0.5 * inteiro, \
        f"{conserto}: {deslocado:.2f} contra {inteiro:.2f} com o valor deslocado de +1"

def test_the_normalized_fpu_stays_inside_the_measured_range():
    """`fpu="pai"` sob normalização precisa caber em `[0, 1]` — e não cabia.

    O `MinMax` é alimentado com **Q** (`r + γ·V`) e `no.valor` é um **V**, e o valor do nó
    continua se movendo depois de a faixa registrar o dele. Com a faixa ainda estreita, uma
    diferença absoluta minúscula vira um normalizado grande: medido numa busca real com o
    valor na escala treinada, 9,1% dos FPU saíam acima de 1 e chegavam a **+5,15**. Um
    filho não visitado que vale 5 ganha de todos os irmãos incondicionalmente — a busca
    passa a abrir filhos novos em vez de aprofundar, que é exatamente a patologia que a
    normalização existe para remover.
    """
    mm = MinMax()
    for q in (27.9, 28.0):          # uma faixa estreita, como no começo de uma árvore
        mm.atualiza(q)
    no = No()
    com_norma = MCTS(uniforme, fpu="pai", q_normalizado=True)
    for valor, esperado in ((28.5, 1.0), (20.0, 0.0), (27.95, 0.5), (27.9, 0.0)):
        no.visitas, no.soma_valor = 1, valor
        assert com_norma._q_virgem(no, mm) == pytest.approx(esperado, abs=1e-9), \
            f"FPU normalizado de V={valor} saiu de [0, 1]"

    # sem normalização o FPU continua sendo o valor cru do pai
    no.visitas, no.soma_valor = 1, 28.5
    assert MCTS(uniforme, fpu="pai")._q_virgem(no, None) == pytest.approx(28.5)
    # e `fpu="zero"` nunca é normalizado: `0` já é o piso da faixa (MuZero, Apêndice B)
    assert MCTS(uniforme, q_normalizado=True)._q_virgem(no, mm) == 0.0


def test_tie_breaking_is_biased_towards_the_first_masked_action_by_default():
    """O empate exato não é raro: na primeira descida de um nó nenhum filho foi visitado.

    Com prior uniforme todos têm o mesmo `Q` e o mesmo `u`, e `"ordem"` fica sempre com o
    primeiro do dicionário — que é a primeira ação liberada pela máscara, `np.nonzero`
    crescente, ou seja *virar à esquerda*. O viés é sistemático e sempre para o mesmo lado.
    """
    env = VecSnake(64, 10, rng=np.random.default_rng(3))
    obs, mask = env.reset()
    ordem = MCTS(uniforme, num_simulations=1, rng=np.random.default_rng(0))
    sorteio = MCTS(uniforme, num_simulations=1, desempate="aleatorio",
                   rng=np.random.default_rng(0))
    v_ordem, _ = ordem.run(env.get_state(), mask, obs)
    v_sorteio, _ = sorteio.run(env.get_state(), mask, obs)

    # com 1 simulação, a única visita extra vai para a ação escolhida no desempate
    primeira = mask.argmax(1)
    escolhida_ordem = v_ordem.argmax(1)
    assert (escolhida_ordem == primeira).mean() > 0.95, \
        "o desempate por ordem deixou de preferir a primeira ação da máscara"
    assert not np.array_equal(v_ordem, v_sorteio), \
        "o desempate aleatório não mudou nenhuma escolha"


def test_random_tie_breaking_still_never_picks_a_masked_action():
    env = VecSnake(32, 10, rng=np.random.default_rng(4))
    obs, mask = env.reset()
    m = MCTS(uniforme, num_simulations=16, desempate="aleatorio",
             rng=np.random.default_rng(1))
    for _ in range(10):
        visitas, _ = m.run(env.get_state(), mask, obs)
        assert (visitas[~mask] == 0).all()
        obs, mask, *_ = env.step(visitas.argmax(1).astype(np.int32))


def test_unknown_tie_breaking_is_rejected_at_construction():
    with pytest.raises(ValueError, match="desempate"):
        MCTS(uniforme, desempate="talvez")


# ------------------------------------------------------------------ AlphaZero
def cfg_min(**kw):
    base = dict(net="resnet_tiny", num_envs=8, rollout=3, num_simulations=6,
                batch_size=16, memory_size=2000, total_steps=1000,
                eval_every_steps=10**9, eval_episodes=40, eval_envs=20,
                log_every_steps=10**9, salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return AlphaZeroConfig(**base)


def test_collect_fills_the_buffer_with_search_targets():
    ag = AlphaZero(cfg_min())
    stats = ag.collect()
    n = ag.cfg.rollout * ag.cfg.num_envs
    assert ag._cheio == n
    assert np.allclose(ag._buf_pi[:n].sum(1), 1.0, atol=1e-5), \
        "o alvo de política tem que ser distribuição"
    assert np.isfinite(ag._buf_z[:n]).all()
    assert "valor_raiz" in stats


def test_policy_target_never_puts_mass_on_masked_actions():
    ag = AlphaZero(cfg_min())
    ag.collect()
    n = ag.cfg.rollout * ag.cfg.num_envs
    assert (ag._buf_pi[:n][~ag._buf_mask[:n]] == 0).all()


def test_official_policy_has_no_search():
    """A curva do contrato mede a rede pura — busca é coluna separada, como o flood-fill."""
    ag = AlphaZero(cfg_min())
    fn = ag.politica()
    obs, mask = ag.env.reset()
    a, b = fn(obs, mask), fn(obs, mask)
    assert np.array_equal(a, b)
    assert (a[~mask] == MASK_NEG).all()


def test_search_evaluation_follows_the_same_protocol():
    ag = AlphaZero(cfg_min())
    st = ag.avaliar_com_busca(episodes=32, num_simulations=4)
    assert st["episodes"] == 32
    assert st["num_simulations"] == 4
    assert 0.0 <= st["score_mean"] <= 97


def test_temperature_decays():
    ag = AlphaZero(cfg_min(total_steps=1000, temp_inicio=1.0, temp_fim=0.25,
                           temp_frac=0.5, temp_passos=0))
    assert ag.temperatura() == pytest.approx(1.0)
    ag.global_step = 500
    assert ag.temperatura() == pytest.approx(0.25)


def test_per_move_temperature_replaces_the_training_schedule():
    """`temp_passos` liga o agendamento do paper: τ por lance do EPISÓDIO, não do treino."""
    ag = AlphaZero(cfg_min(temp_passos=5, temp_inicio=1.0, temp_fim=0.05))
    ag.env.steps[:] = 0
    assert np.allclose(ag.temperatura(), 1.0)
    ag.env.steps[:] = 99
    assert np.allclose(ag.temperatura(), 0.05)
    ag.env.steps[: ag.cfg.num_envs // 2] = 0        # metade quente, metade fria
    t = ag.temperatura()
    assert t.shape == (ag.cfg.num_envs,) and set(np.unique(t)) == {1.0, 0.05}


def test_target_temperature_is_independent_of_the_acting_one():
    """Com `temp_alvo=1`, o alvo é a contagem de visitas crua, por mais fria que seja a
    política que escolheu a ação."""
    ag = AlphaZero(cfg_min(temp_inicio=0.05, temp_fim=0.05, temp_alvo=1.0))
    ag.collect()
    n = ag.cfg.rollout * ag.cfg.num_envs
    pi = ag._buf_pi[:n]
    assert np.allclose(pi.sum(1), 1.0, atol=1e-5)
    assert (pi[~ag._buf_mask[:n]] == 0).all()
    # τ=0,05 no alvo teria produzido quase-argmax; a distribuição crua não é isso
    assert pi.max(1).mean() < 0.99


def test_symlog_keeps_the_search_on_the_real_value_scale():
    """A busca soma `recompensa + γ·valor` com recompensas de verdade (+1 por maçã).

    Se `_frente` devolvesse o valor comprimido, a árvore compararia maçãs com logaritmos —
    e o `valor_symlog` viraria uma mudança silenciosa da dinâmica da busca em vez de uma
    mudança da representação que a rede aprende. Aqui a rede é a mesma nos dois agentes:
    o que se confere é que a leitura desfaz a transformação.
    """
    ag = AlphaZero(cfg_min(valor_symlog=True))
    cru = AlphaZero(cfg_min(valor_symlog=False))
    cru.model.set_weights(ag.model.get_weights())
    obs, mask = ag.env.reset()
    _, v_sym = ag._avaliar(obs, mask)
    _, v_cru = cru._avaliar(obs, mask)
    esperado = np.sign(v_cru) * np.expm1(np.abs(v_cru))
    assert np.allclose(v_sym, esperado, atol=1e-5)


def test_symlog_trains_the_head_against_the_compressed_target():
    """E o treino tem que ir para o outro lado: a cabeça aprende `symlog(z)`, não `z`."""
    ag = AlphaZero(cfg_min(valor_symlog=True, batch_size=16))
    ag.collect()
    assert ag._aprender() is not None      # não estoura com alvo comprimido
    grande = np.full(8, 50.0, np.float32)
    assert float(np.abs(ag._symlog(grande).numpy()).max()) < 5.0


def test_the_last_step_of_the_window_has_no_bootstrap_by_default():
    """O buraco: em `t = T-1` não há estado seguinte no buffer, e o alvo é a recompensa nua.

    Num jogo de recompensa esparsa isso é um zero em 1/16 das amostras — e um zero que
    ensina "aqui não há futuro", que é falso. Com `bootstrap_fim_janela` o valor da rede no
    estado em que a coleta parou fecha a janela.
    """
    sem = AlphaZero(cfg_min(rollout=4, n_step=2, bootstrap_fim_janela=False))
    sem.collect()
    N = sem.cfg.num_envs
    z_sem = sem._buf_z[: sem.cfg.rollout * N].reshape(sem.cfg.rollout, N)

    com = AlphaZero(cfg_min(rollout=4, n_step=2, bootstrap_fim_janela=True))
    com.model.set_weights(sem.model.get_weights())
    com.collect()
    z_com = com._buf_z[: com.cfg.rollout * N].reshape(com.cfg.rollout, N)

    assert np.isfinite(z_sem).all() and np.isfinite(z_com).all()
    # a última linha é a que muda de significado; com bootstrap ela deixa de ser só `r`
    assert not np.allclose(z_sem[-1], z_com[-1]), \
        "ligar o bootstrap não mudou o alvo do último passo da janela"


def test_symexp_is_clipped_so_a_diverged_head_cannot_poison_the_tree():
    """Uma cabeça que disparou tem que virar número grande e finito, não `2e17`.

    Sem o teto, `symexp` de um valor absurdo entra no backup do MCTS como `r + γ·v` e
    contamina a árvore inteira — e um treino de 8 horas não pode depender de a cabeça nunca
    passar do ponto.
    """
    teto = float(np.expm1(AlphaZero.LIMITE_SYMLOG))
    for x in (0.0, 1.0, 4.6, 40.0, -40.0, 1e6):
        v = float(AlphaZero._symexp(np.float32(x)).numpy())
        assert np.isfinite(v) and abs(v) <= teto + 1e-3, f"symexp({x}) = {v}"
    # e dentro da faixa útil continua sendo a inversa exata do symlog
    z = np.array([0.0, 0.5, 9.0, 97.0], np.float32)
    assert np.allclose(AlphaZero._symexp(AlphaZero._symlog(z)).numpy(), z, atol=1e-3)


def test_the_learning_rate_decays_only_when_asked():
    """`lr` constante era o padrão — e é o único agente do repo sem decaimento.

    Na execução de 5 M passos o score oscila entre 9,6 e 12,5 depois de 3 M sem tendência,
    e o `best` fica 2,4 pontos acima do `last`, que é o número oficial. Passo grande demais
    no fim de um treino tem exatamente esse desenho.
    """
    fixo = AlphaZero(cfg_min(total_steps=1000, lr_final=0.0))
    fixo.iterate()
    assert float(fixo.optimizer.learning_rate) == pytest.approx(fixo.cfg.lr)
    fixo.global_step = 1000
    fixo.iterate()
    assert float(fixo.optimizer.learning_rate) == pytest.approx(fixo.cfg.lr), \
        "o padrão passou a decair sozinho — a execução de controle deixa de ser controle"

    decai = AlphaZero(cfg_min(total_steps=1000, lr=3e-4, lr_final=5e-5))
    cedo = decai.iterate()
    # `iterate` já consumiu passos, então o lr do primeiro update é um pouco abaixo de `lr`
    assert 5e-5 < cedo["lr"] <= 3e-4
    decai.global_step = 1000
    tarde = decai.iterate()
    assert tarde["lr"] == pytest.approx(5e-5, rel=1e-3)
    assert float(decai.optimizer.learning_rate) == pytest.approx(5e-5, rel=1e-3)
    assert tarde["lr"] < cedo["lr"]


def test_the_gradient_budget_is_reported():
    """`atualizacoes` é o eixo do §2.1 e o registro da execução de 5 M nasceu sem ele."""
    ag = AlphaZero(cfg_min(epochs_por_iter=3))
    ag.iterate()
    assert ag.iterate()["atualizacoes"] == 3


def test_iterate_trains_and_reports():
    ag = AlphaZero(cfg_min(batch_size=16))
    antes = [w.numpy().copy() for w in ag.model.trainable_variables]
    ag.iterate()
    stats = ag.iterate()
    depois = [w.numpy() for w in ag.model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(antes, depois))
    assert np.isfinite(stats["perda_pi"]) and np.isfinite(stats["perda_v"])


def test_checkpoint_roundtrip(tmp_path):
    cfg = cfg_min(ckpt_dir=str(tmp_path))
    ag = AlphaZero(cfg); ag.iterate(); ag.salvar("last")
    outro = AlphaZero(cfg_min(ckpt_dir=str(tmp_path)))
    assert outro.retomar("last")
    outro.iterate()
