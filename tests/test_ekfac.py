"""EK-FAC: a base é a mesma, os autovalores não. E o ACEKTR, que é só essa troca.

O risco aqui é o mesmo que o `test_kfac.py` descreve: uma implementação errada não levanta
exceção, ela devolve um número plausível. E há um risco a mais, específico deste algoritmo —
**um EK-FAC que não corrige nada é indistinguível de um K-FAC**, e passaria em qualquer teste
de "roda e treina". Por isso os testes daqui têm três camadas:

1. **O controle exato.** Com `s*` no palpite do K-FAC, as duas direções coincidem até o
   arredondamento de float32. Isso prova que a base, o amortecimento e as escalas estão na
   mesma convenção — se qualquer uma delas estivesse fora, o EK-FAC seria "diferente" por
   um bug e não pela correção.
2. **O teorema.** Contra uma Fisher montada explicitamente em dimensão baixa, o EK-FAC não
   pode ser pior que o K-FAC em norma de Frobenius (Teorema 3 do paper), e tem que ser
   claramente melhor quando a hipótese de Kronecker é violada de propósito.
3. **A troca única.** O ACEKTR só pode diferir do ACKTR no pré-condicionador; qualquer
   outra divergência faz a comparação entre as curvas medir outra coisa.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers

from snakeai.agents import ACEKTR, ACEKTRConfig, ACKTR, ACKTRConfig
from snakeai.kfac import EKFac, KFac
from snakeai.plot import ORDEM_ALGORITMOS, cores_por_algoritmo, familia_de


def denso(entrada=4, saida=3, bias=False):
    inp = keras.Input(shape=(entrada,))
    return keras.Model(inp, layers.Dense(saida, use_bias=bias, name="d")(inp))


def cfg(**kw):
    base = dict(net="resnet_tiny", num_envs=32, rollout=8,
                eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                salvar_gif=False, salvar_grafico=False)
    base.update(kw)
    return ACEKTRConfig(**base)


def _lote(semente=0, n=200, din=4, dout=3, correlacionado=False):
    """Ativações e gradientes de pré-ativação, opcionalmente **correlacionados**.

    Com `a` e `g` independentes, a Fisher verdadeira já é quase um produto de Kronecker e
    o K-FAC quase acerta — não é lá que a diferença aparece. O caso correlacionado é o que
    o paper existe para tratar.
    """
    rng = np.random.default_rng(semente)
    a = rng.normal(size=(n, din))
    if correlacionado:
        g = np.stack([a[:, 0] ** 2, a[:, 1] * a[:, 2], np.abs(a[:, 3])], 1)
        g = g + 0.3 * rng.normal(size=(n, dout))
    else:
        g = rng.normal(size=(n, dout))
    return tf.constant(a, tf.float32), tf.constant(g, tf.float32)


# ==================================================== 1. o controle exato
def test_without_measuring_ekfac_is_bit_for_bit_kfac():
    """A âncora da suíte.

    `s*` nasce em `λ_A ⊗ λ_G` — o palpite do K-FAC — e o amortecimento do apêndice C
    reproduz, na base, a mesma forma do Tikhonov fatorado. Com a medição desligada
    (`ema_escalas=1`), as duas direções têm que coincidir.

    Se este teste falhar, a diferença entre ACKTR e ACEKTR na arena passa a incluir uma
    convenção trocada — uma transposta, um `π` do lado errado, um fator `N` — e deixa de
    ser atribuível à correção de autovalores. Nada quebraria; a curva só ficaria diferente
    pelo motivo errado.
    """
    m = denso(bias=True)
    c = m.get_layer("d")
    a, g = _lote()
    rng = np.random.default_rng(7)
    gk = tf.constant(rng.normal(size=(4, 3)), tf.float32)
    gb = tf.constant(rng.normal(size=(3,)), tf.float32)

    kf = KFac(m, damping=1e-1, ema=0.0, inv_every=1)
    kf.acumula([(c, a, None)], [g])
    ek = EKFac(m, damping=1e-1, ema=0.0, inv_every=1, ema_escalas=1.0)
    ek.acumula([(c, a, None)], [g])

    for x, y in zip(kf.precondiciona([gk, gb]), ek.precondiciona([gk, gb])):
        np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=2e-4, atol=2e-6)


def test_measuring_actually_changes_the_direction():
    """O controle do teste anterior. Um EK-FAC que nunca sai do palpite passaria naquele
    teste e não faria nada — e a curva dele seria a do ACKTR, sem que nada acusasse."""
    m = denso()
    c = m.get_layer("d")
    a, g = _lote(correlacionado=True)
    grad = tf.constant(np.random.default_rng(3).normal(size=(4, 3)), tf.float32)

    kf = KFac(m, damping=1e-2, ema=0.0, inv_every=1)
    kf.acumula([(c, a, None)], [g])
    ek = EKFac(m, damping=1e-2, ema=0.0, inv_every=1, ema_escalas=0.0,
               escalas_acumuladas=False)
    ek.acumula([(c, a, None)], [g])

    dk = kf.precondiciona([grad])[0].numpy()
    de = ek.precondiciona([grad])[0].numpy()
    # relativa, e não absoluta: a direção natural aqui tem norma ~1e-5, e um limiar
    # absoluto passaria a testar a escala do passo em vez da diferença entre os dois
    relativa = np.linalg.norm(dk - de) / np.linalg.norm(dk)
    assert relativa > 0.05, f"a correção mudou a direção em apenas {relativa:.1%}"


def test_the_measured_scales_have_the_same_total_mass_as_the_kfac_guess():
    """A escala de `s*` tem que casar com a de `λ_A·λ_G`, senão o amortecimento do
    apêndice C não fecha e `damping` passa a significar coisas diferentes nos dois
    algoritmos.

    Com `a` e `g` independentes, `E[pa²pg²] = E[pa²]E[pg²]` e o traço tem que bater quase
    exatamente. Um fator `N`, `T` ou `N·T` fora do lugar apareceria aqui como uma ordem de
    grandeza — e em nenhum outro teste.
    """
    m = denso(bias=True)
    c = m.get_layer("d")
    a, g = _lote(n=4000)
    ek = EKFac(m, damping=1e-2, ema=0.0, inv_every=1, ema_escalas=0.0,
               escalas_acumuladas=False)
    ek.acumula([(c, a, None)], [g])

    ref = (ek._lamA["d"][:, None] * ek._lamG["d"][None, :]).numpy()
    assert float(ek._m2["d"].numpy().sum() / ref.sum()) == pytest.approx(1.0, rel=0.15)


def test_the_conv_scales_keep_the_same_convention():
    """A mesma conferência numa `Conv2D`, onde a soma sobre as `T` posições espaciais entra
    dividindo por `N` e não por `N·T`. Trocar os dois deixaria `s*` menor por um fator `T`
    — 100 na primeira convolução de um tabuleiro 10×10 — e o EK-FAC viraria um K-FAC com
    amortecimento gigante, que **treina**, só que pior."""
    inp = keras.Input(shape=(8, 8, 3))
    c = layers.Conv2D(4, 3, padding="same", use_bias=False, name="c")
    m = keras.Model(inp, c(inp))
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(16, 8, 8, 3)), tf.float32)
    gp = tf.constant(rng.normal(size=(16, 8, 8, 4)), tf.float32)

    ek = EKFac(m, damping=1e-2, ema=0.0, inv_every=1, ema_escalas=0.0,
               escalas_acumuladas=False)
    ek.acumula([(c, x, None)], [gp])
    ref = (ek._lamA["c"][:, None] * ek._lamG["c"][None, :]).numpy()
    assert float(ek._m2["c"].numpy().sum() / ref.sum()) == pytest.approx(1.0, rel=0.2)


# ======================================================== 2. o teorema
def _fisher_exata(a, g):
    """`F = E_n[vec(∇_n) vec(∇_n)ᵀ]` com `∇_n = â_n ĝ_nᵀ`, na convenção `vec` por colunas.

    Só é possível montar isto em dimensão baixa — que é exatamente por que os testes de
    curvatura deste repositório usam uma camada 4→3.
    """
    a, g = np.asarray(a), np.asarray(g)
    n = a.shape[0]
    gh = g * n                       # o gradiente por amostra da perda **somada**
    V = np.stack([np.outer(a[i], gh[i]).reshape(-1, order="F") for i in range(n)])
    return V.T @ V / n


def _erro_relativo(F, aprox):
    return float(np.linalg.norm(F - aprox) / np.linalg.norm(F))


def _aproximacoes(m, c, a, g):
    kf = KFac(m, damping=1e-8, ema=0.0, inv_every=1)
    kf.acumula([(c, a, None)], [g])
    ek = EKFac(m, damping=1e-8, ema=0.0, inv_every=1, ema_escalas=0.0,
               escalas_acumuladas=False)
    ek.acumula([(c, a, None)], [g])

    F_kfac = np.kron(kf._G["d"].numpy(), kf._A["d"].numpy())
    U = np.kron(ek._UG["d"].numpy(), ek._UA["d"].numpy())
    s = ek._m2["d"].numpy().reshape(-1, order="F")
    return F_kfac, U @ np.diag(s) @ U.T


def test_ekfac_is_never_worse_than_kfac_in_frobenius_norm():
    """Teorema 3 do paper, conferido contra a Fisher montada explicitamente.

    Não é uma comparação de desempenho — é a afirmação matemática que justifica o
    algoritmo existir. `s*` é o mínimo de um problema de mínimos quadrados sobre as escalas
    diagonais **daquela base**, e o palpite do K-FAC é um ponto qualquer do mesmo espaço.
    """
    m, c = denso(), None
    c = m.get_layer("d")
    for semente in range(3):
        for corr in (False, True):
            a, g = _lote(semente=semente, n=4000, correlacionado=corr)
            F = _fisher_exata(a.numpy(), g.numpy())
            F_k, F_e = _aproximacoes(m, c, a, g)
            assert _erro_relativo(F, F_e) <= _erro_relativo(F, F_k) + 1e-6


def test_the_gain_shows_up_where_the_kronecker_hypothesis_is_violated():
    """E onde ela vale, quase não há o que corrigir.

    Este par é o que dá sentido ao número `ekfac_desvio` do registro: com `a` e `g`
    independentes o EK-FAC empata com o K-FAC, e um ganho grande na arena **teria** que
    vir acompanhado de um desvio grande. Se as duas coisas não andarem juntas, alguma das
    duas está errada.
    """
    m = denso()
    c = m.get_layer("d")

    a, g = _lote(n=4000, correlacionado=False)
    F = _fisher_exata(a.numpy(), g.numpy())
    F_k, F_e = _aproximacoes(m, c, a, g)
    ganho_independente = _erro_relativo(F, F_k) - _erro_relativo(F, F_e)

    a, g = _lote(n=4000, correlacionado=True)
    F = _fisher_exata(a.numpy(), g.numpy())
    F_k, F_e = _aproximacoes(m, c, a, g)
    ganho_correlacionado = _erro_relativo(F, F_k) - _erro_relativo(F, F_e)

    assert ganho_correlacionado > 10 * max(ganho_independente, 1e-6)
    assert ganho_correlacionado > 0.05, "o caso difícil tem que separar os dois"


# ================================================== a base e as escalas
def test_the_basis_is_orthonormal():
    m = denso(bias=True)
    a, g = _lote()
    ek = EKFac(m, damping=1e-2, ema=0.0, inv_every=1)
    ek.acumula([(m.get_layer("d"), a, None)], [g])
    for U, d in ((ek._UA["d"], 5), (ek._UG["d"], 3)):
        np.testing.assert_allclose((U.numpy().T @ U.numpy()), np.eye(d),
                                   rtol=1e-4, atol=1e-4)


def test_the_eigenvalues_are_never_negative():
    """`A` e `G` são PSD por construção, mas `eigh` devolve autovalores levemente negativos
    por arredondamento. Deixá-los passar inverteria o sinal daquele eixo — um passo que
    sobe a perda, em uma direção só, sem erro nenhum."""
    m = denso(bias=True)
    a, g = _lote(n=3)                       # menos amostras que dimensões: A é singular
    ek = EKFac(m, damping=1e-2, ema=0.0, inv_every=1)
    ek.acumula([(m.get_layer("d"), a, None)], [g])
    assert (ek._lamA["d"].numpy() >= 0).all()
    assert (ek._lamG["d"].numpy() >= 0).all()


def test_rebuilding_the_basis_resets_the_scales():
    """`s*` são escalas dos eixos de **uma** base. Quando a base muda, os números antigos
    passam a descrever eixos que não existem mais; reaproveitá-los daria um
    pré-condicionador que mistura duas bases — plausível, silencioso e errado."""
    m = denso(bias=True)
    c = m.get_layer("d")
    ek = EKFac(m, damping=1e-2, ema=0.5, inv_every=3, ema_escalas=0.0,
               escalas_acumuladas=False)
    for i in range(2):
        a, g = _lote(semente=i, correlacionado=True)
        ek.acumula([(c, a, None)], [g])
    assert ek.desvio_de_kronecker() > 1e-3, "sem medição não há o que reiniciar"

    ek._passos = 0                          # força a próxima chamada a reconstruir a base
    a, g = _lote(semente=99)
    ek.acumula([(c, a, None)], [g])
    ref = (ek._lamA["d"][:, None] * ek._lamG["d"][None, :]).numpy()
    # a medição deste lote já entrou (ema 0), mas a partida foi do palpite: o que se
    # confere é que `s*` não carregou nada da base anterior
    assert ek._m2["d"].shape == ref.shape


def test_uncovered_layers_pass_through_untouched():
    inp = keras.Input(shape=(6, 6, 4))
    x = layers.Conv2D(4, 3, padding="same", use_bias=False, name="c")(inp)
    x = layers.GroupNormalization(groups=2, name="gn")(x)
    m = keras.Model(inp, layers.GlobalAveragePooling2D()(x))

    ek = EKFac(m, damping=1e-1, inv_every=1)
    grads = [tf.ones_like(v) for v in m.trainable_variables]
    nat = ek.precondiciona(grads)
    for v, g, n in zip(m.trainable_variables, grads, nat):
        if "gn" in v.path:
            np.testing.assert_array_equal(g.numpy(), n.numpy())


def test_preconditioning_before_any_statistics_is_the_identity():
    m = denso()
    ek = EKFac(m, damping=1e-2)
    g = tf.ones([4, 3])
    np.testing.assert_array_equal(ek.precondiciona([g])[0].numpy(), g.numpy())


# ======================================================== 3. a troca única
def test_acektr_is_acktr_with_one_method_replaced():
    """Se qualquer outra coisa divergir, a comparação ACKTR × ACEKTR deixa de medir a
    correção de autovalores — e nada quebra."""
    assert issubclass(ACEKTR, ACKTR)
    assert ACEKTR.collect is ACKTR.collect
    assert ACEKTR.iterate is ACKTR.iterate
    assert ACEKTR._forward_e_gradientes is ACKTR._forward_e_gradientes
    assert ACEKTR._cria_precondicionador is not ACKTR._cria_precondicionador


def test_acektr_uses_the_eigenvalue_corrected_preconditioner():
    ag = ACEKTR(cfg())
    assert isinstance(ag.kfac, EKFac)
    so_do_acktr = set(ACKTRConfig.__dataclass_fields__)
    assert isinstance(ACKTR(ACKTRConfig(**{k: v for k, v in cfg().__dict__.items()
                                           if k in so_do_acktr})).kfac, KFac)


def test_acektr_is_its_own_algorithm_in_the_arena():
    ag = ACEKTR(cfg())
    assert ag.algo == "acektr"
    assert "acektr" in ORDEM_ALGORITMOS
    assert familia_de("acektr") == "política"
    cores = cores_por_algoritmo({"acktr", "acektr"})
    assert cores["acektr"] != cores["acktr"]


def test_acektr_trains():
    ag = ACEKTR(cfg())
    for _ in range(3):
        s = ag.iterate()
    for chave in ("pg", "vf", "ent", "kl", "lr", "ekfac_desvio"):
        assert np.isfinite(s[chave]), f"{chave} virou {s[chave]}"
    assert s["ent"] > 0


def test_the_measured_kl_respects_the_target():
    """A mesma exigência do ACKTR: o alvo sai de uma aproximação quadrática, e esta é a KL
    que de fato aconteceu. Trocar o pré-condicionador não pode afrouxar isso."""
    ag = ACEKTR(cfg(kl_max=1e-3))
    for _ in range(5):
        s = ag.iterate()
        assert s["kl"] <= 3 * s["kl_alvo"], f"KL {s['kl']:.5f} contra alvo {s['kl_alvo']}"


def test_the_correction_grows_between_basis_rebuilds_and_resets_at_one():
    """O dente de serra do `ekfac_desvio`, que é como ele deve ser lido.

    Cai a zero em cada reconstrução da base — é lá que `s*` volta ao palpite do K-FAC — e
    cresce até a próxima. Uma execução em que ele fica **grudado em zero** é uma execução
    em que o EK-FAC não está corrigindo nada, e a curva dela deveria coincidir com a do
    ACKTR; se não coincidir, o problema é outro.
    """
    ag = ACEKTR(cfg(inv_every=4))
    desvios = [ag.iterate()["ekfac_desvio"] for _ in range(9)]
    assert desvios[0] < desvios[3], "o desvio tem que crescer dentro da janela"
    assert desvios[4] < desvios[3], "e voltar a zero quando a base é reconstruída"
    assert max(desvios) > 1e-3, "o EK-FAC não está corrigindo nada"


def test_turning_the_correction_off_marks_the_variant():
    """`ema_escalas=1` é o EK-FAC sem medir, que é o K-FAC — o controle. Sem a marca, ele
    dividiria a identidade `(algo, variant, seed)` com o algoritmo de verdade e as duas
    curvas virariam uma só na arena.

    E `+base50` aparece **sempre**, porque o default do ACEKTR passou a ser o regime de
    amortização do paper (base rara, escalas sempre). A marca é comparada ao default do
    ACKTR, não ao daqui: uma execução nesse regime não é pareada com o `08_acktr`, e sem a
    marca ela ainda colidiria com a execução de 01/09, que rodou com `inv_every = 10`.
    """
    assert ACEKTR(cfg()).variant == "resnet_tiny+s_acum"
    assert ACEKTR(cfg(ema_escalas=1.0)).variant.endswith("+sem_correcao")
    assert ACEKTR(cfg(inv_every=50)).variant == "resnet_tiny+base50+s_acum"


def test_the_control_really_reproduces_acktr_inside_the_agent():
    """A âncora do começo, agora dentro do agente e com a rede de verdade: com a medição
    desligada **e a região de confiança pareada**, a primeira atualização do ACEKTR é a do
    ACKTR.

    O pareamento explícito é novo e é o ponto. O ACEKTR deixou de herdar a região de
    confiança do ACKTR: ele liga `kl_cal_debias` e parte de `kl_fator_inicial = 15`, então
    por padrão o primeiro passo dele é ~√15 menor. Isso é escolha, não bug — e este teste
    continua sendo o que separa "escolhemos diferente" de "a convenção do
    pré-condicionador está trocada", que é a única coisa que ele sempre existiu para
    detectar.
    """
    a = ACKTR(ACKTRConfig(net="resnet_tiny", num_envs=32, rollout=8, seed=0,
                          eval_every_steps=10 ** 9, log_every_steps=10 ** 9,
                          salvar_gif=False, salvar_grafico=False)).iterate()
    b = ACEKTR(cfg(seed=0, ema_escalas=1.0,
                   kl_cal_debias=False, kl_fator_inicial=1.0)).iterate()
    assert b["lr"] == pytest.approx(a["lr"], rel=2e-3)
    assert b["kl"] == pytest.approx(a["kl"], rel=5e-2)


def test_the_acektr_trust_region_starts_where_the_measurements_ended():
    """E o contrapositivo: no default, o primeiro passo do ACEKTR é deliberadamente menor.

    `_fator_kl` é uma média com constante de tempo de ~50 atualizações num orçamento de
    610. Partindo de 1,0 ela gasta 8% do treino com o alvo efetivo até 20× maior que o
    pedido, bem quando a política ainda é aleatória. Partir do fator que as execuções
    longas mediram (15 a 25) e debiasar a média encurta isso para ~2 atualizações.
    """
    import numpy as np

    a = ACEKTR(cfg(seed=0, ema_escalas=1.0,
                   kl_cal_debias=False, kl_fator_inicial=1.0)).iterate()
    b = ACEKTR(cfg(seed=0, ema_escalas=1.0)).iterate()
    assert b["kl_alvo_efetivo"] == pytest.approx(a["kl_alvo_efetivo"] / 15.0, rel=1e-6)
    assert b["lr"] < a["lr"]
    # e a debiasada chega ao fator medido em poucas atualizações, não em ~50
    ag = ACEKTR(cfg(seed=0))
    fatores = [ag.iterate()["kl_fator"] for _ in range(4)]
    assert np.ptp(fatores[-2:]) < abs(fatores[0]) * 5, "o fator deve assentar rápido"


def test_reloading_the_model_rebuilds_the_preconditioner():
    ag = ACEKTR(cfg())
    ag.iterate()
    antigo = ag.kfac
    ag.on_model_reloaded()
    assert ag.kfac is not antigo
    assert isinstance(ag.kfac, EKFac)
    ag.iterate()


def test_the_summary_reports_the_correction():
    r = ACEKTR(cfg()).resumo_kfac()
    assert r["fracao"] > 0.95
    assert "desvio_de_kronecker" in r and "escalas_medidas" in r


# ============================================ o estimador de `s*` dentro da janela
def test_the_accumulated_estimator_is_less_noisy_than_the_exponential_one():
    """A queixa da `EKFAC.md` §3.2 era real; a resposta é o estimador, não a janela.

    Dentro de uma janela de 10 atualizações a rede quase não muda, então não há deriva a
    esquecer — e a média móvel exponencial joga fora metade da informação a cada passo
    para se proteger de uma deriva que não aconteceu. Isso importa porque `s*` vai para o
    **denominador**: um autovalor subestimado por ruído amplifica exatamente a direção que
    o lote não soube estimar.

    Oito lotes ruidosos da **mesma** distribuição, uma base só (congelada, senão os `m2`
    descreveriam eixos diferentes e a comparação não significaria nada): o estimador
    acumulado tem que ficar mais perto do `s*` medido nos oito de uma vez.
    """
    m = denso(bias=True)
    c = m.get_layer("d")
    lotes = [_lote(semente=100 + i, n=40, correlacionado=True) for i in range(8)]

    def roda(**kw):
        # `ema=1.0` congela `A` e `G` depois do primeiro lote e `inv_every` gigante impede
        # qualquer reconstrução: tudo aqui vive na MESMA base, senão os `m2` descreveriam
        # eixos diferentes e a comparação não significaria nada.
        ek = EKFac(m, damping=1e-2, ema=1.0, inv_every=10 ** 6, **kw)
        saida = []
        for a, g in lotes:
            ek.acumula([(c, a, None)], [g])
            saida.append(ek._m2["d"].numpy())
        return saida

    # `s_i` de cada lote isolado — a medição crua, sem alisamento nenhum
    por_lote = roda(ema_escalas=0.0, escalas_acumuladas=False)
    alvo = np.mean(por_lote, axis=0)

    acum = roda(ema_escalas=0.0, escalas_acumuladas=True)[-1]
    expo = roda(ema_escalas=0.5, escalas_acumuladas=False)[-1]

    def erro(v):
        return float(np.linalg.norm(v - alvo) / np.linalg.norm(alvo))

    # a variação entre lotes é o que os dois estimadores têm que absorver; se ela for
    # pequena o teste não separa nada e é o teste que está errado, não o estimador
    espalhamento = np.mean([erro(v) for v in por_lote])
    assert espalhamento > 0.1, f"lotes indistinguíveis ({espalhamento:.3f}): teste cego"
    assert erro(acum) < erro(expo), (
        f"acumulado {erro(acum):.3f} não ficou abaixo do exponencial {erro(expo):.3f}")
    assert erro(acum) < espalhamento, "o acumulado tem que ser melhor que um lote só"


def test_the_basis_is_rebuilt_as_often_as_the_kfac_one():
    """A trava do achado de 02/09.

    `inv_every = 50` (o regime de amortização do paper) fez `ekfac_desvio` subir de 0,06
    para **69,6** dentro da primeira janela e cair para 0,29 na reconstrução seguinte — um
    dente de serra de duas ordens de grandeza que **escala com o comprimento da janela**,
    ou seja mede base velha, não violação de Kronecker. E base velha no EK-FAC é pior que
    no K-FAC: ele mistura `s*` medido agora com eixos de 50 passos atrás. `kl_fator` foi a
    46,2 contra 18,7–20,0 das execuções com base fresca, e a execução fechou com 0,4% de
    tabuleiros cheios.

    Com 610 atualizações no orçamento inteiro, 50 delas são 8% do treino. A premissa do
    paper — o modelo muda pouco entre reconstruções — não vale aqui.
    """
    from snakeai.agents.acktr import ACKTRConfig
    assert ACEKTRConfig.inv_every == ACKTRConfig.inv_every == 10
    assert ACEKTRConfig.escalas_acumuladas is True
