"""K-FAC: a álgebra de Kronecker, as escalas, e se de fato pré-condiciona.

O risco de uma implementação de K-FAC não é dar erro — é dar um número plausível com uma
transposta trocada ou um fator `N` no lugar errado. Nesse caso o treino não quebra, só fica
um pouco pior que o gradiente comum, e ninguém descobre. Por isso os testes daqui comparam
contra a matriz de Fisher **construída explicitamente**, que é possível em dimensão baixa.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers

from snakeai.kfac import (KFac, captura_kfac, fatores_de_camada, patches_de_entrada,
                          perda_fisher_categorica, perda_fisher_gaussiana)


def denso(entrada=4, saida=3, bias=False, ativacao=None):
    inp = keras.Input(shape=(entrada,))
    out = layers.Dense(saida, use_bias=bias, activation=ativacao, name="d")(inp)
    return keras.Model(inp, out)


# ------------------------------------------------------------------- a captura
def test_captured_forward_is_bit_identical():
    """A captura reimplementa `call`; se a reimplementação divergir, tudo depois é lixo."""
    m = denso(bias=True, ativacao="relu")
    x = tf.constant(np.random.randn(9, 4), tf.float32)
    esperado = m(x).numpy()

    with captura_kfac([m.get_layer("d")]) as cap:
        obtido = m(x).numpy()

    np.testing.assert_array_equal(esperado, obtido)
    assert len(cap) == 1
    _, entrada, pre = cap[0]
    # a pré-ativação é anterior ao ReLU: tem que ter negativos que a saída não tem
    assert (pre.numpy() < 0).any()
    np.testing.assert_allclose(np.maximum(pre.numpy(), 0), obtido, rtol=1e-6)


def test_capture_is_undone_even_on_error():
    """E sem deixar um `call` de instância sombreando o da classe."""
    m = denso()
    c = m.get_layer("d")
    assert "call" not in c.__dict__
    with pytest.raises(RuntimeError):
        with captura_kfac([c]):
            assert "call" in c.__dict__
            raise RuntimeError("boom")
    assert "call" not in c.__dict__


def test_preactivation_gradient_flows_to_the_tape():
    m = denso(bias=True)
    x = tf.constant(np.random.randn(6, 4), tf.float32)
    with captura_kfac([m.get_layer("d")]) as cap:
        with tf.GradientTape() as tape:
            y = m(x, training=True)
            perda = tf.reduce_sum(y ** 2)
        g = tape.gradient(perda, [z for _, _, z in cap])
    assert g[0] is not None and g[0].shape == (6, 3)


# --------------------------------------------------------------------- patches
def test_conv_patches_reproduce_the_convolution():
    """A ordem do achatamento tem que casar com `kernel.reshape(-1, cout)`.

    Se `extract_patches` devolvesse `(cin, kh, kw)` em vez de `(kh, kw, cin)`, `A` estaria
    permutada e o pré-condicionamento embaralharia canais com posições — sem erro nenhum.
    """
    inp = keras.Input(shape=(7, 7, 3))
    c = layers.Conv2D(5, 3, padding="same", use_bias=False, name="c")
    m = keras.Model(inp, c(inp))
    x = tf.constant(np.random.randn(2, 7, 7, 3), tf.float32)

    esperado = m(x).numpy().reshape(-1, 5)
    p = patches_de_entrada(c, x)
    obtido = (p @ tf.reshape(c.kernel, [-1, 5])).numpy()
    np.testing.assert_allclose(esperado, obtido, rtol=1e-4, atol=1e-4)


def test_patch_count_is_batch_times_output_positions():
    inp = keras.Input(shape=(10, 10, 4))
    c = layers.Conv2D(8, 3, strides=2, padding="valid", use_bias=False)
    m = keras.Model(inp, c(inp))
    x = tf.zeros([3, 10, 10, 4])
    assert patches_de_entrada(c, x).shape == (3 * 4 * 4, 3 * 3 * 4)


# ---------------------------------------------------------------------- escalas
def test_factor_scales_match_the_documented_formulas():
    """`A = (1/N) âᵀâ` e `G = N·gᵀg`. Um `N` fora do lugar só muda o passo efetivo — e
    então o K-FAC vira um gradiente comum com learning rate errado, e ninguém nota."""
    m = denso(entrada=4, saida=3, bias=True)
    c = m.get_layer("d")
    a = tf.constant(np.random.randn(11, 4), tf.float32)
    g = tf.constant(np.random.randn(11, 3), tf.float32)

    A, G, n, t = fatores_de_camada(c, a, g)
    assert float(n) == 11.0 and float(t) == 1.0

    ah = np.concatenate([a.numpy(), np.ones((11, 1))], axis=1)
    np.testing.assert_allclose(A.numpy(), ah.T @ ah / 11, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(G.numpy(), 11 * (g.numpy().T @ g.numpy()),
                               rtol=1e-4, atol=1e-4)


# ------------------------------------------------------ o teste que vale por todos
def test_preconditioned_direction_solves_the_kronecker_system():
    """`Δ` tem que satisfazer `F Δ = ∇` com `F = A ⊗ G` amortecida, montada à mão.

    Este é o teste que pega transposta trocada. `A⁻¹∇WG⁻¹` e `G⁻¹∇WA⁻¹` têm formas
    incompatíveis só quando as dimensões diferem — por isso a camada aqui é 4→3, e não
    quadrada. Com dimensões iguais, o erro passaria despercebido.
    """
    m = denso(entrada=4, saida=3, bias=False)
    c = m.get_layer("d")
    kf = KFac(m, damping=1e-1, ema=0.0, inv_every=1)

    rng = np.random.default_rng(0)
    a = tf.constant(rng.normal(size=(20, 4)), tf.float32)
    g = tf.constant(rng.normal(size=(20, 3)), tf.float32)
    kf.acumula([(c, a, None)], [g])

    grad = tf.constant(rng.normal(size=(4, 3)), tf.float32)
    delta = kf.precondiciona([grad])[0].numpy()

    # F montada explicitamente, com o mesmo amortecimento fatorado
    A = kf._A["d"].numpy()
    G = kf._G["d"].numpy()
    pi = np.sqrt((np.trace(A) / 4) / (np.trace(G) / 3))
    raiz = np.sqrt(kf.damping)
    Ad = A + np.eye(4) * (raiz * pi)
    Gd = G + np.eye(3) * (raiz / pi)

    # vec por colunas: vec(A X G) = (Gᵀ ⊗ A) vec(X)
    F = np.kron(Gd, Ad)
    vec = delta.reshape(-1, order="F")
    np.testing.assert_allclose((F @ vec).reshape(4, 3, order="F"), grad.numpy(),
                               rtol=1e-3, atol=1e-3)


def test_bias_is_preconditioned_as_the_extra_input_row():
    """O viés entra como uma coluna de 1 na entrada — e sai como a última linha de `Δ`."""
    m = denso(entrada=4, saida=3, bias=True)
    c = m.get_layer("d")
    kf = KFac(m, damping=1e-1, ema=0.0, inv_every=1)
    rng = np.random.default_rng(1)
    kf.acumula([(c, tf.constant(rng.normal(size=(30, 4)), tf.float32), None)],
               [tf.constant(rng.normal(size=(30, 3)), tf.float32)])

    gk = tf.constant(rng.normal(size=(4, 3)), tf.float32)
    gb = tf.constant(rng.normal(size=(3,)), tf.float32)
    nat = kf.precondiciona([gk, gb])

    assert nat[0].shape == (4, 3) and nat[1].shape == (3,)
    assert kf._A["d"].shape == (5, 5), "A tem que ganhar a linha/coluna do viés"


def test_uncovered_layers_pass_through_untouched():
    """Uma `GroupNormalization` no meio não pode ser silenciosamente zerada."""
    inp = keras.Input(shape=(6, 6, 4))
    x = layers.Conv2D(4, 3, padding="same", use_bias=False, name="c")(inp)
    x = layers.GroupNormalization(groups=2, name="gn")(x)
    m = keras.Model(inp, layers.GlobalAveragePooling2D()(x))

    kf = KFac(m, damping=1e-1, inv_every=1)
    assert [c.name for c in kf.camadas] == ["c"]

    grads = [tf.ones_like(v) for v in m.trainable_variables]
    nat = kf.precondiciona(grads)
    for v, g, n in zip(m.trainable_variables, grads, nat):
        if "gn" in v.path:
            np.testing.assert_array_equal(g.numpy(), n.numpy())


def test_resumo_reports_real_coverage():
    from snakeai.nets import build_actor_critic

    m = build_actor_critic(net="resnet_tiny")
    r = KFac(m).resumo()
    assert r["params_total"] > 0
    assert r["fracao"] > 0.9, f"cobertura baixa demais: {r['fracao']:.2%}"
    assert r["params_cobertos"] < r["params_total"], "o GroupNorm não é coberto"


# --------------------------------------------------------------- região de confiança
def test_kl_scale_shrinks_when_the_step_is_curvature_expensive():
    nat = [tf.ones([4, 3])]
    peq = KFac.escala_kl(nat, [tf.ones([4, 3]) * 0.01], kl_max=1e-3, lr_max=1.0)
    gra = KFac.escala_kl(nat, [tf.ones([4, 3]) * 10.0], kl_max=1e-3, lr_max=1.0)
    assert float(gra) < float(peq)


def test_kl_scale_is_capped_by_the_learning_rate():
    """Sem o teto, um lote com curvatura quase nula pediria um passo gigante."""
    nat = [tf.ones([4, 3]) * 1e-8]
    e = KFac.escala_kl(nat, [tf.ones([4, 3]) * 1e-8], kl_max=1e-2, lr_max=0.25)
    assert float(e) == pytest.approx(0.25)


def test_kl_scale_matches_the_closed_form():
    nat = [tf.constant([[2.0, 0.0]])]
    cru = [tf.constant([[0.5, 0.0]])]
    # Δᵀ∇ = 1.0 → η = sqrt(2·0.02/1.0)
    e = KFac.escala_kl(nat, cru, kl_max=0.02, lr_max=10.0)
    assert float(e) == pytest.approx(np.sqrt(0.04), rel=1e-5)


# ------------------------------------------------------------------ comportamento
def _passo(m, kf, X, Y, lr, fisher=True, g_exata=None):
    """Um passo: perda real para o gradiente, perda de Fisher para as estatísticas."""
    with captura_kfac(kf.camadas if kf else []) as cap:
        with tf.GradientTape(persistent=True) as tape:
            z = m(X, training=True)
            # soma sobre as saídas, média sobre o lote — a mesma normalização da
            # log-verossimilhança gaussiana, senão a direção natural sai escalada por `d`
            perda = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z - Y), axis=-1))
            pf = perda_fisher_gaussiana(z) if fisher else perda
        grads = tape.gradient(perda, m.trainable_variables)
        gs = tape.gradient(pf, [t for _, _, t in cap]) if kf else None
    if kf:
        kf.acumula(cap, gs)
        if g_exata is not None:
            for nome in kf._G:
                kf._G[nome] = g_exata
            kf.atualiza_inversos()
        grads = kf.precondiciona(grads)
    for v, g in zip(m.trainable_variables, grads):
        v.assign_sub(lr * g)
    return float(0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(m(X) - Y), axis=-1)))


def _problema(escala=(1.0, 30.0, 0.05, 8.0), n=8192):
    """Mínimos quadrados lineares mal-condicionado, com o ótimo conhecido em forma fechada."""
    rng = np.random.default_rng(7)
    X = (rng.normal(size=(n, 4)) * np.array(escala)).astype("float32")
    Y = (X @ rng.normal(size=(4, 3)) + rng.normal(size=(n, 3)) * 0.1).astype("float32")
    res = X @ np.linalg.lstsq(X, Y, rcond=None)[0] - Y
    return X, Y, float(0.5 * np.mean(np.sum(res ** 2, axis=-1)))


def test_one_kfac_step_with_the_exact_fisher_lands_on_the_optimum():
    """A prova de que a direção é Newton, e não apenas "melhor".

    Em mínimos quadrados lineares a Fisher gaussiana **é** a Gauss-Newton, que **é** a
    Hessiana. Então um único passo de tamanho 1 tem que aterrissar no ótimo de mínimos
    quadrados — não chegar perto: aterrissar. Nenhum método de primeira ordem faz isso,
    com learning rate nenhum (o teste seguinte mostra).

    Aqui `G` é fixada no seu valor analítico (`I`, porque o alvo gaussiano tem variância
    unitária) para isolar a álgebra do erro de amostragem, que o teste depois mede.
    """
    X, Y, otimo = _problema()
    keras.utils.set_random_seed(0)
    m = denso(bias=False)
    kf = KFac(m, damping=1e-12, ema=0.0, inv_every=10 ** 9)
    depois = _passo(m, kf, X, Y, lr=1.0, g_exata=tf.eye(3))
    assert depois == pytest.approx(otimo, rel=1e-4), f"ótimo {otimo:.5f}, obtido {depois:.5f}"


def test_gradient_descent_cannot_do_that_with_any_learning_rate():
    """O controle. Sem pré-condicionamento, um passo só não chega nem perto."""
    X, Y, otimo = _problema()
    melhor = np.inf
    for lr in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        keras.utils.set_random_seed(0)
        v = _passo(denso(bias=False), None, X, Y, lr=lr)
        if np.isfinite(v):
            melhor = min(melhor, v)
    assert melhor > 50 * otimo


def test_the_sampled_fisher_costs_precision_and_the_cost_is_sampling_noise():
    """Com `G` estimada por amostragem, o passo erra — e o erro é ruído, não viés.

    `G` sai de `N` amostras gaussianas, então tem ruído de ordem `1/√N` (~1% com N=8192).
    Num problema com número de condição alto esse 1% aparece amplificado no resultado. É
    por isso que o K-FAC de verdade usa média móvel entre passos e amortecimento: ambos
    existem tanto para regularizar quanto para absorver este ruído.
    """
    X, Y, otimo = _problema()
    keras.utils.set_random_seed(0)
    m = denso(bias=False)
    kf = KFac(m, damping=1e-12, ema=0.0, inv_every=1)
    amostrada = _passo(m, kf, X, Y, lr=1.0)

    assert amostrada > 2 * otimo, "sem ruído nenhum, este teste não estaria medindo nada"
    assert amostrada < 20 * otimo, "ruído demais: G não está convergindo para I"

    # e o ruído encolhe com o lote, que é a assinatura de erro de amostragem
    Xg, Yg, otimo_g = _problema(n=8192 * 8)
    keras.utils.set_random_seed(0)
    m2 = denso(bias=False)
    kf2 = KFac(m2, damping=1e-12, ema=0.0, inv_every=1)
    maior = _passo(m2, kf2, Xg, Yg, lr=1.0)
    assert (maior / otimo_g) < (amostrada / otimo)


def test_the_empirical_fisher_is_a_worse_preconditioner():
    """Por que `perda_fisher_*` existe: usar os gradientes da perda real estraga a direção.

    Com o Fisher empírico, `G` vira a covariância dos **resíduos** em vez de `I`, e a
    direção deixa de ser Newton.
    """
    X, Y, otimo = _problema()
    keras.utils.set_random_seed(0)
    m = denso(bias=False)
    kf = KFac(m, damping=1e-12, ema=0.0, inv_every=1)
    assert _passo(m, kf, X, Y, lr=1.0, fisher=False) > 20 * otimo


def test_categorical_fisher_uses_a_sampled_action_not_the_taken_one():
    """A diferença entre Fisher e Fisher empírico, isolada num teste.

    Com uma política uniforme, `G` amostrado da política é aproximadamente `diag(p) - ppᵀ`.
    Se a ação viesse dos dados (aqui, sempre a ação 0), `G` teria uma cara completamente
    diferente — e é esse `G` que degenera perto do ótimo.
    """
    n = 20000
    logits = tf.zeros([n, 3])
    with tf.GradientTape() as tape:
        tape.watch(logits)
        perda = perda_fisher_categorica(logits, seed=3)
    g = tape.gradient(perda, logits)
    G = n * tf.matmul(g, g, transpose_a=True)

    p = np.full(3, 1 / 3)
    esperado = np.diag(p) - np.outer(p, p)
    np.testing.assert_allclose(G.numpy(), esperado, atol=0.02)

    with tf.GradientTape() as tape:
        tape.watch(logits)
        fixa = tf.reduce_mean(tf.nn.log_softmax(logits)[:, 0])
    ge = tape.gradient(fixa, logits)
    Ge = n * tf.matmul(ge, ge, transpose_a=True)
    assert np.abs(Ge.numpy() - esperado).max() > 0.1, "Fisher empírico tem que diferir"


def test_masked_actions_never_enter_the_fisher():
    """Uma ação mascarada tem probabilidade zero; amostrá-la envenenaria `G`."""
    logits = tf.zeros([5000, 3])
    mask = tf.constant([[True, False, True]] * 5000)
    with tf.GradientTape() as tape:
        tape.watch(logits)
        perda = perda_fisher_categorica(logits, mask=mask, seed=11)
    g = tape.gradient(perda, logits).numpy()
    # a coluna mascarada recebe gradiente ~0: nunca é escolhida e sua prob é ~0
    assert np.abs(g[:, 1]).max() < 1e-6
