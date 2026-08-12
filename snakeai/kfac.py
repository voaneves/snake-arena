"""K-FAC — curvatura aproximada por fatores de Kronecker, em Keras 3.

O que é
-------
Descida de gradiente natural precisa de `F⁻¹∇`, onde `F` é a matriz de Fisher. Para uma
rede com 300 mil parâmetros, `F` tem 9×10¹⁰ entradas: não cabe, muito menos inverte. O
K-FAC (Martens & Grosse, 2015) aproxima `F` por **blocos, um por camada**, e aproxima cada
bloco por um **produto de Kronecker de duas matrizes pequenas**::

    F_ℓ  ≈  A_ℓ ⊗ G_ℓ

* `A_ℓ` = covariância das **ativações que entram** na camada — lado `(entrada × entrada)`;
* `G_ℓ` = covariância dos **gradientes na pré-ativação** que sai — lado `(saída × saída)`.

A conta que torna isso viável é a identidade `(A ⊗ G)⁻¹ vec(∇W) = vec(A⁻¹ ∇W G⁻¹)`: em vez
de inverter uma matriz de `(in·out)²`, inverte-se uma de `in²` e uma de `out²`. Numa camada
de 288×64, isso é 340 mil entradas em vez de 340 **bilhões**.

Por que ele não entrou no eixo `optimizer`
------------------------------------------
Um `keras.optimizers.Optimizer` recebe apenas pares `(gradiente, variável)`. O K-FAC precisa
das **ativações de entrada** e dos **gradientes de pré-ativação** de cada camada — coisas
que só existem durante o passo forward/backward e que nenhum otimizador do Keras enxerga.
A API Keras do `tensorflow/kfac` contornava isso recebendo `model=` e `loss=` e refazendo o
forward por dentro; aquele repositório foi arquivado em 19/04/2026 e depende de
`tensorflow.compat.v1`, então não é um caminho.

Aqui o K-FAC é um **pré-condicionador**, não um otimizador: ele se coloca entre o gradiente
e o `optimizer.apply_gradients`. Quem o usa é o `ACKTR` (`snakeai/agents/acktr.py`), que é
o uso historicamente correto do K-FAC em RL.

Fisher de verdade, não Fisher empírico
--------------------------------------
`G_ℓ` tem que ser a covariância dos gradientes do **log-likelihood do modelo com rótulos
amostrados do próprio modelo** — não dos gradientes da perda de RL. Usar a perda de RL dá o
*Fisher empírico*, que é uma matriz diferente e um pré-condicionador reconhecidamente pior:
perto de um ótimo ele colapsa, porque os gradientes vão a zero por acerto, não por
curvatura baixa. Então o `passo_kfac` faz **duas** retropropagações sobre o mesmo forward:

1. a perda real, que dá o gradiente a ser pré-condicionado;
2. a *perda de Fisher* — `log π(a')` com `a' ~ π(·|s)` amostrada, mais um alvo gaussiano
   para o crítico — que dá **só** as estatísticas de `G`.

Amortecimento
-------------
`A` e `G` são singulares na prática (mais parâmetros que amostras no lote). O amortecimento
de Tikhonov fatorado (Martens & Grosse, §6.3) distribui `λ` entre os dois fatores de forma
que `(A + √λ·π·I) ⊗ (G + √λ/π·I)` fique o mais perto possível de `A⊗G + λ·I`::

    π = sqrt( (tr(A)/dim A) / (tr(G)/dim G) )

Sem o `π`, o amortecimento cai desigual sobre os dois lados e a direção sai enviesada.

Cobertura
---------
Cobre `Dense` e `Conv2D` — que no `resnet_tiny` são 96% dos parâmetros. As camadas restantes
(`GroupNormalization`) recebem o gradiente cru. `KFac.resumo()` diz exatamente qual fração
dos parâmetros está sob pré-condicionamento, porque um K-FAC que cobre metade da rede e não
avisa é pior que nenhum.
"""

from __future__ import annotations

import contextlib
import types

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf
from keras import layers

__all__ = ["KFac", "captura_kfac", "patches_de_entrada", "fatores_de_camada",
           "perda_fisher_categorica", "perda_fisher_gaussiana"]

REGISTRAVEIS = (layers.Dense, layers.Conv2D)


# --------------------------------------------------------------------- captura
def _call_dense(self, inputs, *a, **kw):
    z = keras.ops.matmul(inputs, self.kernel)
    if self.use_bias:
        z = z + self.bias
    _REGISTRO[-1].append((self, inputs, z))
    return self.activation(z) if self.activation is not None else z


def _call_conv(self, inputs, *a, **kw):
    z = tf.nn.convolution(
        inputs, self.kernel,
        strides=list(self.strides), padding=self.padding.upper(),
        dilations=list(self.dilation_rate),
    )
    if self.use_bias:
        z = tf.nn.bias_add(z, self.bias)
    _REGISTRO[-1].append((self, inputs, z))
    return self.activation(z) if self.activation is not None else z


#: Pilha de listas de captura. Pilha, e não uma lista só, para que um `captura_kfac`
#: aninhado (ou uma retraçagem do `tf.function` no meio de outra) não misture tensores de
#: escopos diferentes — que seria um bug silencioso, do tipo que dá números plausíveis.
_REGISTRO = []

_AUSENTE = object()


@contextlib.contextmanager
def captura_kfac(camadas):
    """Durante o bloco, cada camada de `camadas` registra `(camada, entrada, pré-ativação)`.

    Reimplementa o `call` de `Dense` e `Conv2D` porque a pré-ativação **não existe como
    tensor** quando a ativação vem fundida (`Dense(64, activation="relu")`): o Keras calcula
    `relu(x @ W + b)` de uma vez e só o resultado final vira nó do grafo. `G` precisa do
    gradiente em `x @ W + b`, antes da ativação.

    Duplicar a semântica de uma camada é arriscado — por isso
    `tests/test_kfac.py::test_captured_forward_is_bit_identical` compara a saída do modelo
    com e sem captura e exige igualdade exata.
    """
    _REGISTRO.append([])
    tocadas = []
    try:
        for c in camadas:
            novo = _call_dense if isinstance(c, layers.Dense) else _call_conv
            # Guarda o `call` de instância anterior — ou a ausência dele, que é o caso
            # normal. Restaurar com `c.call = c.call` deixaria um atributo de instância
            # sombreando o método da classe para sempre.
            tocadas.append((c, c.__dict__.get("call", _AUSENTE)))
            c.call = types.MethodType(novo, c)
        yield _REGISTRO[-1]
    finally:
        for c, anterior in tocadas:
            if anterior is _AUSENTE:
                c.__dict__.pop("call", None)
            else:
                c.call = anterior
        _REGISTRO.pop()


# ---------------------------------------------------------------- perda de Fisher
def perda_fisher_categorica(logits, mask=None, seed=None):
    """`log π(a')` com `a' ~ π(·|s)`. É daqui que sai `G` da cabeça de política.

    O detalhe que decide entre Fisher e Fisher empírico está na **origem da ação**: aqui
    ela é amostrada da política atual, não é a ação que o agente de fato tomou. Usar a ação
    tomada daria o Fisher empírico — outra matriz, e um pré-condicionador que degenera perto
    do ótimo, quando os gradientes vão a zero por acerto e não por curvatura baixa.
    """
    if mask is not None:
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e9))
    amostra = tf.random.categorical(logits, 1, seed=seed)[:, 0]
    logp = tf.nn.log_softmax(logits)
    return tf.reduce_mean(tf.gather(logp, amostra, batch_dims=1))


def perda_fisher_gaussiana(valor, seed=None):
    """Alvo gaussiano de variância 1 em volta da própria predição — `G` do crítico.

    O crítico não tem distribuição de saída explícita; o K-FAC trata a regressão como uma
    gaussiana de variância unitária, que é o que faz a "Fisher" do valor ser a
    Gauss-Newton. Na prática o gradiente que chega na pré-ativação é ruído branco puro,
    então `G ≈ I` e o pré-condicionamento do crítico vira Newton sobre `A`.

    Normalização, que é onde isto costuma sair errado por um fator igual à dimensão de
    saída: a log-verossimilhança de uma gaussiana `d`-dimensional **soma** sobre as `d`
    componentes e só depois tira a média sobre o lote. Trocar essa soma por uma média
    encolhe `G` por `d` e o passo natural sai `d` vezes maior — o que num crítico escalar
    (`d = 1`) não faz diferença nenhuma e por isso passa despercebido até alguém usar uma
    saída vetorial.
    """
    ruido = tf.random.normal(tf.shape(valor), seed=seed)
    alvo = tf.stop_gradient(valor) + ruido
    quadrado = tf.square(valor - alvo)
    return 0.5 * tf.reduce_mean(tf.reduce_sum(
        tf.reshape(quadrado, [tf.shape(quadrado)[0], -1]), axis=-1))


# ---------------------------------------------------------------------- fatores
def patches_de_entrada(camada, entrada):
    """Achata a entrada de uma camada na forma `(amostras, dim_entrada)` que `A` espera.

    Para `Dense` é a própria entrada. Para `Conv2D` é o truque do KFC (Grosse & Martens,
    2016): cada posição espacial da saída consome um *patch* `kh × kw × cin` da entrada, e
    a convolução é uma `Dense` aplicada a cada patch. Extraindo os patches, a camada
    convolucional vira densa e o resto da conta é idêntico.

    A ordem do achatamento (`kh, kw, cin`) é a mesma de `kernel.reshape(-1, cout)` — o que
    não é óbvio e por isso está testado em `test_conv_patches_reproduce_the_convolution`.
    """
    if isinstance(camada, layers.Dense):
        return tf.reshape(entrada, [-1, tf.shape(entrada)[-1]])

    kh, kw = camada.kernel_size
    sh, sw = camada.strides
    dh, dw = camada.dilation_rate
    p = tf.image.extract_patches(
        entrada, sizes=[1, kh, kw, 1], strides=[1, sh, sw, 1],
        rates=[1, dh, dw, 1], padding=camada.padding.upper(),
    )
    return tf.reshape(p, [-1, kh * kw * entrada.shape[-1]])


def fatores_de_camada(camada, entrada, grad_pre):
    """Devolve `(A, G, n_amostras, n_posicoes)` para uma camada.

    Convenções de escala, que decidem se o pré-condicionamento está certo ou só parece
    certo. A perda é uma **média** sobre `N` amostras, então o gradiente de pré-ativação
    por amostra vale `N · g`. Daí::

        A = (1/N) Σ â âᵀ            â = [a, 1] se a camada tem viés
        G = (N/T) Σ g gᵀ            T = posições espaciais (1 em `Dense`)

    e `Δ = A⁻¹ ∇W G⁻¹`, com `∇W` na forma `(dim_entrada[+1], saída)`.
    """
    a = patches_de_entrada(camada, entrada)
    n = tf.cast(tf.shape(entrada)[0], tf.float32)
    t = tf.cast(tf.shape(a)[0], tf.float32) / n

    if camada.use_bias:
        a = tf.concat([a, tf.ones([tf.shape(a)[0], 1], a.dtype)], axis=1)

    g = tf.reshape(grad_pre, [-1, tf.shape(grad_pre)[-1]])

    A = tf.matmul(a, a, transpose_a=True) / n
    G = tf.matmul(g, g, transpose_a=True) * (n / t)
    return A, G, n, t


# ------------------------------------------------------------------------ K-FAC
class KFac:
    """Pré-condicionador K-FAC para um `keras.Model` funcional.

    Uso::

        kf = KFac(model, damping=1e-2)
        with captura_kfac(kf.camadas) as cap:
            with tf.GradientTape(persistent=True) as tape:
                ...  # forward
                perda, perda_fisher = ...
            grads = tape.gradient(perda, model.trainable_variables)
            gs = tape.gradient(perda_fisher, [z for _, _, z in cap])
        kf.acumula(cap, gs)
        nat = kf.precondiciona(grads)
    """

    def __init__(self, model, damping=1e-2, ema=0.95, inv_every=20, eps=1e-8):
        self.model = model
        self.damping = float(damping)
        self.ema = float(ema)
        self.inv_every = int(inv_every)
        self.eps = float(eps)

        self.camadas = [c for c in model.layers
                        if isinstance(c, REGISTRAVEIS) and c.trainable_weights]
        if not self.camadas:
            raise ValueError("nenhuma camada Dense ou Conv2D registrável no modelo")

        self._A, self._G = {}, {}
        self._cholA, self._cholG = {}, {}
        self._passos = 0
        #: Índice das variáveis de cada camada dentro de `model.trainable_variables`.
        #: Guardado por `id`, porque comparar `tf.Variable` com `==` faz broadcast.
        ordem = {id(v): i for i, v in enumerate(model.trainable_variables)}
        self._idx = {c.name: (ordem[id(c.kernel)],
                              ordem[id(c.bias)] if c.use_bias else None)
                     for c in self.camadas}

    # ------------------------------------------------------------- estatísticas
    def acumula(self, capturado, grads_pre):
        """Atualiza as médias móveis de `A` e `G` com um lote."""
        for (camada, entrada, _), gp in zip(capturado, grads_pre):
            if gp is None:
                continue
            A, G, _, _ = fatores_de_camada(camada, entrada, gp)
            nome = camada.name
            if nome in self._A:
                d = self.ema
                self._A[nome] = d * self._A[nome] + (1.0 - d) * A
                self._G[nome] = d * self._G[nome] + (1.0 - d) * G
            else:
                self._A[nome], self._G[nome] = A, G
        self._passos += 1
        if (self._passos - 1) % self.inv_every == 0:
            self.atualiza_inversos()

    def atualiza_inversos(self):
        """Refatora `A` e `G` amortecidos. Cholesky, não inversa explícita.

        `cholesky_solve` resolve o sistema com metade das operações de uma inversão e sem
        o erro de arredondamento de multiplicar por uma inversa formada explicitamente.
        """
        for nome in self._A:
            A, G = self._A[nome], self._G[nome]
            dA = tf.cast(tf.shape(A)[0], tf.float32)
            dG = tf.cast(tf.shape(G)[0], tf.float32)
            trA = tf.linalg.trace(A) / dA
            trG = tf.linalg.trace(G) / dG
            pi = tf.sqrt((trA + self.eps) / (trG + self.eps))
            raiz = np.sqrt(self.damping)
            self._cholA[nome] = tf.linalg.cholesky(
                A + tf.eye(tf.shape(A)[0]) * (raiz * pi))
            self._cholG[nome] = tf.linalg.cholesky(
                G + tf.eye(tf.shape(G)[0]) * (raiz / pi))

    # ----------------------------------------------------------- condicionamento
    def precondiciona(self, grads):
        """`grads` cru → direção natural. Camadas não cobertas passam intactas."""
        if not self._cholA:
            return list(grads)
        saida = list(grads)
        for camada in self.camadas:
            nome = camada.name
            if nome not in self._cholA:
                continue
            ik, ib = self._idx[nome]
            gk = grads[ik]
            if gk is None:
                continue

            forma = tf.shape(camada.kernel)
            cout = camada.kernel.shape[-1]
            plano = tf.reshape(gk, [-1, cout])
            if ib is not None:
                plano = tf.concat([plano, tf.reshape(grads[ib], [1, cout])], axis=0)

            # Δ = A⁻¹ ∇W G⁻¹, via dois sistemas triangulares
            x = tf.linalg.cholesky_solve(self._cholA[nome], plano)
            x = tf.transpose(tf.linalg.cholesky_solve(
                self._cholG[nome], tf.transpose(x)))

            if ib is not None:
                saida[ib] = x[-1]
                x = x[:-1]
            saida[ik] = tf.reshape(x, forma)
        return saida

    # ------------------------------------------------------------- região de confiança
    @staticmethod
    def escala_kl(naturais, crus, kl_max, lr_max):
        """Passo maior que respeita uma KL alvo — a parte "trust region" do ACKTR.

        Com `Δ = F⁻¹∇`, a KL de segunda ordem induzida por um passo `ηΔ` vale
        `½η² ΔᵀFΔ`, e `ΔᵀFΔ = Δᵀ∇` — um produto interno, sem tocar em `F`. Igualando a
        `kl_max` sai `η = sqrt(2·kl_max / Δᵀ∇)`, limitado por `lr_max`.

        É isto que permite ao ACKTR usar passos que derrubariam um A2C: o tamanho não é
        fixo, é o maior que ainda cabe dentro da KL pedida.
        """
        quad = tf.add_n([tf.reduce_sum(n * g)
                         for n, g in zip(naturais, crus) if n is not None and g is not None])
        quad = tf.maximum(quad, 1e-12)
        return tf.minimum(lr_max, tf.sqrt(2.0 * kl_max / quad))

    # ------------------------------------------------------------------- relato
    def resumo(self):
        """Quantos parâmetros estão de fato sob pré-condicionamento."""
        cobertos = sum(int(np.prod(v.shape))
                       for c in self.camadas for v in c.trainable_weights)
        total = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        return {
            "camadas": [c.name for c in self.camadas],
            "params_cobertos": cobertos,
            "params_total": total,
            "fracao": cobertos / max(1, total),
            "maior_fator": max((int(self._A[n].shape[0]) for n in self._A), default=0),
        }
