"""Exportar o modelo — `.keras` para retomar treino, TFLite para embarcar.

Uma armadilha silenciosa do Keras 3, registrada aqui para ninguém repetir
--------------------------------------------------------------------------
Converter para TFLite **precisa passar por um SavedModel**.
`TFLiteConverter.from_concrete_functions(...)` compila sem erro, gera um arquivo
minúsculo — e **não captura os pesos**. A inferência devolve NaN, sem nenhum aviso. O
sintoma é um `.tflite` de poucos KB quando deveria ter centenas.

Por isso `export_model` sempre passa por `model.export(dir, format="tf_saved_model")`, e
sempre valida a paridade contra o modelo original antes de declarar sucesso.

A segunda armadilha: "a saída da rede" não tem uma forma só
-----------------------------------------------------------
A paridade compara a **ação escolhida**, e por muito tempo esse cálculo assumiu que a
saída da política é `(lote, ações)`. Neste repositório ela é isso em três dos formatos e
outra coisa nos demais:

======================================  ==============================  ================
construtor                              saída de política                quem usa
======================================  ==============================  ================
``build_actor_critic``                  ``(lote, ações)``                PPO, A2C, ACKTR…
``build_q_network`` (sem C51)           ``(lote, ações)``                DQN
``build_q_network`` (``n_atoms > 0``)   ``(lote, ações, átomos)``        Rainbow, C51
``build_actor_critic_populacao``        ``(lote, políticas, ações)``     LBC
``build_policy_q``                      ``(lote, ações)`` **duas vezes** ACER
======================================  ==============================  ================

O eixo das ações muda de lugar, e no ACER duas saídas diferentes têm exatamente a mesma
forma. Por isso a redução para "um escore por ação" mora em `_escores_por_acao`, e a
escolha de qual tensor do `.tflite` corresponde à política mora em `_indice_da_politica`
— as duas aplicadas **do mesmo jeito nos dois lados** da comparação. Reduzir só o lado
Keras é o defeito que quebrava o Rainbow no fim de um treino inteiro:

    ValueError: operands could not be broadcast together with shapes (200,) (200,121)

É o mesmo erro de `DQN.politica_do_modelo` (§2.17) um passo adiante — lá o C51 quebrava a
avaliação do checkpoint, aqui quebrava a exportação. Ver `docs/REVISAO_ALGORITMOS.md`.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np

from .env.vec_snake import N_ACTIONS, N_CHANNELS

__all__ = ["export_model", "medir_latencia", "conferir_paridade", "canais_do_modelo"]


def canais_do_modelo(modelo, padrao=N_CHANNELS):
    """Quantos canais a rede espera na entrada — **perguntando à rede**, não à constante.

    O contrato são 5 canais, mas uma execução com `canal_fome=True` treina uma rede de 6.
    Exportar essa rede alimentando-a com a constante quebra na primeira inferência, com
    uma mensagem sobre formas — e isso acontece **depois** do treino inteiro, na última
    célula do notebook. Ver `snakeai.eval.evaluate`, que tem o mesmo cuidado.
    """
    try:
        forma = modelo.input_shape
        if isinstance(forma, (list, tuple)) and forma and isinstance(forma[0], (list, tuple)):
            forma = forma[0]                      # modelos de múltiplas entradas
        canais = forma[-1]
        return int(canais) if canais else padrao
    except Exception:                             # rede sem `input_shape` conhecido
        return padrao


def medir_latencia(fn, board_size=10, repeticoes=200, aquecimento=20, canais=N_CHANNELS):
    """Latência de inferência com lote 1 — o que importa se o modelo for para o jogo."""
    x = np.zeros((1, board_size, board_size, canais), dtype=np.float32)
    for _ in range(aquecimento):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        fn(x)
    return (time.perf_counter() - t0) / repeticoes * 1000.0


def _q_de_logits_c51(logits):
    """`(…, ações, átomos)` de logits → `(…, ações)`, para **escolher a ação**.

    O `Q` do C51 é `Σ_z p(z)·z` com `z` no suporte `linspace(v_min, v_max, n_atoms)`. O
    exportador não conhece `v_min`/`v_max` — e não precisa: como o suporte é afim e
    crescente no índice do átomo (`z_i = v_min + i·Δz`, com `Δz > 0`), vale

        argmax_a Σ_i p(a,i)·z_i  =  argmax_a Σ_i p(a,i)·i

    ou seja, a **ação escolhida** não depende do suporte, só da esperança do índice. É por
    isso que esta função devolve o índice esperado em vez do `Q` de verdade: o número não
    é o `Q`, mas o `argmax` é o mesmo, e é só o `argmax` que esta comparação usa.

    A média simples dos logits, que estava aqui antes, **não** tem essa propriedade — ela
    ignora a softmax e pode trocar a ação escolhida.
    """
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    indices = np.arange(p.shape[-1], dtype=p.dtype)
    return (p * indices).sum(axis=-1)


def _escores_por_acao(t, n_actions=N_ACTIONS):
    """Reduz a saída da política a **um escore por ação**, com as ações no último eixo.

    Cobre os três formatos que os construtores de `snakeai.nets.registry` produzem:

    * `(lote, ações)` — devolvido como está;
    * `(lote, políticas, ações)` (LBC) — devolvido como está, e o `argmax` do chamador
      passa a comparar a escolha de **cada** cabeça da população. Conferir todas é mais
      forte que conferir só a `indice_alvo`, que o exportador não conhece;
    * `(lote, ações, átomos)` (C51) — colapsado por `_q_de_logits_c51`.

    A desambiguação é por posição do eixo com `N_ACTIONS`, com o **último** ganhando o
    desempate: `(lote, 3, 3)` é a população de três políticas do LBC, não um C51 de três
    átomos — que seria uma configuração sem sentido (o C51 existe para ter resolução).
    """
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2:
        return t
    if t.ndim >= 3 and t.shape[-1] == n_actions:
        return t
    if t.ndim == 3 and t.shape[1] == n_actions:
        return _q_de_logits_c51(t)
    raise ValueError(
        f"saída de forma {t.shape} sem um eixo de {n_actions} ações reconhecível — "
        "se for um formato novo de política, ensine-o a `_escores_por_acao`"
    )


def _indice_da_politica(candidatos, referencia):
    """Qual das saídas do `.tflite` é a política — casando a **forma** com a do Keras.

    A regra antiga era "a que tem `N_ACTIONS` colunas", e ela erra dos dois jeitos:

    * **não acha** a saída certa quando a política é `(lote, ações, átomos)` (C51: a
      última dimensão são os átomos), e caía no `cand[0]` sem avisar;
    * **acha duas** no ACER, cujas saídas `logits` e `Q(s,·)` têm a mesma forma, e no LBC,
      onde `(lote, 3, 3)` e `(lote, 3)` casam as duas. A ordem das saídas do
      `Interpreter` não é a ordem das saídas do `keras.Model` — o SavedModel as nomeia
      `output_0`, `output_1`… e o conversor pode reordená-las. Pegar a primeira é sortear.

    Aqui a forma decide, e quando ela empata o **valor** desempata: a saída correta é a
    que se parece com a do Keras. Isso é exatamente o que a paridade afirma, então usar o
    critério para escolher não enfraquece nada — se nenhuma das candidatas se parecer com
    a referência, todas reprovam igual.
    """
    forma = tuple(referencia.shape[1:])
    iguais = [i for i, c in enumerate(candidatos) if tuple(c.shape[1:]) == forma]
    if not iguais:
        formas = ", ".join(str(tuple(c.shape[1:])) for c in candidatos)
        raise ValueError(
            f"nenhuma saída do .tflite tem a forma da política do Keras {forma} — "
            f"saídas disponíveis: {formas}"
        )
    if len(iguais) == 1:
        return iguais[0]
    return min(iguais, key=lambda i: float(np.abs(candidatos[i] - referencia).max()))


def conferir_paridade(modelo, blob_tflite, board_size=10, n=200, seed=0, canais=None):
    """O `.tflite` escolhe a mesma ação que o `.keras`, em `n` estados aleatórios?

    Não basta comparar os logits: o que importa para o jogo é a **ação escolhida**. Uma
    diferença numérica de quantização é aceitável; uma ação diferente não é.

    Os dois lados passam pela **mesma** redução (`_escores_por_acao`) — reduzir só um deles
    é comparar coisas de formas diferentes, que é como isto quebrava no Rainbow.
    """
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    canais = canais or canais_do_modelo(modelo)
    x = rng.normal(size=(n, board_size, board_size, canais)).astype(np.float32)

    saida = modelo(x, training=False)
    logits_keras = np.asarray(saida[0] if isinstance(saida, (list, tuple)) else saida,
                              dtype=np.float32)

    itp = tf.lite.Interpreter(model_content=blob_tflite)
    itp.allocate_tensors()
    entrada = itp.get_input_details()[0]
    saidas = itp.get_output_details()

    indice, logits_lite = None, []
    for i in range(n):
        itp.set_tensor(entrada["index"], x[i: i + 1])
        itp.invoke()
        cand = [itp.get_tensor(o["index"]) for o in saidas]
        if indice is None:
            indice = _indice_da_politica(cand, logits_keras[i: i + 1])
        logits_lite.append(cand[indice][0])
    logits_lite = np.asarray(logits_lite, dtype=np.float32)

    escolha_keras = _escores_por_acao(logits_keras).argmax(-1)
    escolha_lite = _escores_por_acao(logits_lite).argmax(-1)
    return {
        "acoes_iguais": float((escolha_keras == escolha_lite).mean()),
        "erro_max_logits": float(np.abs(logits_keras - logits_lite).max()),
    }


def export_model(modelo, out_dir="export", board_size=10, formatos=("fp16", "int8"),
                 validar=True):
    """Exporta e **mede**: tamanho, latência e paridade de ação.

    Devolve um dicionário pronto para virar linha do `MODELS.md`.

    Uma falha da **conferência** não derruba a exportação: ela vira
    `{"erro": ...}` no relatório, no lugar de `acoes_iguais`. Os arquivos já estão em
    disco quando ela roda, e esta função é a penúltima célula de um notebook que gastou
    horas de GPU — deixar a validação levar o treino junto foi exatamente o que aconteceu
    com o Rainbow. O relatório é impresso, então a falha continua visível; o que ela não
    faz mais é apagar o resto.
    """
    import tensorflow as tf

    os.makedirs(out_dir, exist_ok=True)
    canais = canais_do_modelo(modelo)
    caminho_keras = os.path.join(out_dir, "modelo.keras")
    modelo.save(caminho_keras)

    resultado = {
        "params": int(modelo.count_params()),
        "keras_kb": round(os.path.getsize(caminho_keras) / 1024, 1),
        "canais": canais,
        "tf_ms": round(medir_latencia(lambda x: modelo(x, training=False), board_size,
                                      canais=canais), 4),
    }

    sm_dir = os.path.join(out_dir, "saved_model")
    if os.path.isdir(sm_dir):
        shutil.rmtree(sm_dir)
    modelo.export(sm_dir, format="tf_saved_model")

    for nome in formatos:
        conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        if nome != "fp32":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if nome == "fp16":
            conv.target_spec.supported_types = [tf.float16]
        blob = conv.convert()

        caminho = os.path.join(out_dir, f"modelo_{nome}.tflite")
        with open(caminho, "wb") as f:
            f.write(blob)
        resultado[f"{nome}_kb"] = round(len(blob) / 1024, 1)

        itp = tf.lite.Interpreter(model_content=blob)
        itp.allocate_tensors()
        ent = itp.get_input_details()[0]
        xi = np.zeros(ent["shape"], dtype=np.float32)

        def roda(_x, _itp=itp, _ent=ent):
            _itp.set_tensor(_ent["index"], xi)
            _itp.invoke()

        resultado[f"{nome}_ms"] = round(medir_latencia(roda, board_size,
                                                       canais=canais), 4)

        if validar:
            try:
                resultado[f"{nome}_paridade"] = conferir_paridade(modelo, blob, board_size,
                                                                  canais=canais)
            except Exception as e:                # noqa: BLE001 — ver docstring
                resultado[f"{nome}_paridade"] = {"erro": f"{type(e).__name__}: {e}"}
            if resultado[f"{nome}_kb"] < resultado["params"] / 4096:
                resultado[f"{nome}_alerta"] = (
                    "arquivo pequeno demais para esse número de parâmetros — "
                    "provável perda de pesos na conversão"
                )

    return resultado
