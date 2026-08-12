"""Exportar o modelo — `.keras` para retomar treino, TFLite para embarcar.

Uma armadilha silenciosa do Keras 3, registrada aqui para ninguém repetir
--------------------------------------------------------------------------
Converter para TFLite **precisa passar por um SavedModel**.
`TFLiteConverter.from_concrete_functions(...)` compila sem erro, gera um arquivo
minúsculo — e **não captura os pesos**. A inferência devolve NaN, sem nenhum aviso. O
sintoma é um `.tflite` de poucos KB quando deveria ter centenas.

Por isso `export_model` sempre passa por `model.export(dir, format="tf_saved_model")`, e
sempre valida a paridade contra o modelo original antes de declarar sucesso.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np

from .env.vec_snake import N_ACTIONS, N_CHANNELS

__all__ = ["export_model", "medir_latencia", "conferir_paridade"]


def medir_latencia(fn, board_size=10, repeticoes=200, aquecimento=20):
    """Latência de inferência com lote 1 — o que importa se o modelo for para o jogo."""
    x = np.zeros((1, board_size, board_size, N_CHANNELS), dtype=np.float32)
    for _ in range(aquecimento):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(repeticoes):
        fn(x)
    return (time.perf_counter() - t0) / repeticoes * 1000.0


def conferir_paridade(modelo, blob_tflite, board_size=10, n=200, seed=0):
    """O `.tflite` escolhe a mesma ação que o `.keras`, em `n` estados aleatórios?

    Não basta comparar os logits: o que importa para o jogo é a **ação escolhida**. Uma
    diferença numérica de quantização é aceitável; uma ação diferente não é.
    """
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, board_size, board_size, N_CHANNELS)).astype(np.float32)

    saida = modelo(x, training=False)
    logits_keras = np.asarray(saida[0] if isinstance(saida, (list, tuple)) else saida)

    itp = tf.lite.Interpreter(model_content=blob_tflite)
    itp.allocate_tensors()
    entrada = itp.get_input_details()[0]
    saidas = itp.get_output_details()

    logits_lite = []
    for i in range(n):
        itp.set_tensor(entrada["index"], x[i: i + 1])
        itp.invoke()
        cand = [itp.get_tensor(o["index"]) for o in saidas]
        # a saída de política é a que tem N_ACTIONS colunas
        pol = next((c for c in cand if c.shape[-1] == N_ACTIONS), cand[0])
        logits_lite.append(pol[0])
    logits_lite = np.array(logits_lite)

    if logits_keras.ndim == 3:      # C51: colapsa átomos só para comparar a escolha
        logits_keras = logits_keras.mean(-1)
    iguais = (logits_keras.argmax(1) == logits_lite.argmax(1)).mean()
    return {
        "acoes_iguais": float(iguais),
        "erro_max_logits": float(np.abs(logits_keras - logits_lite).max()),
    }


def export_model(modelo, out_dir="export", board_size=10, formatos=("fp16", "int8"),
                 validar=True):
    """Exporta e **mede**: tamanho, latência e paridade de ação.

    Devolve um dicionário pronto para virar linha do `MODELS.md`.
    """
    import tensorflow as tf

    os.makedirs(out_dir, exist_ok=True)
    caminho_keras = os.path.join(out_dir, "modelo.keras")
    modelo.save(caminho_keras)

    resultado = {
        "params": int(modelo.count_params()),
        "keras_kb": round(os.path.getsize(caminho_keras) / 1024, 1),
        "tf_ms": round(medir_latencia(lambda x: modelo(x, training=False), board_size), 4),
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

        resultado[f"{nome}_ms"] = round(medir_latencia(roda, board_size), 4)

        if validar:
            resultado[f"{nome}_paridade"] = conferir_paridade(modelo, blob, board_size)
            if resultado[f"{nome}_kb"] < resultado["params"] / 4096:
                resultado[f"{nome}_alerta"] = (
                    "arquivo pequeno demais para esse número de parâmetros — "
                    "provável perda de pesos na conversão"
                )

    return resultado
