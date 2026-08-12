"""Gera os notebooks do Colab — autocontidos, a partir do pacote.

O acordo que este script implementa
-----------------------------------
O único arquivo que vai para o Colab é o `.ipynb`. Ele precisa abrir e rodar do zero, sem
`git clone`, sem `pip install` do repositório, sem nada. Ao mesmo tempo, o ambiente e o
protocolo de avaliação precisam ser **byte a byte idênticos** em todos os notebooks — senão
as curvas voltam a ser incomparáveis, que é exatamente como os treze notebooks do
`colab-rl` acabaram.

A solução é não escolher: **o pacote é a fonte, o notebook é gerado**. Este script injeta o
código-fonte dos módulos dentro de cada notebook, entre marcadores. Se alguém editar a
cópia dentro do notebook, `tests/test_notebooks.py` quebra e diz qual arquivo divergiu.

Cópia idêntica por construção, não por disciplina.

Uso::

    python tools/gerar_notebooks.py            # gera todos
    python tools/gerar_notebooks.py --check    # só verifica se estão em dia
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

MARCA_INICIO = "# ==== GERADO A PARTIR DO PACOTE — NÃO EDITE AQUI ===="
MARCA_FIM = "# ==== FIM DO CÓDIGO GERADO ===="

#: Módulos comuns a todos os notebooks, em ordem de dependência. São o que **precisa** ser
#: idêntico: o ambiente, a régua de avaliação e o registro.
NUCLEO = [
    "snakeai/env/vec_snake.py",
    "snakeai/eval.py",
    "snakeai/record.py",
    "snakeai/env/render.py",
    "snakeai/export.py",
    "snakeai/plot.py",
    "snakeai/nets/resnet.py",
    "snakeai/nets/classic.py",
    "snakeai/nets/heads.py",
    "snakeai/nets/registry.py",
    "snakeai/agents/base.py",
]

NOTEBOOKS = [
    {
        "arquivo": "01_ppo.ipynb",
        "titulo": "PPO",
        "modulos": ["snakeai/agents/ppo.py"],
        "agente": "PPO",
        "config": "PPOConfig",
        "resumo": "A referência do benchmark. Clipping, GAE(λ), early stop por KL.",
    },
    {
        "arquivo": "02_dqn.ipynb",
        "titulo": "DQN — a família inteira",
        "modulos": ["snakeai/memory/replay.py", "snakeai/agents/dqn.py"],
        "agente": "DQN",
        "config": "DQNConfig",
        "resumo": "double, dueling, PER, noisy, n-step e C51 como flags independentes. "
                  "Ligar todas é Rainbow; nenhuma é o DQN de 2013.",
    },
    {
        "arquivo": "03_a2c.ipynb",
        "titulo": "A2C — o controle experimental",
        "modulos": ["snakeai/agents/ppo.py", "snakeai/agents/a2c.py"],
        "agente": "A2C",
        "config": "A2CConfig",
        "resumo": "PPO sem clipping e sem reaproveitar o rollout. A diferença entre as "
                  "duas curvas mede exatamente quanto essas duas coisas valem.",
    },
    {
        "arquivo": "04_acer.ipynb",
        "titulo": "ACER",
        "modulos": ["snakeai/memory/trajectory.py", "snakeai/agents/acer.py"],
        "agente": "ACER",
        "config": "ACERConfig",
        "resumo": "Retrace(λ), IS truncado com correção de viés, região de confiança.",
    },
    {
        "arquivo": "05_alphazero.ipynb",
        "titulo": "AlphaZero — busca sobre o simulador real",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/agents/alphazero.py"],
        "agente": "AlphaZero",
        "config": "AlphaZeroConfig",
        "resumo": "Snake é determinístico e o simulador é rápido — então a árvore percorre "
                  "o jogo de verdade. Use `num_simulations` alto: é o parâmetro que decide "
                  "se a destilação funciona.",
    },
    {
        "arquivo": "06_muzero.ipynb",
        "titulo": "MuZero — a mesma busca, sobre um modelo aprendido",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/nets/muzero.py", "snakeai/agents/muzero.py"],
        "agente": "MuZero",
        "config": "MuZeroConfig",
        "resumo": "Deve perder para o AlphaZero — o simulador aqui é exato e gratuito. "
                  "O que se mede é quanto custa não tê-lo.",
    },
]

RE_IMPORT_RELATIVO = re.compile(r"^from\s+\.+[\w.]*\s+import\s+.*(\(\s*)?$")
RE_FUTURE = re.compile(r"^from\s+__future__\s+import\s+")


def _limpa(fonte, caminho):
    """Remove imports relativos e docstring de módulo, mantendo o resto intacto."""
    linhas = fonte.splitlines()
    saida, pulando_parenteses = [], False
    for linha in linhas:
        if pulando_parenteses:
            if ")" in linha:
                pulando_parenteses = False
            continue
        if RE_IMPORT_RELATIVO.match(linha.strip()):
            if linha.rstrip().endswith("("):
                pulando_parenteses = True
            continue
        # `from __future__` só é válido na PRIMEIRA linha do arquivo; com N módulos
        # concatenados, a partir do segundo vira SyntaxError. Sai daqui e volta uma vez
        # só, no topo do bloco gerado.
        if RE_FUTURE.match(linha.strip()):
            continue
        saida.append(linha)
    corpo = "\n".join(saida).strip()
    return f"# --- {caminho} ---\n{corpo}\n"


def fonte_combinada(modulos):
    partes = ["from __future__ import annotations"]
    for m in modulos:
        with open(os.path.join(RAIZ, m), encoding="utf-8") as f:
            partes.append(_limpa(f.read(), m))
    return "\n\n".join(partes)


def _hash(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def _md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(True)}


def _code(texto, titulo=None):
    if titulo:
        texto = f"# @title {titulo}\n{texto}"
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": texto.splitlines(True)}


def monta_notebook(spec, usuario="voaneves", repo="snake-arena"):
    modulos = NUCLEO + [m for m in spec["modulos"] if m not in NUCLEO]
    fonte = fonte_combinada(modulos)
    marca = _hash(fonte)
    agente, config = spec["agente"], spec["config"]
    caminho_colab = (f"https://colab.research.google.com/github/{usuario}/{repo}"
                     f"/blob/main/notebooks/{spec['arquivo']}")

    celulas = [
        _md(f"""# snake-arena · {spec['titulo']}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({caminho_colab})

{spec['resumo']}

**Este notebook é autocontido.** Não precisa clonar nada: o ambiente, a rede, o protocolo
de avaliação e o agente estão todos aqui dentro. O código do núcleo é **gerado a partir do
pacote** ([`{usuario}/{repo}`](https://github.com/{usuario}/{repo})) e é byte a byte igual
em todos os notebooks — é isso que torna as curvas comparáveis.

`Runtime → Change runtime type → GPU (T4)` antes de rodar.

Assinatura do código gerado: `{marca}`
"""),
        _code("""import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json, math, time, glob, csv, platform, subprocess, sys, shutil, argparse
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, ops, regularizers

print("TensorFlow", tf.__version__, "| Keras", keras.__version__,
      "| backend", keras.backend.backend())
GPUS = tf.config.list_physical_devices("GPU")
print("GPU:", GPUS or "nenhuma — vai rodar em CPU, muito mais lento")
for g in GPUS:
    tf.config.experimental.set_memory_growth(g, True)
""", "Ambiente"),
        _md(f"""## O núcleo, gerado a partir do pacote

A célula abaixo é **gerada**. Editá-la aqui não muda o repositório e faz o teste
`tests/test_notebooks.py` acusar divergência — o que é de propósito: é o que garante que
os {len(NOTEBOOKS)} notebooks rodem exatamente o mesmo jogo, com a mesma régua.

Para mudar algo aqui, mude no pacote e rode `python tools/gerar_notebooks.py`.
"""),
        _code(f"{MARCA_INICIO}\n# assinatura: {marca}\n\n{fonte}\n\n{MARCA_FIM}"),
        _md(f"""## Configuração

Os padrões abaixo são os do **contrato**: tabuleiro 10×10, 5 M passos de orçamento,
avaliação de 1.000 episódios com semente 123. Mexer neles é legítimo para experimentar,
mas o resultado só entra na arena se o contrato for respeitado — o `Recorder` recusa
qualquer outra coisa e diz o motivo.
"""),
        _code(f"""SEMENTE = 0        # @param {{type:"integer"}}
PASSOS = 5000000   # @param {{type:"integer"}}
REDE = "resnet_small"  # @param ["resnet_tiny", "resnet_small", "resnet_base", "cnn_rainbow", "cnn_alphazero", "cnn_vgg", "cnn_vgg_dropout", "cnn_vgg_sem_pool"]
USAR_DRIVE = False  # @param {{type:"boolean"}}

PASTA = "/content/snake-arena"
if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    PASTA = "/content/drive/MyDrive/snake-arena"

cfg = {config}(
    seed=SEMENTE,
    net=REDE,
    total_steps=PASSOS,
    ckpt_dir=os.path.join(PASTA, "checkpoints"),
    runs_dir=os.path.join(PASTA, "runs"),
)
print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))""", "Parâmetros"),
        _md("""## Treino

**Retomável.** A sessão do Colab vai cair — é questão de quando, não de se. Rode a célula
de novo e ela continua do último checkpoint. Com `USAR_DRIVE = True` os checkpoints
sobrevivem à queda da máquina.
"""),
        _code(f"""agente = {agente}(cfg)
if agente.retomar("last"):
    print("retomando do checkpoint")
print("parâmetros:", f"{{agente.model.count_params():,}}")

registro = agente.train(verbose=True)""", "Treinar"),
        _md("""## Veredito

Três regimes na mesma execução: o piso aleatório, a política pura e a política com o
filtro de segurança. Se a coluna do meio não estiver bem acima do piso, não aprendeu — e
aí o problema é hiperparâmetro ou tempo de treino, não código.
"""),
        _code("""resultado = verdict(agente.politica(), episodes=1000)
print(format_verdict(resultado))

fig, _ = plot_run(registro.record)
plt.show()""", "Veredito"),
        _md("""## O agente jogando

Um GIF vale mais que a curva para entender *como* o agente perde. Morrer preso no próprio
corpo e morrer de fome dão a mesma linha no gráfico e são problemas completamente
diferentes.
"""),
        _code("""from IPython.display import Image, display

for semente in (7, 21, 42):
    caminho, score, motivo = render_episode(
        agente.politica(), caminho=f"episodio_s{semente}.gif", seed=semente)
    print(f"semente {semente}: score {score}, terminou por {motivo}")
    display(Image(filename=caminho))""", "GIF"),
        _md("""## Exportar

`.keras` para retomar treino, TFLite fp16/int8 para embarcar no jogo. A paridade de **ação**
contra o `.keras` é conferida — diferença numérica de quantização é aceitável, ação
diferente não é.
"""),
        _code("""relatorio = export_model(agente.model, out_dir=os.path.join(PASTA, "export"))
print(json.dumps(relatorio, indent=2, ensure_ascii=False))""", "Exportar"),
        _md("""## Onde ficou o resultado

O `history.json` da execução está em `runs/<algo>/<variante>/seed<N>/`, junto com a curva e
os GIFs. **Baixe essa pasta** e coloque em `runs/` do repositório para a execução entrar na
arena — depois basta rodar `python -m snakeai.arena --all`.
"""),
        _code("""print("registro:", registro.save(skip_validation=True))
problemas = validate(registro.record)
print("entra na arena?" , "sim" if not problemas else "NÃO:")
for p in problemas:
    print("  -", p)""", "Conferir o contrato"),
    ]

    return {
        "cells": celulas,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True,
                      "name": spec["arquivo"]},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "snake_arena": {"gerado_de": modulos, "assinatura": marca},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def gerar(destino="notebooks", check=False):
    os.makedirs(os.path.join(RAIZ, destino), exist_ok=True)
    divergentes = []
    for spec in NOTEBOOKS:
        nb = monta_notebook(spec)
        caminho = os.path.join(RAIZ, destino, spec["arquivo"])
        novo = json.dumps(nb, ensure_ascii=False, indent=1)

        if check:
            if not os.path.exists(caminho):
                divergentes.append(f"{spec['arquivo']} não existe")
                continue
            with open(caminho, encoding="utf-8") as f:
                atual = json.load(f)
            a = atual.get("metadata", {}).get("snake_arena", {}).get("assinatura")
            b = nb["metadata"]["snake_arena"]["assinatura"]
            if a != b:
                divergentes.append(
                    f"{spec['arquivo']}: assinatura {a} != {b} — o pacote mudou; "
                    "rode `python tools/gerar_notebooks.py`")
            continue

        with open(caminho, "w", encoding="utf-8") as f:
            f.write(novo)
        print(f"  {caminho}  ({len(novo) / 1024:.0f} kB, assinatura "
              f"{nb['metadata']['snake_arena']['assinatura']})")
    return divergentes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="só verifica se os notebooks estão em dia com o pacote")
    p.add_argument("--destino", default="notebooks")
    args = p.parse_args(argv)

    divergentes = gerar(args.destino, check=args.check)
    if divergentes:
        for d in divergentes:
            print("DIVERGENTE:", d)
        raise SystemExit(1)
    if args.check:
        print("notebooks em dia com o pacote")


if __name__ == "__main__":
    main()
