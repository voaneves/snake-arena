"""Quantos parâmetros cada notebook treina, na configuração padrão dele.

Por que isto existe
-------------------
A arena iguala o orçamento de **passos de ambiente**. Ela não iguala capacidade — e a
capacidade variava 19× entre os doze sem que nada dissesse isso em lugar nenhum. O
`params` do `docs/RESULTADOS.md` só aparece **depois** que a execução roda, então a
pergunta "o ACER tem quase o dobro do A2C?" não tinha onde ser respondida antes de gastar
5 M de passos para descobrir.

Esta tabela é o número **antes** da execução, computado dos mesmos construtores que os
agentes chamam. Ela não julga: um número maior não é erro, é um confundidor declarado. O
lugar de julgar é a comparação, e `docs/COMPARABILITY.md` é onde ela mora.

O que está sendo contado
------------------------
* **`model`** — o que vai para o `.keras` e o que a coluna `params` da arena publica.
* **extras** — o resto do que o otimizador move. Dois agentes têm, por motivos
  diferentes: o DreamerV3, cujo `self.model` é só o **ator** e o modelo do mundo inteiro
  fica fora (`modelos_extra()`, §1.4); e o MuZero, cujo `model` é o composto `h`+`f` e
  deixa de fora a **dinâmica** `g` — que ele salva por conta própria, em três `.keras`
  separados, e por isso não aparece em `modelos_extra()`.
* **total** — o que o otimizador de fato move. Redes-alvo não entram: o alvo do DQN e o
  `critico_alvo` do Dreamer são cópias da mesma arquitetura, e contá-las dobraria a
  capacidade sem que exista capacidade nova.

Uso::

    python tools/tabela_parametros.py              # tabela markdown no stdout
    python tools/tabela_parametros.py --json       # os mesmos números, para um teste
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

#: Um por notebook de algoritmo, na ordem em que a arena os lista. `99_ablacoes` fica de
#: fora: ele não tem configuração padrão, é uma varredura de troncos e otimizadores.
ALGORITMOS = [
    ("01_ppo", "PPO", "PPOConfig"),
    ("02_dqn", "DQN", "DQNConfig"),
    ("03_rainbow", "Rainbow", "RainbowConfig"),
    ("04_a2c", "A2C", "A2CConfig"),
    ("05_acer", "ACER", "ACERConfig"),
    ("06_alphazero", "AlphaZero", "AlphaZeroConfig"),
    ("07_muzero", "MuZero", "MuZeroConfig"),
    ("08_acktr", "ACKTR", "ACKTRConfig"),
    ("09_dreamerv3", "DreamerV3", "DreamerV3Config"),
    ("10_lbc", "LBC", "LBCConfig"),
    ("11_soap", "SOAP", "SOAPConfig"),
    ("12_acektr", "ACEKTR", "ACEKTRConfig"),
]

#: Campos encolhidos só para o agente **caber na memória** ao ser construído: o replay do
#: DQN aloca `memory_size × board² × canais` de uma vez. Nenhum deles entra em construtor
#: de rede nenhum — a arquitetura sai de `board_size`, `net`, `canais` e das flags — então
#: o número contado é o da configuração padrão. `_confere_que_nao_muda_a_rede` prova isso
#: em vez de pedir confiança.
ENCOLHER = {"num_envs": 2, "memory_size": 64, "total_steps": 32, "warmup_steps": 0,
            "batch_size": 8, "eval_episodes": 2, "eval_envs": 2, "salvar_gif": False,
            "salvar_grafico": False}

#: Os campos que **de fato** definem a rede. Se um destes aparecesse em `ENCOLHER`, o
#: número publicado seria de outra arquitetura.
DA_REDE = {"board_size", "net", "canal_fome", "largura_densa", "dueling", "noisy",
           "n_atoms", "n_politicas", "n_opcoes", "n_bins", "preset", "unroll"}


def _confere_que_nao_muda_a_rede():
    colisao = ENCOLHER.keys() & DA_REDE
    if colisao:
        raise AssertionError(
            f"{sorted(colisao)} muda a arquitetura — a tabela deixaria de ser da "
            "configuração padrão"
        )


def _instancia(agente, config):
    from dataclasses import fields

    import snakeai.agents as A

    Config = getattr(A, config)
    nomes = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in ENCOLHER.items() if k in nomes})
    return getattr(A, agente)(cfg)


def _aquece(ag):
    """Uma jogada, só para as camadas preguiçosas nascerem.

    A GRU do DreamerV3 e a projeção de entrada dela são `Layer` cruas, construídas na
    primeira passada — antes disso `count_params()` levanta em vez de devolver zero, e
    contar zero seria pior: o modelo do mundo é a maior peça da tabela.
    """
    import numpy as np

    canais = getattr(ag.env, "n_channels", 5)
    b = ag.cfg.board_size
    obs = np.zeros((2, b, b, canais), dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)
    try:
        ag.politica()(obs, mask)
    except Exception:                    # pragma: no cover — agente sem política pura
        pass


def _params(modelo):
    """Conta pelos pesos, não por `count_params()`, que exige a camada construída."""
    import numpy as np

    return int(sum(np.prod(w.shape) for w in modelo.weights))


#: Sufixo das redes-alvo no repositório: `critico_alvo` no DreamerV3. Elas entram em
#: `modelos_extra()` porque `retomar()` precisa delas em disco — mas **não** são capacidade:
#: são cópias periódicas de uma rede que já está contada. Somá-las contaria o crítico duas
#: vezes, e ainda por cima só no Dreamer, já que o alvo do DQN mora em `self.target` e nunca
#: apareceu nesta conta. A tabela compara capacidade; o critério tem de ser o mesmo nas doze
#: linhas.
SUFIXO_ALVO = "_alvo"


def _extras(ag):
    """O que o agente treina além de `self.model`, sem contar rede-alvo.

    `modelos_extra()` cobre o DreamerV3, e não cobre o MuZero: ele salva `h`, `g` e `f` em
    três arquivos por conta própria, então nunca precisou do `.npz` — mas a dinâmica `g`
    continua sendo capacidade treinada, e sem ela a linha do MuZero na tabela ficaria 40%
    menor do que a rede que ele de fato otimiza. Onde o agente expõe `_variaveis()`, essa
    lista é a resposta certa: é literalmente o que vai para o `apply_gradients`.
    """
    if hasattr(ag, "_variaveis"):
        import numpy as np

        vistos = {id(w) for w in ag.model.weights}
        return int(sum(np.prod(v.shape) for v in ag._variaveis()
                       if id(v) not in vistos))

    extras = ag.modelos_extra()
    total = 0
    for nome, m in extras.items():
        if not nome.endswith(SUFIXO_ALVO):
            total += _params(m)
            continue
        # o alvo só é descartável porque é cópia; se um dia deixar de ser, isto acusa
        fonte = extras.get(nome[: -len(SUFIXO_ALVO)])
        if fonte is not None and _params(fonte) != _params(m):
            raise AssertionError(
                f"{nome} não é cópia de {nome[: -len(SUFIXO_ALVO)]} — "
                "descartá-la esconderia capacidade de verdade"
            )
    return total


def coleta():
    """Uma linha por algoritmo, com os parâmetros do `model` e dos extras."""
    _confere_que_nao_muda_a_rede()
    linhas = []
    for notebook, agente, config in ALGORITMOS:
        ag = _instancia(agente, config)
        _aquece(ag)
        extras = _extras(ag)
        linhas.append({
            "notebook": notebook,
            "algoritmo": ag.algo,
            "tronco": getattr(ag.cfg, "net", "—"),
            "model": _params(ag.model),
            "extras": int(extras),
            "total": _params(ag.model) + int(extras),
        })
    return linhas


def markdown(linhas):
    ref = next(l["total"] for l in linhas if l["algoritmo"] == "ppo")
    out = ["| notebook | algoritmo | tronco | `model` | extras | total | × PPO |",
           "|---|---|---|---:|---:|---:|---:|"]
    for l in linhas:
        extras = f"{l['extras']:,}" if l["extras"] else "—"
        out.append(f"| `{l['notebook']}` | {l['algoritmo']} | `{l['tronco']}` | "
                   f"{l['model']:,} | {extras} | {l['total']:,} | "
                   f"{l['total'] / ref:.2f}× |")
    return "\n".join(out).replace(",", ".")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="números crus, sem formatação")
    args = ap.parse_args()

    linhas = coleta()
    print(json.dumps(linhas, indent=2) if args.json else markdown(linhas))
