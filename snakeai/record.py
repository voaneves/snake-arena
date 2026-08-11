"""Registro de execuções — o esquema do `history.json` e o validador do contrato.

Este módulo é o porteiro do benchmark. Toda execução de todo algoritmo escreve o mesmo
arquivo, com os mesmos campos, e passa pela mesma validação antes de virar uma linha no
gráfico. **Um resultado que não valida não entra na arena** — não porque seja ruim, mas
porque não é comparável, que é pior.

A regra vale inclusive para as curvas históricas do `colab-rl`: elas são convertidas para
este mesmo esquema, mas com `comparable=False` e o motivo registrado em `caveat`. Assim
elas aparecem no gráfico como contexto (tracejado cinza) sem nunca serem confundidas com
um competidor.

Sem dependências além da biblioteca padrão e do NumPy — o validador roda no CI em segundos.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "CONTRATO",
    "ContractViolation",
    "RunRecord",
    "Recorder",
    "validate",
    "save",
    "load",
    "load_all",
    "from_legacy_csv",
]

SCHEMA_VERSION = 1

#: Os valores que **todos** os resultados oficiais precisam compartilhar.
#: Espelha a tabela do README; mudar aqui é mudar o contrato, e invalida o histórico.
CONTRATO = {
    "env": "VecSnake",
    "board_size": 10,
    "starve_base": 100,
    "n_channels": 5,
    "n_actions": 3,
    "obs": "egocentric",
    "metric": "score",
    "reward_food": 1.0,
    "reward_death": -1.0,
    "eval_episodes": 1000,
    "eval_seed": 123,
    "eval_greedy": True,
    "eval_safety": False,
}

#: Piso e teto do 10x10, medidos e documentados no README.
PISO_ALEATORIO = 1.21
SCORE_PERFEITO = 97


class ContractViolation(Exception):
    """Levantada quando um registro não obedece ao contrato de comparabilidade."""


# ------------------------------------------------------------------- estrutura
@dataclass
class RunRecord:
    """Uma execução completa de um algoritmo, com curva e resultado final.

    Campos
    ------
    algo, variant, seed
        Identidade da execução. `runs/<algo>/<variant>/seed<N>/history.json`.
    net, params
        Tronco usado e número de parâmetros treináveis — o eixo "arquitetura importa?".
    config
        Hiperparâmetros do agente, como dicionário livre. Não é validado; é documentação.
    env_spec
        O recorte do contrato que esta execução usou. **É** validado.
    curve
        Lista de pontos ao longo do treino. Cada ponto tem, no mínimo, `global_step`;
        `eval_score_mean` aparece só nos passos em que a avaliação rodou.
    final
        O `stats` devolvido por `snakeai.eval.evaluate` no fim.
    comparable, caveat
        `False` marca uma curva que entra no gráfico como contexto histórico, com o
        motivo em `caveat`. Toda execução nova nasce `True`.
    """

    algo: str
    variant: str = "default"
    seed: int = 0
    net: str = ""
    params: int = 0
    config: dict = field(default_factory=dict)
    env_spec: dict = field(default_factory=lambda: dict(CONTRATO))
    curve: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    comparable: bool = True
    caveat: str = ""
    meta: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # ---------------------------------------------------------------- derivados
    @property
    def run_id(self):
        return f"{self.algo}/{self.variant}/seed{self.seed}"

    @property
    def rel_path(self):
        return os.path.join("runs", self.algo, self.variant, f"seed{self.seed}",
                            "history.json")

    def steps(self):
        return np.array([p["global_step"] for p in self.curve], dtype=np.int64)

    def eval_curve(self):
        """`(passos, scores)` só dos pontos em que a avaliação rodou."""
        pts = [p for p in self.curve if p.get("eval_score_mean") is not None]
        x = np.array([p["global_step"] for p in pts], dtype=np.int64)
        y = np.array([p["eval_score_mean"] for p in pts], dtype=np.float64)
        return x, y


# -------------------------------------------------------------------- gravação
class Recorder:
    """Acumula a curva durante o treino e grava o `history.json` no fim.

    Uso típico, dentro do laço de treino::

        rec = Recorder("ppo", variant="resnet_small", seed=0, net="resnet_small",
                       params=model.count_params(), config=asdict(cfg))
        ...
        rec.log(global_step=n, episodes=e, train_score_mean=m)
        rec.log(global_step=n, eval_score_mean=stats["score_mean"])   # nos passos de eval
        ...
        rec.finish(stats)
        rec.save()            # valida antes de escrever; levanta se violar o contrato
    """

    def __init__(self, algo, variant="default", seed=0, net="", params=0,
                 config=None, env_spec=None, root="runs"):
        self.root = root
        self.t0 = time.perf_counter()
        self.record = RunRecord(
            algo=algo, variant=variant, seed=seed, net=net, params=int(params),
            config=dict(config or {}),
            env_spec=dict(env_spec or CONTRATO),
            meta=_ambiente(),
        )

    def log(self, global_step, **metrics):
        """Anexa um ponto à curva. `global_step` é o eixo oficial."""
        ponto = {"global_step": int(global_step),
                 "wall_s": round(time.perf_counter() - self.t0, 3)}
        for k, v in metrics.items():
            ponto[k] = _jsonable(v)
        self.record.curve.append(ponto)
        return ponto

    def finish(self, final_stats, comparable=True, caveat=""):
        self.record.final = {k: _jsonable(v) for k, v in dict(final_stats).items()}
        self.record.comparable = bool(comparable)
        self.record.caveat = str(caveat)
        self.record.meta["wall_s_total"] = round(time.perf_counter() - self.t0, 3)
        return self.record

    def save(self, path=None, skip_validation=False):
        if not skip_validation:
            problemas = validate(self.record)
            if problemas:
                raise ContractViolation(
                    f"{self.record.run_id} viola o contrato:\n  - "
                    + "\n  - ".join(problemas)
                )
        destino = path or os.path.join(self.root, self.record.algo,
                                       self.record.variant,
                                       f"seed{self.record.seed}", "history.json")
        return save(self.record, destino)


def save(record: RunRecord, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, ensure_ascii=False, indent=2)
    return path


def load(path) -> RunRecord:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d.pop("schema_version", None)
    return RunRecord(**d, schema_version=SCHEMA_VERSION)


def load_all(root="runs"):
    """Carrega todo `history.json` sob `root`, ordenado por algoritmo/variante/seed."""
    achados = []
    for base, _, arquivos in os.walk(root):
        for nome in arquivos:
            if nome == "history.json":
                achados.append(load(os.path.join(base, nome)))
    achados.sort(key=lambda r: (r.algo, r.variant, r.seed))
    return achados


# ------------------------------------------------------------------- validação
def validate(record: RunRecord, strict_eval=True):
    """Devolve a lista de violações do contrato. Lista vazia = pode entrar na arena.

    Curvas marcadas `comparable=False` só precisam ter identidade, curva e um `caveat`
    explicando por que não competem — o resto do contrato não se aplica a elas.
    """
    p = []

    if not record.algo:
        p.append("`algo` vazio")
    if record.schema_version != SCHEMA_VERSION:
        p.append(f"schema_version {record.schema_version} != {SCHEMA_VERSION}")
    if not record.curve:
        p.append("curva vazia")
    else:
        steps = [pt.get("global_step") for pt in record.curve]
        if any(s is None for s in steps):
            p.append("ponto da curva sem `global_step`")
        elif list(steps) != sorted(steps):
            p.append("`global_step` não é monotônico")

    if not record.comparable:
        if not record.caveat:
            p.append("`comparable=False` exige um `caveat` explicando por quê")
        return p

    # --- daqui para baixo, só para execuções que querem competir
    for chave, esperado in CONTRATO.items():
        obtido = record.env_spec.get(chave, "<ausente>")
        if obtido != esperado:
            p.append(f"env_spec['{chave}'] = {obtido!r}, contrato exige {esperado!r}")

    if not record.final:
        p.append("`final` vazio — falta o resultado do protocolo de avaliação")
    elif strict_eval:
        f = record.final
        if f.get("episodes") != CONTRATO["eval_episodes"]:
            p.append(f"avaliação final com {f.get('episodes')} episódios, "
                     f"contrato exige {CONTRATO['eval_episodes']}")
        if not f.get("completo", True):
            p.append("avaliação final incompleta (bateu `max_steps`)")
        media = f.get("score_mean")
        if media is None:
            p.append("`final.score_mean` ausente")
        elif not (0.0 <= media <= SCORE_PERFEITO):
            p.append(f"score_mean fora da faixa possível: {media}")

    if record.params <= 0:
        p.append("`params` deve ser o número de parâmetros treináveis")
    if not record.net:
        p.append("`net` vazio — a arquitetura é um eixo de comparação")

    return p


def assert_valid(record: RunRecord, **kw):
    problemas = validate(record, **kw)
    if problemas:
        raise ContractViolation(
            f"{record.run_id} viola o contrato:\n  - " + "\n  - ".join(problemas)
        )
    return record


# ---------------------------------------------------------------------- legado
def from_legacy_csv(path, algo="dqn-legacy", variant=None, caveat=None):
    """Converte um CSV de treino do `colab-rl` para o esquema do repositório.

    Os CSVs antigos têm colunas sem nome: `índice, comprimento, passos, loss, reward`.
    O comprimento vira score pela regra `score = comprimento - 3`, e o registro nasce
    `comparable=False` — foi medido em outro ambiente, com outra recompensa e outra
    unidade de tempo. Ele é contexto histórico, não competidor.
    """
    import csv

    linhas = []
    with open(path, newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor, None)
        for row in leitor:
            if len(row) < 5:
                continue
            try:
                ep = int(float(row[0]))
                comprimento = float(row[1])
                passos = float(row[2])
                perda = float(row[3])
                recompensa = float(row[4])
            except ValueError:
                continue
            linhas.append((ep, comprimento, passos, perda, recompensa))

    if not linhas:
        raise ValueError(f"nenhuma linha aproveitável em {path} (cabeçalho: {cabecalho})")

    # Nos CSVs originais o nome do arquivo é sempre `keras_training_data.csv` e quem
    # identifica a variante é a pasta; nos normalizados de `results/legacy/` é o
    # contrário. Aceita os dois.
    if variant is None:
        raiz = os.path.splitext(os.path.basename(path))[0]
        variant = (os.path.basename(os.path.dirname(path))
                   if raiz in ("keras_training_data", "training_data") else raiz)
    curva = [
        {
            "global_step": ep,               # aqui o eixo é episódio, não passo — ver caveat
            "episodes": ep,
            "train_score_mean": comprimento - 3.0,
            "train_length_mean": comprimento,
            "episode_steps": passos,
            "loss": perda,
            "reward": recompensa,
        }
        for ep, comprimento, passos, perda, recompensa in linhas
    ]

    scores = np.array([c["train_score_mean"] for c in curva], dtype=np.float64)
    rec = RunRecord(
        algo=algo,
        variant=variant,
        seed=0,
        net="cnn-legado",
        params=0,
        env_spec={"env": "snake-on-pygame (legado)"},
        curve=curva,
        final={
            "episodes": len(curva),
            "score_mean": float(scores[-100:].mean()),
            "score_max": float(scores.max()),
        },
        comparable=False,
        caveat=(
            caveat
            or "Medido no ambiente antigo (snake-on-pygame): recompensa +comprimento ao "
               "comer, estado ordinal de 1 canal com a cabeça sobrescrita, 5 ações "
               "absolutas, eixo em episódios. Convertido por score = comprimento - 3 "
               "apenas para posicionar a curva; não é comparável com as execuções novas."
        ),
        meta={"fonte": os.path.basename(path)},
    )
    return rec


# ------------------------------------------------------------------ utilidades
def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _ambiente():
    """Carimbo de proveniência: sem isto, um número no gráfico não é rastreável."""
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip()
    except Exception:
        meta["commit"] = "desconhecido"
    for mod in ("tensorflow", "keras"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    return meta
