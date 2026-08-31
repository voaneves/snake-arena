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
    "ORCAMENTO_OFICIAL",
    "SEMENTES_OFICIAIS",
    "ContractViolation",
    "RunRecord",
    "Recorder",
    "validate",
    "save",
    "load",
    "load_all",
    "configuracoes_incompletas",
    "from_legacy_csv",
]

#: 2 — `busca` virou campo de primeira classe; antes era `meta["com_busca"]`,
#: gravado com `skip_validation=True` e portanto fora de qualquer conferência.
#: `load` migra o lugar antigo, então nenhum `history.json` precisou ser reescrito.
SCHEMA_VERSION = 2

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

#: Orçamento oficial, em passos de ambiente. Fica fora do `CONTRATO` porque não descreve o
#: ambiente, mas é igualmente obrigatório: comparar um algoritmo que treinou 5 M passos com
#: outro que treinou 500 mil não mede algoritmo, mede paciência. Validado a partir de
#: `config["total_steps"]`.
ORCAMENTO_OFICIAL = 5_000_000

#: Sementes por configuração. Era convenção escrita no `COMPARABILITY.md` e nada mais —
#: e foi assim que a arena publicou um ACKTR de **uma** semente ao lado de um PPO de três,
#: com a amplitude entre sementes do PPO valendo 19 pontos. Convenção que ninguém confere
#: não é contrato. Ver `configuracoes_incompletas`.
SEMENTES_OFICIAIS = 3

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
        O `stats` de `snakeai.eval.evaluate` para o modelo do **último** passo. É este que
        entra na curva e na arena.
    melhor
        O mesmo `stats`, para o **melhor checkpoint** já visto, mais o `global_step` em que
        ele apareceu. RL profundo não melhora monotonicamente — não há garantia nenhuma
        fora do caso tabular — e uma execução pode terminar pior do que já esteve. Guardar
        os dois separa duas perguntas diferentes: *como o algoritmo terminou* (final) e
        *o melhor que ele produziu* (melhor). A primeira é a da arena; a segunda é a de
        quem vai levar o modelo para o jogo. Ver `docs/COMPARABILITY.md`.
    busca
        Os mesmos `stats`, para o agente medido **com a máquina que ele usa para jogar** —
        a busca em árvore do AlphaZero e do MuZero. Um dicionário `"<checkpoint>_sims<N>"`
        → `stats`, porque um agente pode ser medido em mais de um orçamento de busca e em
        mais de um checkpoint.

        Mora aqui, e não em `meta`, porque **é um resultado**, e resultado se valida. Fica
        numa coluna separada de `final` pelo motivo do `docs/COMPARABILITY.md`: a busca
        gasta dezenas de avaliações de rede por jogada contra uma do PPO, então ela não
        divide eixo com a curva oficial. As três colunas respondem a três perguntas:
        *como o algoritmo terminou* (`final`), *o melhor que ele produziu* (`melhor`) e
        *o que você levaria para jogar* (`busca`).

        Cada entrada carrega `num_simulations`, `checkpoint` e `episodes`. Só as que
        cumprem o protocolo do contrato (1000 episódios, `completo=True`) contam para a
        arena — as demais ficam gravadas, marcadas, como o que são: uma espiada.
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
    melhor: dict = field(default_factory=dict)
    busca: dict = field(default_factory=dict)
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

    @property
    def oficial(self):
        """Pode competir na arena? Comparável **e** sem violação de contrato registrada.

        Separado de `comparable` de propósito: uma execução de fumaça não é uma curva
        histórica. Ela não compete, mas também não vira contexto — simplesmente não
        aparece, e o motivo fica em `meta["contract_violations"]`.
        """
        return self.comparable and not self.meta.get("contract_violations")

    @property
    def busca_oficial(self):
        """As entradas de `busca` que cumprem o protocolo do contrato.

        Mesma régua de `final`: 1000 episódios e `completo=True`. Uma medição de 200
        episódios tem erro padrão ~2× o da oficial e uma que estourou o teto de tempo é
        uma amostra enviesada para episódios **curtos** — justamente os ruins. As duas
        ficam gravadas, e nenhuma das duas entra na arena.
        """
        return {k: st for k, st in (self.busca or {}).items()
                if isinstance(st, dict)
                and st.get("episodes") == CONTRATO["eval_episodes"]
                and st.get("completo", False)}

    def melhor_com_busca(self, checkpoint=None):
        """A melhor entrada oficial de `busca`, ou `None`.

        `checkpoint` filtra por `"last"`/`"best"`; sem ele, o melhor de qualquer um. Não
        há escolha "correta" entre os dois — quem leva o modelo para jogar leva o melhor
        que tem —, e é por isso que o critério fica explícito aqui em vez de implícito
        num max espalhado pelo gráfico.
        """
        itens = [(k, st) for k, st in self.busca_oficial.items()
                 if checkpoint is None or st.get("checkpoint") == checkpoint
                 or (st.get("checkpoint") is None and k.startswith(f"{checkpoint}_"))]
        if not itens:
            return None
        return max((st for _k, st in itens), key=lambda st: st.get("score_mean", -1.0))

    def eval_curve(self):
        """`(passos, scores)` só dos pontos em que a avaliação rodou."""
        pts = self._pontos_de_eval()
        x = np.array([p["global_step"] for p in pts], dtype=np.int64)
        y = np.array([p["eval_score_mean"] for p in pts], dtype=np.float64)
        return x, y

    def _pontos_de_eval(self):
        return [p for p in self.curve if p.get("eval_score_mean") is not None]

    def eval_curve_tempo(self):
        """`(horas, scores)` — a mesma curva no eixo de **custo**, não de dados.

        O eixo oficial da arena são passos de ambiente, que igualam os *dados vistos* e
        escondem o *esforço gasto*: o AlphaZero faz busca em árvore a cada passo e custa
        ordens de grandeza mais que o DQN para chegar ao mesmo x. Este eixo mostra a outra
        metade.

        O `wall_s` inclui as avaliações periódicas, e isso é proposital: elas são custo
        real de quem roda. Mas veja `mesmo_hardware` — comparar tempo entre execuções
        feitas em GPUs diferentes não significa nada.
        """
        pts = self._pontos_de_eval()
        h = np.array([p.get("wall_s", np.nan) for p in pts], dtype=np.float64) / 3600.0
        y = np.array([p["eval_score_mean"] for p in pts], dtype=np.float64)
        return h, y

    def passos_ate(self, limiar):
        """Primeiro passo **medido** em que a avaliação atingiu `limiar`. `None` se nunca.

        Sem interpolação, de propósito: a resolução é a cadência de avaliação
        (`eval_every_steps`), e interpolar inventaria uma precisão que a amostragem não
        tem. O número devolvido é um passo em que a medição de fato aconteceu.
        """
        x, y = self.eval_curve()
        atingiu = np.nonzero(y >= limiar)[0]
        return int(x[atingiu[0]]) if atingiu.size else None

    @property
    def hardware(self):
        """Identidade do que rodou isto, para o eixo de tempo saber quando calar a boca."""
        gpus = self.meta.get("gpus") or []
        return f"{self.meta.get('plataforma', '?')}/{','.join(gpus) or 'cpu'}"


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

    def finish(self, final_stats, comparable=True, caveat="", melhor_stats=None):
        self.record.final = {k: _jsonable(v) for k, v in dict(final_stats).items()}
        if melhor_stats is not None:
            self.record.melhor = {k: _jsonable(v) for k, v in dict(melhor_stats).items()}
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
    # v1 → v2: a coluna com busca morava em `meta["com_busca"]`, gravada com
    # `skip_validation=True`. Migrar na leitura em vez de reescrever os arquivos mantém
    # os registros já publicados byte a byte iguais — a assinatura de um `history.json`
    # é parte do que torna uma execução citável. O lugar antigo continua legível; o novo
    # é o que a arena consulta.
    if not d.get("busca"):
        d["busca"] = dict(d.get("meta", {}).get("com_busca") or {})
    return RunRecord(**d, schema_version=SCHEMA_VERSION)


def configuracoes_incompletas(registros, minimo=SEMENTES_OFICIAIS):
    """As configurações `(algo, variant)` com menos de `minimo` sementes distintas.

    É uma propriedade do **conjunto**, não de uma execução — por isso não cabe em
    `validate`, que olha uma por vez. A arena chama isto para não publicar uma linha de
    uma semente com a mesma tipografia de uma de três.
    """
    grupos = {}
    for r in registros:
        grupos.setdefault((r.algo, r.variant), set()).add(r.seed)
    return [{"algo": a, "variant": v, "sementes": len(s), "faltam": minimo - len(s)}
            for (a, v), s in sorted(grupos.items()) if len(s) < minimo]


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
        # As chaves de causa de fim entraram junto com a correção do protocolo — o mesmo
        # commit que passou a contar a maçã final do episódio vencedor. Um registro sem
        # elas foi medido com a régua **anterior**, e `episodes`/`completo` continuam
        # iguais nos dois casos: é a única marca que distingue. Ver
        # `docs/ANTES_DO_ARTIGO.md`.
        faltando = [k for k in ("fim_fome", "fim_colisao", "fim_tabuleiro_cheio")
                    if k not in f]
        if faltando:
            p.append(f"`final` sem {', '.join(faltando)} — medido com um protocolo "
                     "anterior ao atual; remeça com o `evaluate` desta versão")
        media = f.get("score_mean")
        if media is None:
            p.append("`final.score_mean` ausente")
        elif not (0.0 <= media <= SCORE_PERFEITO):
            p.append(f"score_mean fora da faixa possível: {media}")

    # A coluna com busca é opcional — a maioria dos algoritmos não tem máquina além da
    # rede. Mas uma entrada que **existe** tem de dizer o que é: sem `num_simulations` o
    # número não é interpretável, e sem `checkpoint` não se sabe se mediu o `last` ou o
    # `best`. O tamanho da amostra **não** é violação: uma espiada de 200 episódios é
    # legítima, só não entra na arena (ver `busca_oficial`).
    for chave, st in (record.busca or {}).items():
        if not isinstance(st, dict):
            p.append(f"`busca['{chave}']` não é um dicionário de stats")
            continue
        if st.get("num_simulations") is None:
            p.append(f"`busca['{chave}']` sem `num_simulations` — o número só significa "
                     "alguma coisa junto com o orçamento de busca que o produziu")
        media = st.get("score_mean")
        if media is None:
            p.append(f"`busca['{chave}'].score_mean` ausente")
        elif not (0.0 <= media <= SCORE_PERFEITO):
            p.append(f"`busca['{chave}'].score_mean` fora da faixa possível: {media}")

    orcamento = record.config.get("total_steps")
    if orcamento is None:
        p.append("`config['total_steps']` ausente — o orçamento é parte do contrato")
    elif int(orcamento) != ORCAMENTO_OFICIAL:
        p.append(f"orçamento de {int(orcamento):,} passos; o contrato exige "
                 f"{ORCAMENTO_OFICIAL:,}. Comparar treinos de tamanhos diferentes mede "
                 "paciência, não algoritmo")

    # O `config` diz o orçamento **pretendido**; a curva diz o que foi **gasto**. Conferir
    # só o primeiro deixava passar uma execução interrompida na metade com o `config`
    # intacto — `train(ate_passos=...)` faz exatamente isso, e uma sessão do Colab caindo
    # também. Ver `docs/REVISAO_ALGORITMOS.md` §1.3.
    gasto = max((int(ponto.get("global_step", 0)) for ponto in record.curve), default=0)
    if gasto < ORCAMENTO_OFICIAL:
        p.append(f"a curva vai até {gasto:,} passos, abaixo dos {ORCAMENTO_OFICIAL:,} do "
                 "contrato — o orçamento declarado em `config` não é o que foi gasto")

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

    # No Colab e no Kaggle não existe clone git: o `commit` sai "desconhecido" e a curva
    # fica sem procedência — que é justamente onde quase todas as execuções deste
    # repositório nascem. O gerador de notebooks injeta `ASSINATURA_PACOTE` no bloco de
    # código gerado, e ela identifica o pacote inteiro (é o hash do fonte embutido). Como
    # o notebook é um namespace só, esta busca a encontra lá e não a encontra aqui.
    assinatura = globals().get("ASSINATURA_PACOTE")
    if assinatura:
        meta["assinatura_pacote"] = str(assinatura)
    for mod in ("tensorflow", "keras"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    return meta
