"""Por que a busca do AlphaZero degenera quando o valor aprendido é positivo.

O que este script mede
----------------------
Três tabelas, todas em NumPy puro — não precisa de GPU nem de rede treinada, porque a
pergunta não é sobre a rede. A avaliação da folha é uma heurística (distância de Manhattan
até a comida) e o **ranking de estados que ela produz é o mesmo em todas as variantes**.
O que muda é o deslocamento, a escala do Q dentro do PUCT e o orçamento de busca.

1. `deslocamento` — a mesma heurística somada de uma constante. Nenhuma decisão *deveria*
   mudar; com o PUCT de hoje, muda tudo. É o experimento que motivou `93_alphazero_ablacoes`.
2. `profundidade` — o que `num_simulations` de fato compra, em plies da variação principal
   e em concentração da política de visitas.
3. `dirichlet` — a geometria do ruído da raiz com **3** ações, que não é a de 35 nem a de 250.

Uso::

    python tools/diag_busca.py                  # as três
    python tools/diag_busca.py deslocamento     # uma só
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from snakeai.env.vec_snake import VecSnake            # noqa: E402
from snakeai.search import MCTS                       # noqa: E402

TABULEIRO = 10
GAMA = 0.997
PASSOS_POR_MACA = 12      # medido; ver docs/REVISAO_ALGORITMOS.md §2.25


# --------------------------------------------------------------------- avaliadores
def heuristica(deslocamento=0.0):
    """`(priors uniformes, valor)` com valor = −distância de Manhattan até a comida.

    `deslocamento` soma uma constante. Ordena os estados exatamente igual — e é a diferença
    entre a heurística com que a busca foi medida no docstring do `mcts.py` (negativa) e o
    que uma cabeça de valor treinada neste jogo produz (positiva, e grande).
    """
    def fn(obs, mask):
        obs = np.asarray(obs, dtype=np.float32)
        n, b = obs.shape[0], obs.shape[1]
        cab = obs[..., 1].reshape(n, -1).argmax(1)
        com = obs[..., 3].reshape(n, -1).argmax(1)
        d = np.abs(cab // b - com // b) + np.abs(cab % b - com % b)
        p = np.asarray(mask, dtype=np.float64)
        p /= p.sum(1, keepdims=True)
        return p, (-d / (2.0 * b) + deslocamento).astype(np.float32)
    return fn


def joga(avaliador, sims, passos, n_envs=16, seed=7, **kw):
    """Roda a busca sozinha e devolve score, causas de fim e forma da política de visitas."""
    env = VecSnake(n_envs, TABULEIRO, rng=np.random.default_rng(seed))
    obs, mask = env.reset()
    busca = MCTS(avaliador, board_size=TABULEIRO, gamma=GAMA, num_simulations=sims,
                 c_puct=1.5, starve_base=env.starve_base,
                 rng=np.random.default_rng(seed), **kw)
    prof, pv, ent, comp, scores = [], [], [], [], []
    causas = {"colisao": 0, "fome": 0, "vitoria": 0}
    for _ in range(passos):
        visitas, _ = busca.run(env.get_state(), mask, obs)
        for r in busca._ultimas_raizes:
            prof.append(_profundidade(r))
            pv.append(_profundidade_pv(r))
        p = visitas / np.maximum(visitas.sum(1, keepdims=True), 1e-12)
        ent.extend((-(p * np.log(np.maximum(p, 1e-12))).sum(1) / math.log(3)).tolist())
        comp.extend(env.length.tolist())
        obs, mask, _, _, info = env.step(visitas.argmax(1).astype(np.int32))
        causas["colisao"] += int(info["deaths"])
        causas["fome"] += int(info["starved"])
        causas["vitoria"] += int(info["wins"])
        scores.extend(info["scores"].tolist())
    return {
        "sims": sims, "episodios": len(scores),
        "score": round(float(np.mean(scores)), 2) if scores else None,
        "prof_arvore": round(float(np.mean(prof)), 2),
        "prof_pv": round(float(np.mean(pv)), 2),
        "prof_pv_p95": float(np.percentile(pv, 95)),
        "entropia_visitas": round(float(np.mean(ent)), 3),
        "comprimento_medio": round(float(np.mean(comp)), 1),
        "causas": causas,
    }


def _profundidade(no, d=0):
    if not no.filhos:
        return d
    return max((_profundidade(f, d + 1) for f in no.filhos.values() if f.expandido),
               default=d)


def _profundidade_pv(raiz):
    """Plies do ramo mais visitado — a profundidade que a busca de fato *usa*."""
    d, no = 0, raiz
    while no.filhos:
        f = max(no.filhos.values(), key=lambda c: c.visitas)
        if f.visitas == 0 or not f.expandido:
            break
        no, d = f, d + 1
    return d


# ------------------------------------------------------------------------ tabelas
def tabela_deslocamento(sims=8, passos=250):
    """O experimento central: só o deslocamento do valor muda, e o jogo inteiro muda."""
    linhas = []
    for nome, desl in (("negativo", 0.0), ("positivo", 1.0)):
        for fpu, qn in (("zero", False), ("pai", False), ("zero", True), ("pai", True)):
            r = joga(heuristica(desl), sims, passos, fpu=fpu, q_normalizado=qn)
            linhas.append({"valor": nome, "fpu": fpu, "q_normalizado": qn,
                           "score": r["score"], "causas": r["causas"]})
    return linhas


def tabela_profundidade(orcamentos=(8, 16, 32, 64, 128)):
    """O que `num_simulations` compra. Menos horizonte do que parece, mais concentração."""
    # com o conserto ligado, senão a busca está degenerada e a profundidade não significa
    # nada — é sempre o mesmo ramo
    passos = {8: 300, 16: 300, 32: 200, 64: 120, 128: 60}
    return [{k: v for k, v in joga(heuristica(0.0), s, passos.get(s, 60),
                                   q_normalizado=True).items() if k != "causas"}
            for s in orcamentos]


def tabela_dirichlet(alphas=(0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0, 10 / 3), n=200_000):
    """A geometria do ruído da raiz com 3 ações.

    A heurística do paper é α ∝ 1/(ações legais), calibrada em ~10/n: Go usa 0,03 com ~250
    ações (7,5), Xadrez 0,3 com ~35 (10,5), Shogi 0,15 com ~92 (13,8). Para 3 ações isso dá
    **α ≈ 3,3** — o oposto de baixar para 0,2.
    """
    rng = np.random.default_rng(0)
    saida = []
    for a in alphas:
        d = rng.dirichlet([a] * 3, size=n)
        mx = d.max(1)
        saida.append({
            "alpha": round(float(a), 3),
            "max_medio": round(float(mx.mean()), 3),
            "p_max_maior_08": round(float((mx > 0.8).mean()), 3),
            "p_max_maior_09": round(float((mx > 0.9).mean()), 3),
            "entropia_norm": round(float(
                (-(d * np.log(np.maximum(d, 1e-12))).sum(1) / math.log(3)).mean()), 3),
        })
    return saida


def tabela_horizonte(gamas=(0.98, 0.985, 0.99, 0.995, 0.997)):
    """O tamanho do buraco que o FPU precisa atravessar, por γ.

    Com recompensa `+1` por maçã a cada ~`PASSOS_POR_MACA` passos e cabeça de valor linear,
    o ponto fixo do valor é `1/(1 − γ**k)`. O bônus do PUCT vale no máximo
    `c_puct · P · √N` — com `c_puct = 1,5`, prior uniforme em 3 ações e 32 simulações, 2,8.
    """
    return [{"gamma": g, "horizonte": round(1 / (1 - g), 1),
             "valor_no_ponto_fixo": round(1 / (1 - g ** PASSOS_POR_MACA), 1),
             "teto_do_bonus_puct_32_sims": round(1.5 * (1 / 3) * math.sqrt(33), 2)}
            for g in gamas]


TABELAS = {
    "deslocamento": tabela_deslocamento,
    "profundidade": tabela_profundidade,
    "dirichlet": tabela_dirichlet,
    "horizonte": tabela_horizonte,
}


def main(argv=None):
    """`diag_busca.py [tabela[=args] ...]`.

    `profundidade=8,16,32` roda só esses orçamentos — útil porque a tabela inteira leva
    alguns minutos e as linhas caras são as de 64 e 128 simulações. O JSON é **mesclado**
    com o que já existe, para que rodar em pedaços não apague o resto.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    pedidas = argv or list(TABELAS)
    destino = os.path.join(RAIZ, "docs", "diag_busca.json")
    saida = {}
    if os.path.exists(destino):
        with open(destino, encoding="utf-8") as f:
            saida = json.load(f)
    for pedido in pedidas:
        nome, _, args = pedido.partition("=")
        if nome not in TABELAS:
            raise SystemExit(f"tabela desconhecida: {nome} (use {', '.join(TABELAS)})")
        print(f"== {nome} ==", flush=True)
        novas = (TABELAS[nome](tuple(float(x) if "." in x else int(x)
                                     for x in args.split(",")))
                 if args else TABELAS[nome]())
        # mesclar por orçamento, não sobrescrever: `profundidade=64` complementa a tabela
        antigas = {json.dumps(l, sort_keys=True): l for l in saida.get(nome, [])}
        chave = "sims" if nome == "profundidade" else ("alpha" if nome == "dirichlet"
                                                       else None)
        if chave:
            por_chave = {l[chave]: l for l in saida.get(nome, []) if chave in l}
            por_chave.update({l[chave]: l for l in novas})
            saida[nome] = [por_chave[k] for k in sorted(por_chave)]
        else:
            saida[nome] = novas if novas else list(antigas.values())
        print(json.dumps(saida[nome], indent=1, ensure_ascii=False), flush=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=1, ensure_ascii=False)
    print("gravado em", destino)


if __name__ == "__main__":
    main()
