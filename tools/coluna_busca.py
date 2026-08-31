"""A coluna "com busca" de uma execução que já terminou — sem retreinar nada.

Por que ela é uma coluna, e não uma linha na curva
--------------------------------------------------
A curva oficial mede a **política pura**, greedy, sem nenhuma ajuda. É o que torna as
curvas comparáveis: a busca gasta `num_simulations + 1` avaliações de rede **por jogada**
contra 1 do PPO, e somar as duas no mesmo eixo diria "o AlphaZero ganha do PPO" quando o
que houve foi gastar 33× mais computação na hora de decidir. Mesma regra que manda o filtro
de flood-fill para coluna própria.

Reportar, porém, é obrigação: um algoritmo que existe para buscar, medido só sem buscar, é
meia medição — e a busca é o que se levaria para jogar de verdade, já que em Snake o
simulador está disponível na hora de agir.

Este script existe para o caso em que o treino acabou e a coluna não foi medida. Ele
reconstrói o agente com a configuração **da execução** (e não com os padrões de hoje:
`fpu`, `q_normalizado`, `c_puct` e `gamma` mudam a busca, e medir com outra configuração
mediria outro agente), carrega o modelo salvo, roda o protocolo oficial e **grava de volta** no campo `busca` do
`history.json` — irmão de `final` e `melhor` desde o schema 2, e não um canto de `meta`,
porque o que mora em `meta` não passa por `validate()` e isto é um resultado.

Só entradas com os 1.000 episódios do contrato e `completo=True` entram na coluna *com
busca* da arena; as demais ficam gravadas e marcadas como o que são.

Custo
-----
Medido num container de 2 núcleos, `resnet_small`, 64 ambientes, agente cujos episódios
duram ~930 jogadas:

=======  ===========  ==========  ===========
 sims     jogadas/s    200 epis.   1000 epis.
=======  ===========  ==========  ===========
      8         5,19       12 min       48 min
     16         2,72       23 min       91 min
     32         1,43       43 min      173 min
=======  ===========  ==========  ===========

Num desktop de 6–8 núcleos divida por ~4–6. O gargalo é metade rede e metade o laço de
árvore em Python, que é single-thread — por isso a diferença para uma GPU pequena é bem
menor do que se imaginaria: o lote é 64 num modelo de 180 mil parâmetros.

Uso::

    python tools/coluna_busca.py runs/alphazero/sims32/seed0
    python tools/coluna_busca.py runs/alphazero/sims32/seed0 --episodios 1000 --sims 32
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import keras                                            # noqa: E402
import numpy as np                                      # noqa: E402

import snakeai.agents as agentes                        # noqa: E402


def _classe_e_config(algo):
    """`("alphazero") → (AlphaZero, AlphaZeroConfig)`, pelo nome que o registro guardou."""
    for nome in dir(agentes):
        obj = getattr(agentes, nome)
        if isinstance(obj, type) and getattr(obj, "algo", None) == algo:
            cfg = getattr(agentes, f"{nome}Config", None)
            if cfg is None:
                raise SystemExit(f"achei o agente {nome} mas não o {nome}Config")
            return obj, cfg
    raise SystemExit(f"nenhum agente com algo={algo!r} em snakeai.agents")


def _carrega_pesos(ag, caminho):
    """Pesos no lugar, preservando a arquitetura já construída.

    Tem que ser um agente **novo**: as `tf.function` da busca capturam as variáveis do
    modelo no primeiro traço, então trocar o modelo de um agente já usado é silenciosamente
    ignorado — a medição sairia com os pesos antigos e ninguém notaria.
    """
    try:
        ag.model.load_weights(caminho)
    except Exception:                                   # noqa: BLE001
        ag.model.set_weights(keras.models.load_model(caminho).get_weights())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pasta", help="a pasta da execução (a que tem history.json e modelos/)")
    p.add_argument("--episodios", type=int, default=200,
                   help="1000 é o contrato; 200 dá a ordem de grandeza em 1/5 do tempo")
    p.add_argument("--sims", type=int, nargs="+", default=None,
                   help="orçamentos de busca a medir (padrão: o `sims_avaliacao` do config)")
    p.add_argument("--minutos", type=float, default=60.0,
                   help="teto por orçamento; ao estourar, a amostra volta `completo=False` "
                        "e fica fora da arena. Use 0 para não ter teto — o certo num "
                        "terminal, onde Ctrl-C existe; o teto é para notebook, onde um "
                        "`Run all` sem limite trava a sessão inteira")
    p.add_argument("--ambientes", type=int, default=64)
    p.add_argument("--modelo", default="last", choices=("last", "best"))
    p.add_argument("--checkpoints", default=None,
                   help="pasta de checkpoints do treino. Necessária para agentes cujo "
                        "`modelos/<tag>.keras` não contém tudo que a busca usa — o MuZero "
                        "guarda ali só `h`+`f` compostos, e a árvore percorre `g`")
    p.add_argument("--seco", action="store_true", help="não grava no history.json")
    a = p.parse_args(argv)

    caminho_hist = os.path.join(a.pasta, "history.json")
    if not os.path.exists(caminho_hist):
        raise SystemExit(f"não achei {caminho_hist}")
    with open(caminho_hist, encoding="utf-8") as f:
        rec = json.load(f)

    Agente, Config = _classe_e_config(rec["algo"])
    campos = {c.name for c in dataclasses.fields(Config)}
    salvo = {k: v for k, v in rec["config"].items() if k in campos}
    sumiram = sorted(set(rec["config"]) - campos)
    if sumiram:
        print("campos do registro que não existem mais no config (ignorados):", sumiram)
    salvo.update(ckpt_dir=a.checkpoints or os.path.join(a.pasta, "modelos"),
                 runs_dir=os.path.join(a.pasta, "_tmp"))
    ag = Agente(Config(**salvo))

    if a.checkpoints:
        if not ag.retomar(a.modelo):
            raise SystemExit(f"não consegui retomar `{a.modelo}` de {a.checkpoints}")
    else:
        keras_path = os.path.join(a.pasta, "modelos", f"{a.modelo}.keras")
        if not os.path.exists(keras_path):
            raise SystemExit(f"não achei {keras_path}")
        if not hasattr(ag, "avaliar_com_busca"):
            raise SystemExit(f"{rec['algo']} não busca na hora de agir — a coluna não se aplica")
        if rec["algo"] == "muzero":
            raise SystemExit(
                "o `modelos/<tag>.keras` do MuZero é só `h`+`f` compostos, e a árvore "
                "percorre `g` — a busca não dá para reconstruir dele. Passe "
                "`--checkpoints <pasta>` apontando para onde estão "
                "`muzero_<tag>_h.keras`, `_g` e `_f`.")
        _carrega_pesos(ag, keras_path)
        ag.on_model_reloaded()

    sims = a.sims or [ag.cfg.sims_avaliacao]
    pura = rec.get("final", {}).get("score_mean")
    print(f"execução: {rec['algo']}/{rec['variant']}/seed{rec['seed']}  ·  modelo "
          f"{a.modelo}  ·  {rec.get('params', 0):,} params")
    if pura is not None:
        print(f"rede pura registrada (a curva oficial): {pura:.2f}")
    teto = None if a.minutos <= 0 else a.minutos * 60
    print(f"medindo com busca — {a.episodios} episódios por orçamento, "
          + ("sem teto de tempo" if teto is None else f"teto {a.minutos:.0f} min")
          + f", {a.ambientes} ambientes", flush=True)

    # lê dos dois lugares: `busca` é o campo atual, `meta["com_busca"]` é onde os
    # registros anteriores ao schema 2 guardavam a mesma coisa
    medidas = dict(rec.get("busca") or rec.get("meta", {}).get("com_busca", {}))
    for s in sims:
        t0 = time.time()
        st = ag.avaliar_com_busca(episodes=a.episodios, num_simulations=s,
                                  num_envs=a.ambientes,
                                  max_segundos=teto, verbose=True)
        medidas[f"{a.modelo}_sims{s}"] = {**st, "checkpoint": a.modelo}
        aviso = "" if st["completo"] else "   ATENCAO: parcial, completo=False, fora da arena"
        print(f"  {s:>3} sims: score {st['score_mean']:6.2f} · cheio {st['win_rate']:5.1%} "
              f"· fome {st['fim_fome']:5.1%} · {st['episodes']} episódios · "
              f"{(time.time() - t0) / 60:.1f} min{aviso}", flush=True)

    if pura:
        print()
        for nome, st in medidas.items():
            s_ = st["score_mean"]
            linha = f"{nome:>16}: busca {s_:>6.2f}  ·  rede pura {pura:>6.2f}"
            if pura > 0.5 and s_ > pura:
                linha += f"  ·  {s_ / pura:.2f}x  ·  a rede captura {pura / s_:.0%} da busca"
            elif pura > 0.5:
                linha += f"  ·  {s_ / pura:.2f}x  ·  a rede está À FRENTE da busca aqui"
            print(linha)

    if a.seco:
        print("\n(--seco: nada gravado)")
        return
    rec["busca"] = medidas
    # o lugar antigo sai junto, senão o registro passa a afirmar a mesma coisa em dois
    # lugares que podem divergir na próxima medição
    rec.get("meta", {}).pop("com_busca", None)
    with open(caminho_hist, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("\ngravado no campo `busca` de", caminho_hist)
    oficiais = [k for k, v in medidas.items()
                if v.get("episodes") == 1000 and v.get("completo")]
    print("entram na arena: " + (", ".join(oficiais) if oficiais else
                                 "nenhuma (o contrato pede 1000 episodios completos)"))


if __name__ == "__main__":
    main()
