"""De onde vem o estouro da KL do ACKTR — e a resposta não custa 5 M passos.

A pergunta
----------
`escala_kl` devolve `η = √(2·kl_max / Δᵀ∇)`: o passo tal que **uma** atualização `ηΔ`
induz `kl_max`. A KL medida sai de 4,4× a 12,4× disso. O `docs/REVISAO_ALGORITMOS.md` §2
atribui a diferença à Fisher aproximada — e essa atribuição é a premissa do ACEKTR.

Há dois outros suspeitos no mesmo lugar, e nenhum deles é a Fisher:

* **o momento.** `η` é atribuído como `learning_rate` de um `SGD(momentum=0.9,
  nesterov=True)`. Com momento, o deslocamento em regime é até `ηΔ/(1−μ) = 10·ηΔ`, e a KL
  vai com o **quadrado** do passo. O `baselines` original faz
  `MomentumOptimizer(lr·(1−momentum), momentum)` justamente para cancelar isto;
* **o `clipnorm`.** O `max_grad_norm = 0,5` herdado do PPO é aplicado pelo Keras **por
  variável, dentro do `apply_gradients`** — sobre a direção **já pré-condicionada**. No
  `baselines` o clip nunca toca a direção natural.

Por que o experimento é barato
------------------------------
A KL é medida **por atualização**. Não é preciso um treino inteiro: algumas centenas de
atualizações já dão a mediana da razão `KL_medida / KL_pedida`, e o §2 registra que o
estouro é **maior no começo** — ou seja, o regime que este script mede é o pior caso.

`kl_calibrado` fica **desligado** em todos os braços, e isso não é detalhe: ligado, ele
mede a razão e pede `kl_max/c`, de modo que a KL entregue converge para o alvo **qualquer
que seja a causa**. Medir com ele ligado responderia sempre "está calibrado".

Como ler
--------
`razao` é `KL_medida / KL_pedida`. O que cada braço prevê, se a hipótese dele for a certa:

O braço que decide é `sem_momento_sem_clip`: sem os dois, o que sobra é a fórmula da KL
contra a KL da política de verdade. Se ele ficar longe de 1×, o erro é da Fisher.

**Já foi medido** (512×5, `resnet_small`, 60 atualizações, GPU) e o resultado é um 2×2:

===============  ==================  ====================
                 `clipnorm` ligado   `clipnorm` desligado
===============  ==================  ====================
momento ligado   7,4×                15,3×
momento desligado 3,8×               **12,9×**
===============  ==================  ====================

Tirar o clip **piora** (×2,07 e ×3,39): ele não era suspeito, era um freio acidental que
encurtava a direção e derrubava a KL junto. Tirar o momento ajuda pouco (×0,51 e ×0,84).
E com os dois desligados sobram **12,9×** — o erro da Fisher aproximada, que é o que a §2
sempre disse. As duas hipóteses deste script foram falsificadas, e ele fica como o
instrumento que as falsificou.

Vale repetir com `--iters 300`: as 40 atualizações que entram na mediana ainda estão no
regime frio do K-FAC (o `baselines` usa `cold_iter=100`, e este repositório não tem cold
start), e a §2 registra que o estouro é maior no começo.

Uso::

    python tools/diag_acktr_kl.py                 # ~2 min por braço em CPU
    python tools/diag_acktr_kl.py --iters 400 --net resnet_small
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from snakeai.agents import ACKTR                             # noqa: E402
from snakeai.agents.acktr import ACKTRConfig                 # noqa: E402

#: `kl_calibrado=False` em todos, senão a realimentação esconde a causa.
BRACOS = {
    "controle":              {},
    "sem_momento":           {"momento": 0.0},
    "momento_descontado":    {"descontar_momento": True},
    "sem_clip":              {"max_grad_norm": 0.0},
    "sem_momento_sem_clip":  {"momento": 0.0, "max_grad_norm": 0.0},
}


def mede(nome, extra, iters, net, seed, envs=0, rollout=0):
    forma = {}
    if envs:
        forma["num_envs"] = envs
    if rollout:
        forma["rollout"] = rollout
    cfg = ACKTRConfig(net=net, seed=seed, total_steps=10**9, eval_every_steps=10**9,
                      log_every_steps=10**9, salvar_gif=False, salvar_grafico=False,
                      kl_calibrado=False, **forma, **extra)
    ag = ACKTR(cfg)
    razoes, kls = [], []
    for i in range(iters):
        st = ag.iterate()
        if st is None or st.get("kl") is None:
            continue
        # as primeiras atualizações rodam com a média móvel dos fatores do K-FAC ainda
        # crua; o `baselines` tem um `cold_iter=100` para isso e nós não temos (§2.36),
        # então as 20 primeiras entram no relato à parte em vez de na mediana
        alvo = st.get("kl_alvo_efetivo") or cfg.kl_max
        razoes.append(st["kl"] / max(alvo, 1e-12))
        kls.append(st["kl"])
    r = np.array(razoes[20:] or razoes, dtype=np.float64)
    frio = np.array(razoes[:20], dtype=np.float64)
    return {"braco": nome, "n": int(r.size), "kl_alvo": cfg.kl_max,
            "razao_mediana": float(np.median(r)), "razao_p90": float(np.quantile(r, 0.9)),
            "razao_frio": float(np.median(frio)) if frio.size else None,
            "kl_mediana": float(np.median(kls[20:] or kls)), **extra}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--net", default="resnet_small")
    ap.add_argument("--seed", type=int, default=0)
    #: A forma do contrato (512x5) e o padrao; encolher serve para pegar a DIRECAO do
    #: efeito numa CPU, e o numero que vale e o da forma oficial, numa GPU.
    ap.add_argument("--envs", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--bracos", default="", help="lista separada por vírgula")
    a = ap.parse_args(argv)

    escolhidos = ([b.strip() for b in a.bracos.split(",") if b.strip()]
                  if a.bracos else list(BRACOS))
    linhas = []
    forma = f"{a.envs or 512}x{a.rollout or 5}"
    aviso = "" if not (a.envs or a.rollout) else "  (REDUZIDA: le a direcao, nao o numero)"
    print(f"kl_calibrado=False em todos · {a.iters} iteracoes · net={a.net} · "
          f"forma {forma}{aviso}\n")
    print(f"{'braco':>22} {'KL mediana':>11} {'razao':>8} {'p90':>8} {'frias':>8}",
          flush=True)
    for nome in escolhidos:
        l = mede(nome, BRACOS[nome], a.iters, a.net, a.seed, a.envs, a.rollout)
        linhas.append(l)
        frio = f"{l['razao_frio']:.1f}x" if l["razao_frio"] is not None else "—"
        print(f"{nome:>22} {l['kl_mediana']:>11.5f} {l['razao_mediana']:>7.1f}x "
              f"{l['razao_p90']:>7.1f}x {frio:>8}", flush=True)

    destino = os.path.join(RAIZ, "docs", "diag_acktr_kl.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(linhas, f, indent=1, ensure_ascii=False)
    print("\ngravado em", destino)
    base = next((l for l in linhas if l["braco"] == "controle"), None)
    if base and base["razao_mediana"] < 1.5:
        print(f"\nMEDICAO VAZIA: o controle deu {base['razao_mediana']:.1f}x e nao ha "
              "estouro a explicar.\nCom uma forma reduzida o fenomeno some (15,3x com "
              "512 ambientes, 0,9x com 64). Rode na forma do contrato.")
    elif base:
        for l in linhas:
            if l["braco"] == "controle":
                continue
            # A estatistica certa e a fracao do EXCESSO removida, e nao a razao das
            # razoes: 1,0 e o alvo, entao o que um braco explica e
            # `((controle - 1) - (braco - 1)) / (controle - 1)`. Ler "2,0x contra 15,3x"
            # como "reduziu para 13%" subestima o efeito: o excesso caiu 93%.
            exc_base = base["razao_mediana"] - 1.0
            exc = max(l["razao_mediana"] - 1.0, 0.0)
            frac = 1.0 - exc / exc_base
            veredito = ("EXPLICA quase tudo" if frac > 0.85
                        else "explica boa parte" if frac > 0.5
                        else "contribui pouco" if frac > 0.15 else "nao e a causa")
            print(f"  {l['braco']:>22}: remove {frac:5.1%} do excesso  ->  {veredito}")


if __name__ == "__main__":
    main()
