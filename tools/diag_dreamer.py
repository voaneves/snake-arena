"""Por que o DreamerV3 afunda abaixo do piso aleatório — três hipóteses, com número.

O sintoma: `train_score_mean` cai de 1,27 para 0,63 (piso aleatório: 1,21) e a avaliação
greedy chega a 0,02. Não é lentidão, é a política ficando **pior que aleatória**.

As hipóteses que este script separa
-----------------------------------
**H1 — o sonho não tem fome.** A observação do contrato tem 5 canais (corpo, cabeça,
decaimento, comida, comprimento) e **nenhum deles é o contador de fome**. O limite é
`100 + 2·comprimento` passos sem comer, e `seq_len=32`: nem a observação nem a recorrência
podem representar quão perto da inanição a cobra está. Se for verdade, a cabeça de
continuação prevê morte por colisão e **não** prevê truncamento por fome — e dentro do
sonho "andar em círculo para sempre" tem retorno 0 sem punição nenhuma, enquanto ir buscar
comida arrisca −1. Zero ganha de negativo. E é absorvente: chegando lá, nenhum gradiente
tira o ator de lá, porque no sonho ele já é ótimo.

**H2 — colapso de entropia.** O ator vira determinístico e uma política determinística
mascarada passa fome ~100% das vezes (já medido). Assinatura: `ent_sonho` → 0.

**H3 — a cabeça de recompensa está desalinhada em um passo.** `memoria.add(obs_ant, acao,
r, ...)` guarda em `t` a recompensa **da ação tomada em** `t`, e a cabeça só recebe o
estado, sem a ação. Então ela é obrigada a prever a média sobre ações — e a variância da
recompensa entre as 3 ações no mesmo estado é o erro irredutível que esse alinhamento
impõe. O DreamerV3 usa a outra convenção: `r_t` é a recompensa **recebida ao chegar** em
`s_t`, que é determinística dado `h_t` (que já contém `a_{t-1}`).

O que cada medição decide
-------------------------
=====================================  =================================================
medição                                 confirma
=====================================  =================================================
`p_cont` em terminais de fome           H1, se ficar perto de 1 enquanto os de
                                        colisão ficam perto de 0
fração de episódios por fome ao longo   H1, se subir para ~100% junto com a queda
do treino                                do score
`ent_sonho` ao longo do treino          H2, se cair para perto de 0
`Var_a(r | s)` medida no ambiente        H3: é o erro que a cabeça não pode evitar
=====================================  =================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snakeai.agents.dreamerv3 import DreamerV3, DreamerV3Config
from snakeai.env.vec_snake import N_ACTIONS, VecSnake


# ------------------------------------------------------------------------- H3
def variancia_da_recompensa_entre_acoes(n_envs=256, passos=300, seed=0):
    """`Var_a(r | s)` medida no ambiente de verdade, restaurando o estado.

    Snake é determinístico, então isto não é estimativa: para cada estado visitado,
    aplicamos cada uma das 3 ações a partir do **mesmo** estado e olhamos as 3 recompensas.
    A variância entre elas é exatamente o que uma cabeça que só vê `s` não pode prever.
    """
    rng = np.random.default_rng(seed)
    env = VecSnake(n_envs, 10, rng=rng)
    env.reset()

    variancias, diferentes, total = [], 0, 0
    for _ in range(passos):
        estado = env.get_state()
        recompensas = []
        for a in range(N_ACTIONS):
            env.set_state(estado)
            _, _, r, _, _ = env.step(np.full(n_envs, a, np.int32))
            recompensas.append(r.copy())
        R = np.stack(recompensas, axis=1)                      # (n, 3)
        variancias.append(R.var(axis=1))
        diferentes += int((R.max(axis=1) != R.min(axis=1)).sum())
        total += n_envs

        env.set_state(estado)
        env.step(rng.integers(0, N_ACTIONS, n_envs).astype(np.int32))

    v = np.concatenate(variancias)
    return {
        "var_media": float(v.mean()),
        "rmse_irredutivel": float(np.sqrt(v.mean())),
        "frac_estados_com_r_dependente_da_acao": diferentes / total,
        "estados": total,
    }


# ------------------------------------------------------------------------- H1
class DreamerInstrumentado(DreamerV3):
    """Igual ao original, mais os contadores que dizem **por que** o episódio terminou.

    Guarda `(h, z)` dos passos terminais separados por causa. É a única forma de perguntar
    à cabeça de continuação "você viu essa morte chegando?" — a memória de treino não
    registra a causa, só `cont=0`.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.causas = {"fome": 0, "colisao": 0, "vitoria": 0}
        self._term_fome, self._term_colisao = [], []

    def collect(self):
        cfg = self.cfg
        scores, vitorias = [], 0
        for _ in range(cfg.collect_steps):
            obs_ant, mask_ant, primeiro = self.obs, self.mask, self._primeiro.copy()
            acoes = self._escolher(obs_ant, mask_ant)
            h_t, z_t = self._h.numpy(), self._z.numpy()
            self.obs, self.mask, r, d, info = self.env.step(acoes)

            self.memoria.add(obs_ant, acoes, r, 1.0 - d.astype(np.float32),
                             primeiro, mask_ant)
            self._primeiro = d.copy()
            self.global_step += cfg.num_envs
            scores.extend(info["scores"].tolist())
            vitorias += info["wins"]

            # a causa: `starved` vem em `trunc_idx`; o resto de `done` que não venceu morreu
            fome = np.zeros(cfg.num_envs, bool)
            fome[info["trunc_idx"]] = True
            colisao = d & ~fome
            self.causas["fome"] += int(fome.sum())
            self.causas["colisao"] += int(colisao.sum()) - info["wins"]
            self.causas["vitoria"] += info["wins"]
            for alvo, m in ((self._term_fome, fome), (self._term_colisao, colisao)):
                if m.any() and len(alvo) < 4096:
                    alvo.append(np.concatenate([h_t[m], z_t[m]], axis=-1))

        self.episodes += len(scores)
        return scores, vitorias

    def calibracao_da_continuacao(self):
        """`p(continua)` que o modelo dá aos estados terminais, por causa.

        O alvo correto é 0 nos dois casos. Uma cabeça que prevê colisão e **não** prevê
        fome é a assinatura de H1: no sonho, morrer batendo custa, passar fome é grátis.
        """
        saida = {}
        for nome, buf in (("fome", self._term_fome), ("colisao", self._term_colisao)):
            if not buf:
                saida[f"p_cont_{nome}"] = None
                saida[f"n_{nome}"] = 0
                continue
            estados = np.concatenate(buf, axis=0)[-2048:]
            _, c_lg, _ = self.cabecas(tf.convert_to_tensor(estados, tf.float32))
            saida[f"p_cont_{nome}"] = float(tf.sigmoid(c_lg[:, 0]).numpy().mean())
            saida[f"n_{nome}"] = int(estados.shape[0])
        return saida


def roda(passos_max, saida, preset="dreamer_small", num_envs=64, seed=0):
    cfg = DreamerV3Config(preset=preset, num_envs=num_envs, total_steps=passos_max,
                          seed=seed, eval_every_steps=10**12)
    ag = DreamerInstrumentado(cfg)
    print(f"parâmetros do ator: {ag.ator.count_params():,} · "
          f"train_steps/iter: {cfg.train_steps}", flush=True)

    linhas, t0 = [], time.time()
    prox = 0
    while ag.global_step < passos_max:
        st = ag.iterate()
        if ag.global_step >= prox:
            prox += 25_000
            tot = max(1, sum(ag.causas.values()))
            linha = {
                "passo": ag.global_step,
                "grad_steps": ag._grad_steps,
                "score": st.get("train_score_mean"),
                "media_movel": ag.media_movel(),
                "frac_fome": ag.causas["fome"] / tot,
                "frac_colisao": ag.causas["colisao"] / tot,
                "ent_sonho": st.get("ent_sonho"),
                "rew_sonho": st.get("rew_sonho"),
                "retorno": st.get("retorno"),
                "escala_ret": st.get("escala_ret"),
                "kl_dyn": st.get("kl_dyn"),
                "recon": st.get("recon"),
                "min": (time.time() - t0) / 60.0,
                **ag.calibracao_da_continuacao(),
            }
            linhas.append(linha)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in linha.items()}), flush=True)
            ag.causas = {k: 0 for k in ag.causas}
            with open(saida, "w") as f:
                json.dump(linhas, f, indent=1)
    return linhas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--passos", type=int, default=600_000)
    p.add_argument("--preset", default="dreamer_small")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--saida", default="/tmp/diag_dreamer.json")
    p.add_argument("--so-h3", action="store_true")
    a = p.parse_args()

    print("--- H3: variância da recompensa entre ações, no mesmo estado ---", flush=True)
    print(json.dumps(variancia_da_recompensa_entre_acoes(), indent=1), flush=True)
    if not a.so_h3:
        print("--- H1/H2: treino instrumentado ---", flush=True)
        roda(a.passos, a.saida, a.preset, a.num_envs)
