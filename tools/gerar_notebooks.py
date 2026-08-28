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
    "snakeai/plataforma.py",
    "snakeai/env/vec_snake.py",
    "snakeai/otimizadores.py",
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

#: Os braços da ablação do AlphaZero (`93`). Ficam aqui, e não no notebook, porque a
#: lista de nomes precisa virar o `@param` e o dicionário precisa virar código — as duas
#: coisas saem da mesma fonte para não divergirem.
ENSAIO_MD = """## Ensaio — 2 minutos antes de queimar 8 horas

Um treino de 5 M passos com busca custa ~8 h de T4. Descobrir na hora 6 que a cabeça de
valor divergiu, ou que o alvo saiu `NaN`, é o pior jeito possível de gastar esse tempo.

Esta célula roda **40 iterações** do braço escolhido e imprime o que precisa estar são
antes de valer a pena continuar. O que olhar:

| sinal | saudável | o que significa se sair errado |
|---|---|---|
| `perda_pi`, `perda_v`, `valor_raiz` | finitos, `perda_pi` abaixo de `ln 3 = 1,099` e caindo | `NaN`/`inf` = pare; `perda_pi` colada em 1,099 = o alvo não está ensinando nada |
| `entropia_alvo` | **acima de 0,3** | perto de zero é rótulo duro: a destilação joga fora a distribuição de visitas, que é justamente o que ela deveria aprender (ver o braço `alvo_cru`) |
| `visitas_no_argmax` | entre ~0,4 e ~0,9 | colado em 1,00 é a busca degenerada — todos os rollouts no mesmo filho (ver `docs/BUSCA_DEGENERADA.md`) |
| `‖∇v‖/‖∇π‖` no tronco | idealmente abaixo de ~10× | dezenas ou centenas = o tronco está sendo otimizado para o valor e a política não anda |
| `fome` | não colado em 100% | 100% de fome com score 0 é a cobra andando em círculo |

Ela **não** valida a configuração. São 40 iterações — ~20 mil passos de ambiente, 0,4% do
orçamento — num `resnet_tiny` com 32 ambientes, para caber em dois minutos. Herda do `cfg`
tudo que define o braço (`fpu`, `q_normalizado`, `valor_symlog`, `vf_coef`, temperatura,
`epochs_por_iter`, `num_simulations`) e troca só o tamanho. Serve para pegar o que é
catastrófico e independente de arquitetura: `NaN`, valor explodindo, busca degenerada, alvo
duro. Nada aqui é gravado em `runs/`, e o agente do ensaio é descartado antes do treino
começar.
"""

ENSAIO_CODE = '''import numpy as np, tensorflow as tf

_cfg_ensaio = {**asdict(cfg)}
for _k in ("ckpt_dir", "runs_dir"):
    _cfg_ensaio.pop(_k, None)
_cfg_ensaio.update(net="resnet_tiny", num_envs=32, rollout=16, batch_size=256,
                   memory_size=20_000, total_steps=10**9, eval_every_steps=10**9,
                   log_every_steps=10**9, salvar_gif=False, salvar_grafico=False,
                   ckpt_dir="/tmp/ensaio_az", runs_dir="/tmp/ensaio_az")
_ens = AlphaZero(AlphaZeroConfig(**_cfg_ensaio))

print(f"{'iter':>5} {'busca':>7} {'perda_pi':>9} {'perda_v':>9} {'v_raiz':>8} "
      f"{'ent_alvo':>9} {'argmax':>7} {'fome':>6}")
for _i in range(1, 41):
    _st = _ens.iterate()
    if _i % 10:
        continue
    _n = _ens._cheio
    _pi = _ens._buf_pi[:_n]
    _ent = float(np.mean(-(_pi * np.log(np.maximum(_pi, 1e-12))).sum(1) / np.log(3)))
    _r = _ens.resumo_janela()
    print(f"{_i:>5} {(_st.get('train_score_mean') or 0):>7.2f} "
          f"{_st.get('perda_pi', float('nan')):>9.4f} {_st.get('perda_v', float('nan')):>9.4f} "
          f"{_st.get('valor_raiz', float('nan')):>8.3f} {_ent:>9.3f} "
          f"{float(_pi.max(1).mean()):>7.3f} {_r.get('frac_fome', float('nan')):>6.1%}")

# a razão entre os gradientes NO TRONCO — ver docs/BUSCA_DEGENERADA.md e
# tools/diag_balanco_perdas.py
_tronco = [v for v in _ens.model.trainable_variables
           if not v.path.startswith(("logits", "value", "pi_", "v_"))]
_idx = np.arange(min(256, _ens._cheio))
_obs = tf.convert_to_tensor(_ens._buf_obs[_idx]); _mk = tf.convert_to_tensor(_ens._buf_mask[_idx])
_pa = tf.convert_to_tensor(_ens._buf_pi[_idx]); _z = tf.convert_to_tensor(_ens._buf_z[_idx])
with tf.GradientTape(persistent=True) as _fita:
    _lg, _v = _ens.model(_obs, training=True)
    _v = tf.squeeze(_v, -1)
    _lg = tf.where(_mk, _lg, tf.fill(tf.shape(_lg), MASK_NEG))
    _lpi = -tf.reduce_mean(tf.reduce_sum(_pa * tf.nn.log_softmax(_lg), -1))
    # com o `vf_coef` embutido: é a perda que o otimizador de fato aplica, e sem ele a
    # razão sai 4x maior num braço que usa vf_coef=0,25 — acusando justamente o problema
    # que a configuração já corrigiu
    _lv = cfg.vf_coef * tf.reduce_mean(
        tf.square(_v - (_ens._symlog(_z) if cfg.valor_symlog else _z)))
_gp = float(tf.linalg.global_norm([g for g in _fita.gradient(_lpi, _tronco) if g is not None]))
_gv = float(tf.linalg.global_norm([g for g in _fita.gradient(_lv, _tronco) if g is not None]))
del _fita

_razao = _gv / max(_gp, 1e-9)
_pi_fim = _ens._buf_pi[:_ens._cheio]
_ent_fim = float(np.mean(-(_pi_fim * np.log(np.maximum(_pi_fim, 1e-12))).sum(1) / np.log(3)))
_argmax_fim = float(_pi_fim.max(1).mean())

print()
print(f"|grad_v|/|grad_pi| no tronco: {_razao:.1f}x   "
      f"(ja com vf_coef={cfg.vf_coef}; |z| medio "
      f"{float(np.abs(_ens._buf_z[:_ens._cheio]).mean()):.2f})")

# --- avisos: sinais fracos, que dependem do braço. Não interrompem.
for _cond, _aviso in (
    (_razao > 40, f"o tronco está {_razao:.0f}x mais otimizado para o valor que para a "
                  "política — esperado no `controle`, suspeito com `valor_symlog`"),
    (_ent_fim < 0.3, f"entropia do alvo em {_ent_fim:.3f}: a destilação está aprendendo "
                     "rótulo duro em vez da distribuição de visitas (ver `alvo_cru`)"),
    (_argmax_fim > 0.97, f"{_argmax_fim:.3f} das visitas no argmax: busca degenerada, "
                         "todos os rollouts no mesmo filho (ver docs/BUSCA_DEGENERADA.md)"),
    (_ent_fim > 0.95, f"entropia do alvo em {_ent_fim:.3f}: a busca está quase uniforme e "
                      "o alvo também não ensina nada — a degeneração pelo outro lado"),
    (_argmax_fim < 0.40, f"{_argmax_fim:.3f} das visitas no argmax (acaso = 0,333): a "
                         "busca não está concentrando em lugar nenhum"),
):
    if _cond:
        print("  AVISO:", _aviso)

# --- parada dura: com `Run all`, isto é o que impede 8 horas de treino em cima de um NaN.
_ruins = [_k for _k in ("perda_pi", "perda_v", "valor_raiz")
          if not np.isfinite(_st.get(_k, np.nan))]
if _ruins or not np.isfinite(_ens._buf_z[:_ens._cheio]).all():
    raise RuntimeError(
        f"ensaio reprovado: {_ruins or 'alvo de valor'} não é finito. O treino NÃO vai "
        "começar. Rode o braço `controle` para saber se o problema é da configuração ou "
        "do ambiente, e veja docs/BUSCA_DEGENERADA.md.")

print("ensaio aprovado — perdas, valor e alvo finitos. Pode seguir para o treino.")
del _ens'''


BRACOS_ABLACAO = [
    "consertos", "controle",
    "fpu_pai", "q_normalizado", "valor_symlog", "vf_05", "gradiente_8x", "lr_decai",
    "alvo_cru", "temp_por_lance", "desempate", "bootstrap_janela",
    "busca64", "dirichlet_1", "gamma_995", "tudo",
]

#: O braço que o `@param` oferece pré-selecionado. **Não** é o controle: o controle é o
#: `06_alphazero`, que roda separado. Quem abre este notebook quer medir alguma coisa.
BRACO_PADRAO = "consertos"

_PRE_CFG_ABLACAO = """BRACOS = {
    # ---------------------------------------------------------------- o pacote curado
    # Os consertos com mecanismo medido, mais os que não têm como piorar. Fica de fora
    # tudo que só tem argumento de literatura (`busca64`, `dirichlet_1`, `gamma_995`) e
    # tudo que dobra o tempo de parede. Não isola causa — isola *risco*.
    "consertos": {"fpu": "pai", "q_normalizado": True,          # BUSCA_DEGENERADA §PUCT
                  "valor_symlog": True, "vf_coef": 0.5,         # BUSCA_DEGENERADA §tronco
                  "epochs_por_iter": 8,                         # ORCAMENTO_DE_GRADIENTE
                  "lr_final": 5e-5,                             # como o PPO e o ACKTR
                  "temp_alvo": 1.0, "temp_passos": 30,          # o agendamento do paper
                  "dirichlet_alpha": 1.0,                       # 0,5 é sharp para 3 ações
                  "desempate": "aleatorio",
                  "bootstrap_fim_janela": True},

    # o `06_alphazero` sem tocar em nada, para quando não houver uma execução de `06` na
    # mesma semente com que comparar
    "controle": {},

    # ------------------------------- a escala do valor dentro do PUCT (REVISAO §2.27)
    "fpu_pai":        {"fpu": "pai"},
    "q_normalizado":  {"q_normalizado": True},

    # ------------------------- o balanço das perdas no tronco compartilhado (§2.28)
    "valor_symlog":   {"valor_symlog": True},
    "vf_05":          {"vf_coef": 0.5},
    "lr_decai":       {"lr_final": 5e-5},

    # ------------------------------------- o orçamento de gradiente (o eixo do 95/96)
    "gradiente_8x":   {"epochs_por_iter": 8},

    # ------------------------------------------------ a destilação e a temperatura
    # `alvo_cru` sozinho é o mais barato dos consertos: com tau=0,25 a partir de metade
    # do treino, o alvo de politica vira rotulo duro (entropia 0,66 -> 0,015 nas
    # contagens de visita medidas). `temp_por_lance` mexe so em QUANDO a temperatura cai.
    "alvo_cru":       {"temp_alvo": 1.0},
    "temp_por_lance": {"temp_passos": 30},

    # ------------------------------------------------------- dois vieses pequenos
    "desempate":       {"desempate": "aleatorio"},
    "bootstrap_janela": {"bootstrap_fim_janela": True},

    # --------------------------------- o que a literatura sugere, e que eu nao endosso
    # `busca64` DOBRA o tempo de parede (~16 h em vez de ~8 h por semente).
    "busca64":        {"num_simulations": 64, "sims_avaliacao": 64},
    "dirichlet_1":    {"dirichlet_alpha": 1.0},
    "gamma_995":      {"gamma": 0.995},

    # ------------------------------------------------------------------- tudo junto
    "tudo": {"fpu": "pai", "q_normalizado": True, "valor_symlog": True, "vf_coef": 0.5,
             "epochs_por_iter": 8, "lr_final": 5e-5,
             "temp_alvo": 1.0, "temp_passos": 30,
             "desempate": "aleatorio", "bootstrap_fim_janela": True,
             "num_simulations": 64, "sims_avaliacao": 64, "dirichlet_alpha": 1.0,
             "gamma": 0.995},
}
print(f"braço: {BRACO}")
for _k, _v in sorted(BRACOS[BRACO].items()):
    print(f"   {_k} = {_v!r}")
if not BRACOS[BRACO]:
    print("   (nada — é o 06_alphazero)")

"""


NOTEBOOKS = [
    {
        "arquivo": "99_ablacoes.ipynb",
        "titulo": "Ablações — arquitetura e otimizador",
        "modulos": ["snakeai/memory/replay.py", "snakeai/agents/dqn.py"],
        "agente": "DQN",
        "config": "DQNConfig",
        "resumo": "Dois eixos que o repositório antigo nunca conseguiu medir: qual tronco "
                  "convolucional é melhor, e se o otimizador importa. O segundo é o "
                  "sucessor do K-FAC, que dependia de `tensorflow.contrib` e não roda "
                  "desde o TF2.",
    },
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
        "arquivo": "03_rainbow.ipynb",
        "titulo": "Rainbow — os seis componentes juntos",
        "modulos": ["snakeai/memory/replay.py", "snakeai/agents/dqn.py",
                    "snakeai/agents/rainbow.py"],
        "agente": "Rainbow",
        "config": "RainbowConfig",
        "resumo": "double + dueling + PER + n-step + noisy + C51. Não é algoritmo novo — "
                  "é a soma canônica da família DQN, com linha própria na arena para não "
                  "virar um rótulo ilegível.",
    },
    {
        "arquivo": "04_a2c.ipynb",
        "titulo": "A2C — o controle experimental",
        "modulos": ["snakeai/agents/ppo.py", "snakeai/agents/a2c.py"],
        "agente": "A2C",
        "config": "A2CConfig",
        "resumo": "PPO sem clipping e sem reaproveitar o rollout. A diferença entre as "
                  "duas curvas mede exatamente quanto essas duas coisas valem.",
    },
    {
        "arquivo": "05_acer.ipynb",
        "titulo": "ACER",
        "modulos": ["snakeai/memory/trajectory.py", "snakeai/agents/acer.py"],
        "agente": "ACER",
        "config": "ACERConfig",
        "resumo": "Retrace(λ), IS truncado com correção de viés, região de confiança.",
    },
    {
        "arquivo": "06_alphazero.ipynb",
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
        "arquivo": "07_muzero.ipynb",
        "titulo": "MuZero — a mesma busca, sobre um modelo aprendido",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/nets/muzero.py", "snakeai/agents/muzero.py"],
        "agente": "MuZero",
        "config": "MuZeroConfig",
        "resumo": "Deve perder para o AlphaZero — o simulador aqui é exato e gratuito. "
                  "O que se mede é quanto custa não tê-lo.",
    },
    {
        "arquivo": "08_acktr.ipynb",
        "titulo": "ACKTR — gradiente natural com K-FAC",
        "modulos": ["snakeai/kfac.py", "snakeai/agents/ppo.py", "snakeai/agents/a2c.py",
                    "snakeai/agents/acktr.py"],
        "agente": "ACKTR",
        "config": "ACKTRConfig",
        "resumo": "A dívida de 2019 paga: quatro notebooks do `colab-rl` tentaram K-FAC "
                  "via `tensorflow.contrib` e nenhum roda. Aqui a curvatura é aproximada "
                  "por fatores de Kronecker em Keras 3 puro, e o tamanho do passo sai de "
                  "uma KL alvo, não do learning rate. Compare com `04_a2c`: é o mesmo "
                  "algoritmo com uma única troca.\n\n"
                  "A região de confiança vem **calibrada** por padrão: sem isso `kl_max` "
                  "é um alvo nominal que a Fisher aproximada erra por ~7×, e a mesma "
                  "semente entregou 83,91 num Colab e 64,53 num Kaggle. Com a KL entregue "
                  "presa em ~0,007, o ACKTR fecha ~90% dos tabuleiros. A versão sem "
                  "calibrar virou a ablação `98_acktr_kl_nominal`.",
    },
    {
        "arquivo": "12_acektr.ipynb",
        "titulo": "ACEKTR — os autovalores medidos, não fatorados",
        "modulos": ["snakeai/kfac.py", "snakeai/agents/ppo.py", "snakeai/agents/a2c.py",
                    "snakeai/agents/acktr.py", "snakeai/agents/acektr.py"],
        "agente": "ACEKTR",
        "config": "ACEKTRConfig",
        "resumo": "O `08_acktr` com **uma** troca: o EK-FAC no lugar do K-FAC.\n\n"
                  "De `A ⊗ G = (U_A ⊗ U_G)(S_A ⊗ S_G)(U_A ⊗ U_G)ᵀ` o K-FAC tira duas "
                  "coisas, e só uma se justifica: uma **base** de autovetores (defensável) "
                  "e uma **escala por eixo** obrigada a ter forma de produto, "
                  "`λ_A(j)·λ_G(i)` (que não vem de lugar nenhum além de ter saído junto). "
                  "O EK-FAC fica com a base e **mede** as escalas — o segundo momento "
                  "verdadeiro do gradiente projetado. Pelo Teorema 3 do paper ele nunca é "
                  "pior que o K-FAC, e sai barato porque o gradiente por amostra é um "
                  "produto externo: a média dos quadrados vira um produto de matrizes.\n\n"
                  "**O controle é exato:** com `ema_escalas=1` o EK-FAC não mede nada, "
                  "`s*` fica no palpite do K-FAC e as duas direções coincidem até o "
                  "arredondamento de float32 — `tests/test_ekfac.py` prova isso. Compare "
                  "com `08_acktr` na mesma semente.\n\n"
                  "Olhe `ekfac_desvio` no registro: é o tamanho da correção que está "
                  "sendo aplicada, em dente de serra entre as reconstruções da base. "
                  "Grudado em zero significa que não há o que corrigir neste problema — o "
                  "que é um resultado, e distingue \"não ajudou\" de \"não fez nada\". "
                  "Ver `docs/EKFAC.md`.",
    },
    {
        "arquivo": "93_alphazero_ablacoes.ipynb",
        "titulo": "AlphaZero — as ablações, e o pacote de consertos",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/agents/alphazero.py"],
        "agente": "AlphaZero",
        "config": "AlphaZeroConfig",
        "param_braco": True,
        "celulas_extra": [{"md": ENSAIO_MD, "codigo": ENSAIO_CODE, "titulo": "Ensaio"}],
        "resumo":
            "**Como rodar.** Escolha o `BRACO` no `@param`, rode a célula de ensaio (2 min, "
            "pega o que é catastrófico antes de você gastar 8 h) e depois o treino. O "
            "`sufixo_variante` mantém cada braço separado na arena. O braço vem "
            "pré-selecionado em `consertos`, **não** em `controle`: o controle é o "
            "`06_alphazero` da mesma semente, que roda em outro notebook.\n\n"
            "**O que comparar.** A curva oficial é o `[eval]`, que mede a rede pura, greedy, "
            "sem busca — não o `score` do log de treino, que é o da busca. O piso aleatório "
            "é 1,21. A execução de referência do `06` está em 2,45 em 1 M de passos com a "
            "busca fazendo 17,8: busca boa, destilação parada.\n\n"
            "---\n\n"
            "### O braço `consertos` — o que está dentro e por quê\n\n"
            "Não isola causa. Isola **risco**: entra o que tem mecanismo medido ou não tem "
            "como piorar, e fica de fora tudo que só tem argumento de literatura ou que "
            "dobra o tempo de parede.\n\n"
            "| chave | o que conserta | evidência |\n"
            "|---|---|---|\n"
            "| `fpu=\"pai\"` + `q_normalizado` | o PUCT dá `Q = 0` a filho não visitado; com "
            "`V ≈ 3,5` (medido) o bônus `c_puct·P·√N` só cobre isso onde o prior já é alto, "
            "e a busca passa a confirmar a rede em vez de corrigi-la | medido: somar uma "
            "constante ao valor da folha, sem mudar o ranking de estado nenhum, leva o "
            "score de 21,70 (100% colisão) a **0,00** (100% fome) |\n"
            "| `valor_symlog` + `vf_coef=0.5` | o alvo de valor não é normalizado e domina "
            "o tronco compartilhado | medido **na execução real**: `perda_v/perda_pi` = "
            "**57,6×** depois de 4 M. No `|z|` dessa execução, symlog leva o gradiente de "
            "71× para 14×, e `vf_coef=0,5` para **7,0×** — a faixa saudável. `0,25` daria "
            "3,5× e passaria do ponto: a `perda_v` **não** convergiu (subiu de 0,34 para "
            "1,0), então enfraquecer o valor mais que isso piora a busca, que depende dele |\n"
            "| `epochs_por_iter=8` | ~4.900 atualizações contra as ~38.300 do PPO no mesmo "
            "orçamento de ambiente | **atenção**: na execução real a `perda_pi` já chega a "
            "0,016, então mais gradiente **não** ajuda a política — ajuda o valor, que "
            "ficou em 1,0 sem convergir. Custa ~5% de tempo |\n"
            "| `lr_final=5e-5` | o `lr` era constante, e este é o único agente do "
            "repositório sem decaimento | na execução real o score oscila entre 9,6 e 12,5 "
            "depois de 3 M sem tendência, e o `best` (13,03) fica 2,4 pontos acima do "
            "`last` (10,62), que é o número oficial |\n"
            "| `temp_alvo=1.0` | a mesma π temperada escolhe a ação **e** vira o alvo; com "
            "τ = 0,25 o alvo vira rótulo duro | medido: entropia do alvo de 0,66 a "
            "**0,015** nas contagens de visita reais |\n"
            "| `temp_passos=30` | `temp_frac` é fração do **treino**, não do episódio: a "
            "cobra joga solta quando o tabuleiro aperta | é o agendamento canônico do paper |\n"
            "| `dirichlet_alpha=1.0` | com 3 ações, α = 0,5 põe mais de 90% da massa numa "
            "única ação em 15% dos lances | a heurística do paper é α ≈ 10/n, que dá 3,3 "
            "para 3 ações. **Julgamento, não medição** — é o item mais fraco da lista |\n"
            "| `desempate=\"aleatorio\"` | empate exato no PUCT fica sempre com a primeira "
            "ação da máscara, que é *virar à esquerda* | viés sistemático, sempre para o "
            "mesmo lado |\n"
            "| `bootstrap_fim_janela` | o último passo de cada janela tem alvo sem "
            "bootstrap — 1/16 das amostras aprendendo \"aqui não há futuro\" | |\n\n"
            "### O que a execução de controle já mediu\n\n"
            "`runs/alphazero/…` (5 M passos, 7,54 h, seed 0): política pura **10,62** "
            "(pico 13,03 em 3,0 M), `score_median` 5, **86,9% de fim por fome**, 0% de "
            "tabuleiro cheio. Os três GIFs do fim terminam por fome. Para comparar: PPO "
            "62–81, ACKTR ~84. Este notebook existe para descobrir se os 11 pontos são o "
            "algoritmo ou os três mecanismos acima.\n\n"
            "**Fora do pacote, de propósito:** `busca64` dobra o tempo de parede (~16 h em "
            "vez de ~8 h) e a medição diz que compra ~1,3 plies, não horizonte; "
            "`gamma_995` é alinhamento com o resto do repositório, não conserto, e com "
            "symlog o argumento de escala do valor já está resolvido.\n\n"
            "### Os braços isolados\n\n"
            "Cada um muda **uma** coisa em relação ao `06_alphazero`, para atribuir causa "
            "depois que o `consertos` disser se há o que atribuir. Um cuidado: "
            "`temp_por_lance` sozinho mantém `temp_fim = 0,25`, então ele muda *quando* a "
            "temperatura cai, não o quanto — a dureza do alvo continua sendo assunto do "
            "`alvo_cru`.\n\n"
            "Tudo isto está escrito em [`docs/BUSCA_DEGENERADA.md`]"
            "(https://github.com/voaneves/snake-arena/blob/main/docs/BUSCA_DEGENERADA.md) e "
            "nas §2.27 e §2.28 da revisão, com os scripts que regeneram as tabelas "
            "(`tools/diag_busca.py`, `tools/diag_balanco_perdas.py`).",
    },
    {
        "arquivo": "94_rainbow_nstep3.ipynb",
        "titulo": "Rainbow com a janela de 3 do paper — o braço que não sai do chão",
        "modulos": ["snakeai/memory/replay.py", "snakeai/agents/dqn.py",
                    "snakeai/agents/rainbow.py"],
        "agente": "Rainbow",
        "config": "RainbowConfig",
        "extra_cfg": "    n_steps=3,",
        "resumo": "O braço de controle do `n_steps`, e o resultado mais violento do "
                  "repositório: **0,57 contra 65,43**, com uma única linha de diferença.\n\n"
                  "`n_steps=3` é o valor canônico do Rainbow (Hessel et al.) e era o padrão "
                  "aqui. Ele produziu uma execução que passou 5 M de passos no chão — e o "
                  "score sozinho diria \"não aprendeu\", o que é **falso**. A repartição "
                  "das causas de fim diz o que de fato aconteceu: **100% dos episódios "
                  "terminaram por fome e nenhum por colisão**. O agente aprendeu a "
                  "sobreviver e não a comer; num tabuleiro com máscara de ação, andar em "
                  "círculo é o ponto fixo mais barato que existe.\n\n"
                  "Com `n_steps=20` a mesma configuração faz 65,43, terminando 87,8% por "
                  "colisão — um agente que arrisca, com a decolagem saindo de ~1,85 M "
                  "passos para ~700 k.\n\n"
                  "O mecanismo é o alcance do sinal: o agente gasta ~12 passos por maçã, e "
                  "com uma janela de 3 a decisão que o levou até a comida sai do retorno "
                  "antes de a recompensa entrar — a atribuição de crédito passa a depender "
                  "só do bootstrap, que depende das sincronias do alvo. Com 20 a maçã entra "
                  "na mesma janela da decisão, e `γ**20 = 0,905` ainda reduz o peso do "
                  "bootstrap. O valor vem do **Data-Efficient Rainbow** (van Hasselt et "
                  "al., 2019), a configuração do Rainbow para o regime de poucos dados — "
                  "que é o regime de 5 M passos deste contrato.\n\n"
                  "Compare com `03_rainbow` na mesma semente. **Uma semente de cada lado** "
                  "— o tamanho do efeito não está estabelecido, a diferença qualitativa "
                  "está. Ver `docs/REVISAO_ALGORITMOS.md` §2.25.",
    },
    {
        "arquivo": "98_acktr_kl_nominal.ipynb",
        "titulo": "ACKTR sem calibrar a região de confiança — o que se perde",
        "modulos": ["snakeai/kfac.py", "snakeai/agents/ppo.py", "snakeai/agents/a2c.py",
                    "snakeai/agents/acktr.py"],
        "agente": "ACKTR",
        "config": "ACKTRConfig",
        "extra_cfg": "    kl_calibrado=False,\n    kl_max=2e-3,",
        "resumo": "O braço de controle da calibração: `kl_max` volta a ser um alvo "
                  "**nominal** de 0,002, e o que a rede entrega é ~0,014 — o fator "
                  "sistemático entre a Fisher aproximada e a KL da política de verdade. "
                  "Foi essa a configuração até agosto, e ela produziu 83,91 num Colab e "
                  "64,53 num Kaggle **com a mesma semente**: o fator não controlado muda "
                  "com o hardware. Aqui a medição fica registrada em vez de virar "
                  "anedota. Compare com `08_acktr` na mesma semente.",
    },
    {
        "arquivo": "96_ppo_orcamento_esparso.ipynb",
        "titulo": "PPO com o orçamento de gradiente antigo — o braço de controle",
        "modulos": ["snakeai/agents/ppo.py"],
        "agente": "PPO",
        "config": "PPOConfig.esparso",
        "resumo": "O mesmo orçamento de **ambiente** do contrato, gasto em ~2.400 "
                  "atualizações de gradiente em vez de ~38.300: `rollout` 96, três "
                  "épocas, oito minilotes de 6.144. Era o padrão até a ablação de "
                  "orçamento, e é o braço de controle dela. Três sementes de cada lado "
                  "deram 62,19 contra 80,90 de score e 4,4% contra 60,1% de tabuleiro "
                  "cheio — com a dispersão entre sementes caindo de 9,79 para 1,80. "
                  "Compare com `01_ppo` na mesma semente: a rede, o ambiente e o "
                  "orçamento de passos são idênticos. Ver `docs/ORCAMENTO_DE_GRADIENTE.md`.",
    },
    {
        "arquivo": "95_a2c_orcamento_esparso.ipynb",
        "titulo": "A2C com o rollout antigo \u2014 o mesmo eixo, numa terceira fam\u00edlia",
        "modulos": ["snakeai/agents/ppo.py", "snakeai/agents/a2c.py"],
        "agente": "A2C",
        "config": "A2CConfig.esparso",
        "resumo": "O bra\u00e7o de controle do or\u00e7amento de gradiente fora do PPO. O "
                  "`rollout` volta de 5 para 16, e os mesmos 5 milh\u00f5es de passos de "
                  "ambiente passam a ser gastos em ~610 atualiza\u00e7\u00f5es de lote 8.192 "
                  "em vez de ~1.953 de lote 2.560. \u00c9 o \u00fanico bot\u00e3o: o A2C n\u00e3o tem "
                  "\u00e9pocas nem minilotes para reaproveitar o rollout, ent\u00e3o aqui a "
                  "vari\u00e1vel aparece isolada de qualquer outra coisa \u2014 no PPO, `96` "
                  "mexe em tr\u00eas bot\u00f5es de uma vez. Compare com `04_a2c` na mesma "
                  "semente. Ver `docs/ORCAMENTO_DE_GRADIENTE.md`.",
    },
    {
        "arquivo": "97_ppo_canal_de_fome.ipynb",
        "titulo": "PPO com o sexto canal — quanto custa não ver o relógio da fome",
        "modulos": ["snakeai/agents/ppo.py"],
        "agente": "PPO",
        "config": "PPOConfig",
        "extra_cfg": ("    canal_fome=True,\n"
                      "    comparable=False,\n"
                      '    caveat="observação com 6 canais (fome), fora do contrato de 5",'),
        "resumo": "A observação do contrato tem 5 canais e **nenhum deles é a fome**, "
                  "enquanto o limite é `100 + 2·comprimento` passos sem comer. Dois "
                  "estados visualmente idênticos, um com fome 5 e outro com fome 105, "
                  "valem coisas diferentes — e a rede não tem como saber. Aqui o sexto "
                  "canal traz `fome / limite`, e a pergunta é **quanto** isso vale: o "
                  "PPO já fecha ~90% de vitória cego para ela, então a hipótese é que o "
                  "ganho seja pequeno e apareça na *eficiência* (passos até 40 pontos), "
                  "não no teto. Compare com `01_ppo` **na mesma semente**: é a única "
                  "diferença entre os dois. Esta execução nasce `comparable=False` — ela "
                  "muda a entrada da rede e não pode dividir eixo com as curvas de 5 "
                  "canais.",
    },
    {
        "arquivo": "10_lbc.ipynb",
        "titulo": "LBC — controle de comportamento aprendido",
        "modulos": ["snakeai/bandit.py", "snakeai/agents/ppo.py",
                    "snakeai/agents/lbc.py"],
        "agente": "LBC",
        "config": "LBCConfig",
        "resumo": "O único dos dez em que a exploração é **escolhida** em vez de "
                  "agendada. Nos outros, o ε, o coeficiente de entropia e o σ da rede "
                  "ruidosa descem numa reta decidida antes do treino começar; aqui o "
                  "comportamento é uma mistura de Boltzmann sobre uma população de três "
                  "políticas, e um bandit UCB escolhe a mistura olhando o retorno que "
                  "cada uma rendeu.\n\n"
                  "Como o comportamento não é nenhuma das políticas treinadas, os dados "
                  "são off-policy por construção e o update usa **V-trace**. Compare com "
                  "`01_ppo` na mesma semente: mesma rede, mesmo ambiente, mesmo γ na "
                  "política avaliada — a diferença entre as curvas é o preço (ou o "
                  "prêmio) de trocar exploração agendada por exploração selecionada. "
                  "Ver `docs/LBC.md` para os três desvios declarados em relação ao paper.",
    },
    {
        "arquivo": "11_soap.ipynb",
        "titulo": "SOAP — opções discretas para uma observação que não é markoviana",
        "modulos": ["snakeai/agents/ppo.py", "snakeai/agents/soap.py"],
        "agente": "SOAP",
        "config": "SOAPConfig",
        "resumo": "A observação do contrato tem 5 canais e **nenhum deles é a fome**, "
                  "enquanto o limite é `100 + 2·comprimento` passos sem comer: dois "
                  "estados visualmente idênticos, um com fome 5 e outro com fome 105, "
                  "valem coisas diferentes. O `97_ppo_canal_de_fome` tentou resolver "
                  "isso pela observação e custou a comparabilidade — sem ganho. O SOAP "
                  "tenta pela **memória**: uma opção latente discreta que atravessa os "
                  "passos, dentro dos mesmos 5 canais.\n\n"
                  "São `Z = 4` sub-políticas, uma política de troca `π_ψ(z\'|s,a,z)` e "
                  "uma crença `ζ_t` atualizada pelo filtro para a frente. A vantagem que "
                  "treina a troca é a *Generalized Option Advantage*, uma recursão para "
                  "trás que substitui a retropropagação pelo tempo.\n\n"
                  "**O controle vem embutido:** com `n_opcoes=1` o SOAP é literalmente o "
                  "PPO — `ζ ≡ 1`, `A^GOA = A^GAE`, mesma perda com clipping — e "
                  "`tests/test_soap.py` prova as igualdades numericamente. Compare com "
                  "`01_ppo` na mesma semente. Ver `docs/SOAP.md` para o que olhar quando "
                  "as opções colapsarem.",
    },
    {
        "arquivo": "09_dreamerv3.ipynb",
        "titulo": "DreamerV3 — treinar dentro de um modelo do mundo",
        "modulos": ["snakeai/memory/sequencia.py", "snakeai/nets/dreamer.py",
                    "snakeai/agents/dreamerv3.py"],
        "agente": "DreamerV3",
        "config": "DreamerV3Config",
        "resumo": "O único dos três algoritmos com modelo que não busca nada na hora de "
                  "agir: o modelo serve "
                  "para **treinar**, em rollouts imaginados. symlog, two-hot, KL "
                  "balanceada e free bits são o que dispensam ajuste por ambiente. "
                  "É o mais caro por passo de ambiente — comece com `dreamer_tiny`.",
    },
]

RE_IMPORT_RELATIVO = re.compile(r"^from\s+\.+[\w.]*\s+import\s+")
RE_FUTURE = re.compile(r"^from\s+__future__\s+import\s+")


def _limpa(fonte, caminho):
    """Remove imports relativos e docstring de módulo, mantendo o resto intacto.

    O import relativo pode ocupar várias linhas, e não necessariamente com o parêntese
    sozinho no fim::

        from ..kfac import (KFac, captura_kfac,
                            perda_fisher_gaussiana)

    Por isso a continuação é detectada **contando parênteses**, não olhando se a linha
    termina em `(`. A versão anterior olhava só o fim da linha, deixava a segunda linha
    órfã e o notebook nascia com `IndentationError` — e o gerador não reclamava, porque
    ele não compila o que gera.
    """
    linhas = fonte.splitlines()
    saida, abertos = [], 0
    for linha in linhas:
        if abertos > 0:
            abertos += linha.count("(") - linha.count(")")
            continue
        if RE_IMPORT_RELATIVO.match(linha.strip()):
            # `from ..x import y as z` não pode ser simplesmente removido: no notebook o
            # módulo inlinado define `y`, nunca `z`, e a ligação do apelido morava só no
            # import. O resultado é um `NameError` que só aparece quando aquela linha
            # roda — no caso que motivou isto, no fim de um treino de 5 M passos.
            if re.search(r"\bimport\b.*\bas\b", linha):
                raise ValueError(
                    f"{caminho}: import relativo com apelido não sobrevive ao "
                    f"achatamento do notebook:\n    {linha.strip()}\n"
                    "  Importe sem `as`, ou renomeie a função na origem."
                )
            abertos = linha.count("(") - linha.count(")")
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
    #: Linhas de configuração específicas de uma variante — o que faz o `98` ser o `08`
    #: com uma chave a mais, em vez de um agente novo. `""` para todos os outros.
    extra = spec.get("extra_cfg", "")
    if extra and not extra.endswith("\n"):
        extra += "\n"
    #: Código inserido **antes** do `cfg = ...`, na célula de parâmetros. Existe para o
    #: notebook de ablações, onde um `@param` escolhe o braço e o dicionário do braço
    #: precisa existir antes de ser desempacotado dentro do config.
    pre = spec.get("pre_cfg", "")
    if pre and not pre.endswith("\n"):
        pre += "\n"
    if spec.get("param_braco"):
        # A lista do @param e o dicionário do notebook são duas escritas da mesma coisa e
        # moram em constantes diferentes; sem esta conferência, acrescentar um braço só
        # num dos lados produz um dropdown com uma opção que estoura em `BRACOS[BRACO]`
        # — e só na hora de rodar, no Colab, depois de a célula do núcleo carregar.
        _ns = {}
        exec(_PRE_CFG_ABLACAO.split("print(")[0], _ns)          # noqa: S102
        if set(_ns["BRACOS"]) != set(BRACOS_ABLACAO):
            raise ValueError(
                "BRACOS_ABLACAO e o dicionário de _PRE_CFG_ABLACAO divergiram: "
                f"só na lista {sorted(set(BRACOS_ABLACAO) - set(_ns['BRACOS']))}, "
                f"só no dicionário {sorted(set(_ns['BRACOS']) - set(BRACOS_ABLACAO))}")
        braco_param = ('\n' + f'BRACO = "{BRACO_PADRAO}"  # @param ['
                       + ", ".join(f'"{k}"' for k in BRACOS_ABLACAO) + "]")
        pre = _PRE_CFG_ABLACAO
        extra = "    **BRACOS[BRACO],\n    sufixo_variante=f\"_{BRACO}\",\n"
    else:
        braco_param = ""
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
        # A assinatura vai como **constante**, não só como comentário: `record._ambiente`
        # a lê do namespace do notebook e grava em `meta["assinatura_pacote"]`. No Kaggle
        # não há clone git, e sem isso a curva nasce sem procedência nenhuma.
        _code(f"{MARCA_INICIO}\n# assinatura: {marca}\n\n{fonte}\n\n"
              f'ASSINATURA_PACOTE = "{marca}"\n\n{MARCA_FIM}'),
        _md(f"""## Configuração

Os padrões abaixo são os do **contrato**: tabuleiro 10×10, 5 M passos de orçamento,
avaliação de 1.000 episódios com semente 123. Mexer neles é legítimo para experimentar,
mas o resultado só entra na arena se o contrato for respeitado — o `Recorder` recusa
qualquer outra coisa e diz o motivo.
"""),
        _code(f"""SEMENTE = 0        # @param {{type:"integer"}}
PASSOS = 5000000   # @param {{type:"integer"}}
REDE = "resnet_small"  # @param ["resnet_tiny", "resnet_small", "resnet_base", "cnn_rainbow", "cnn_alphazero", "cnn_vgg", "cnn_vgg_dropout", "cnn_vgg_sem_pool"]{braco_param}

# Armazenamento: nada para configurar. Detecta Colab, Kaggle ou máquina local e escolhe a
# pasta que **persiste** em cada um — Drive, /kaggle/working ou o diretório atual. Se a
# montagem do Drive falhar, avisa e segue, em vez de parar.
PASTA = pasta_de_trabalho()

# No Kaggle a sessão nova nasce com /kaggle/working vazio: o que sobreviveu está montado
# somente-leitura em /kaggle/input. Isto traz os checkpoints de volta — e nunca sobrescreve
# um checkpoint desta sessão, senão o treino andaria para trás.
semear_checkpoints(os.path.join(PASTA, "checkpoints"))

{pre}cfg = {config}(
    seed=SEMENTE,
    net=REDE,
    total_steps=PASSOS,
{extra}    ckpt_dir=os.path.join(PASTA, "checkpoints"),
    runs_dir=os.path.join(PASTA, "runs"),
)
print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))""", "Parâmetros"),
        #: Células específicas de um notebook, inseridas entre a configuração e o treino.
        #: Hoje só o `93` usa: a célula de ensaio, que é o que separa "descobri um NaN em
        #: dois minutos" de "descobri um NaN na sexta hora".
        *[c for extra in spec.get("celulas_extra", [])
          for c in (_md(extra["md"]), _code(extra["codigo"], extra.get("titulo")))],
        _md("""## Treino

**Retomável, e é requisito, não conveniência.** Um treino de 5 M passos não cabe numa
sessão gratuita sem cair pelo menos uma vez. Rode a célula de novo e ela continua do último
checkpoint.

* **Colab** — os checkpoints vão para o Drive e sobrevivem à queda da sessão.
* **Kaggle** — `/kaggle/working` vira a **saída** desta versão. Para continuar depois:
  *Save Version → Save & Run All* (roda headless, sem aba aberta), e na execução seguinte
  *Add Input → Your Work → Notebook Output* apontando para esta. A célula de parâmetros
  recupera os checkpoints sozinha.
"""),
        _code(f"""agente = {agente}(cfg)
if agente.retomar("last"):
    print("retomando do checkpoint")
print("parâmetros:", f"{{agente.model.count_params():,}}")

registro = agente.train(verbose=True)""", "Treinar"),
        _md("""## Veredito — os dois modelos

Duas perguntas diferentes, dois números:

* **`last`** — o modelo do último passo. É ele que entra na curva e na arena, porque é o
  estado final do algoritmo, instabilidade inclusa.
* **`best`** — o melhor checkpoint já visto. É ele que você levaria para o jogo.

Os dois existem porque **RL profundo não melhora monotonicamente**: fora do caso tabular
não há garantia nenhuma, e uma execução pode terminar pior do que já esteve. Na primeira
execução longa do ACKTR, 8 das 21 avaliações tinham um checkpoint anterior melhor que o
modelo daquele momento — numa delas, 21,7 pontos melhor.

Dentro de cada um, três regimes: piso aleatório, política pura e política com o filtro de
segurança. Se a coluna do meio não estiver bem acima do piso, não aprendeu — e aí o
problema é hiperparâmetro ou tempo de treino, não código.
"""),
        _code("""print("=== last · modelo do último passo (é o que entra na arena) ===")
_fome = getattr(agente.env, "canal_fome", False)
resultado = verdict(agente.politica(), episodes=1000, canal_fome=_fome)
print(format_verdict(resultado))

melhor = agente.modelo_melhor()
if melhor is not None:
    print()
    print(f"=== best · checkpoint do passo "
          f"{registro.record.melhor.get('global_step', 0):,} ===")
    _guardado, agente.model = agente.model, melhor
    try:
        print(format_verdict(verdict(agente.politica(), episodes=1000,
                                     canal_fome=_fome)))
    finally:
        agente.model = _guardado

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
        agente.politica(), caminho=f"episodio_last_s{semente}.gif", seed=semente,
        canal_fome=getattr(agente.env, "canal_fome", False))
    print(f"last · semente {semente}: score {score}, terminou por {motivo}")
    display(Image(filename=caminho))""", "GIF"),
        _md("""## Exportar — os dois

`.keras` para retomar treino, TFLite fp16/int8 para embarcar no jogo. A paridade de **ação**
contra o `.keras` é conferida — diferença numérica de quantização é aceitável, ação
diferente não é.

A conferência é pulada quando a política **tem memória** (o SOAP, com a crença de opção; o
DreamerV3, com o latente do modelo do mundo). Não é um detalhe de implementação: um
`.tflite` que recebe só a observação não consegue reproduzir uma política cuja ação depende
de estado interno, então "as ações batem" seria uma afirmação sobre outra coisa. Os arquivos
continuam sendo gerados e medidos; o que não se afirma é a paridade.

Exporta `last` **e** `best`, em pastas separadas. Exportar é para usar, e o que você leva
para o jogo é o melhor; mas o `last` vai junto porque é ele que corresponde ao número da
arena, e misturar os dois é como se perde a rastreabilidade entre o gráfico e o arquivo.
"""),
        _code("""relatorios = {}
# `apos_passo` é o contrato das políticas com memória (ver `snakeai/eval.py`). Quem o
# expõe não pode ter a paridade de ação conferida contra um `.tflite` sem estado.
_COM_MEMORIA = hasattr(agente.politica(), "apos_passo")
if _COM_MEMORIA:
    print("política com memória: TFLite exportado, paridade de ação não conferida")

relatorios["last"] = export_model(
    agente.model, out_dir=os.path.join(PASTA, "export", "last"),
    validar=not _COM_MEMORIA)

_melhor = agente.modelo_melhor()
if _melhor is not None:
    relatorios["best"] = export_model(
        _melhor, out_dir=os.path.join(PASTA, "export", "best"),
        validar=not _COM_MEMORIA)

print(json.dumps(relatorios, indent=2, ensure_ascii=False))""", "Exportar"),
        _md("""## Onde ficou o resultado

O `history.json` da execução vai para `runs/<algo>/<variante>/seed<N>/`, junto com a curva e
os GIFs. Essa pasta é o que entra na arena: coloque em `runs/` do repositório e rode
`python -m snakeai.arena --all`.

Ele carrega os dois resultados: `final` (o modelo do último passo, que é o número oficial)
e `melhor` (o melhor checkpoint, com o passo em que apareceu). Junto vão `modelos/last.keras`
e `modelos/best.keras` — a pasta é autossuficiente, quem a recebe consegue rodar o agente
sem depender de nada que ficou nesta máquina.

Sobre versionar isso no GitHub: o registro vai (`history.json`, `curva.png` e os GIFs), os
**pesos não**. Um `.keras` vai de 0,8 MB (`resnet_small`) a 6,7 MB (`cnn_rainbow` com dueling
e C51), e a arena inteira passa de 100 MB só de modelo — binário em git **nunca some do
histórico**, então cada re-execução deixaria mais uma cópia lá para sempre. O `.gitignore` já
tira `runs/**/*.keras` e `runs/**/*.npz`; o lugar deles é um *Release* do GitHub, que é feito
para binário e não entra no clone. Os arquivos continuam na sua pasta — o que muda é só o que
o git carrega.
"""),
        _code("""CAMINHO_REGISTRO = registro.save(skip_validation=True)
print("registro:", CAMINHO_REGISTRO)

problemas = validate(registro.record)
print("entra na arena?" , "sim" if not problemas else "NÃO:")
for p in problemas:
    print("  -", p)

_f = registro.record.final.get("score_mean")
_m = registro.record.melhor.get("score_mean")
if _f is not None and _m is not None:
    print()
    print(f"last  {_f:.2f}   (passo {registro.record.steps()[-1]:,})")
    print(f"best  {_m:.2f}   (passo {registro.record.melhor.get('global_step', 0):,})")
    if _m > _f:
        print(f"→ a execução terminou {_m - _f:.2f} abaixo do melhor que já esteve. "
              "Normal: RL profundo não melhora monotonicamente.")""",
              "Conferir o contrato"),
        _md("""## Baixar o resultado

Um `.zip` só, com a pasta inteira da execução — registro, curva, GIFs e o modelo exportado.

**Um arquivo, e não vários downloads**, por dois motivos: o navegador bloqueia downloads
múltiplos disparados em sequência, e a pasta da execução só faz sentido inteira — o
`history.json` sem a curva e sem os GIFs perde metade do que ela responde.

A entrega muda com a plataforma, e o `.zip` existe nos dois casos:

* **Colab** — dispara o download pelo navegador, o que exige a aba aberta. Se ela não
  estiver, a célula imprime o caminho em vez de falhar: o download é conveniência, o
  arquivo é o resultado.
* **Kaggle** — não há o que disparar, e é por isso que ele aguenta execução headless: o
  que está em `/kaggle/working` aparece sozinho no painel **Output**, à direita, e é
  baixável de lá com a aba fechada.
"""),
        _code("""import shutil

PASTA_EXECUCAO = os.path.dirname(CAMINHO_REGISTRO)

# o export mora fora da pasta da execução; copiamos para dentro antes de zipar,
# senão o .zip sai sem o modelo — que é justamente o que se leva para o jogo
_export = os.path.join(PASTA, "export")
if os.path.isdir(_export):
    shutil.copytree(_export, os.path.join(PASTA_EXECUCAO, "export"), dirs_exist_ok=True)

_nome = "_".join([registro.record.algo, registro.record.variant,
                  f"seed{registro.record.seed}"])
ZIP = shutil.make_archive(os.path.join(PASTA, _nome), "zip", PASTA_EXECUCAO)
print(f"{ZIP}  ({os.path.getsize(ZIP) / 1e6:.1f} MB)")
for _raiz, _, _arqs in os.walk(PASTA_EXECUCAO):
    for _a in sorted(_arqs):
        print("   ", os.path.relpath(os.path.join(_raiz, _a), PASTA_EXECUCAO))

entregar_arquivo(ZIP)""",
              "Baixar tudo num .zip"),
    ]

    # O gerador compila o que gera. Duas vezes um escape mal escrito virou uma quebra de
    # linha dentro de uma f-string e o notebook nasceu com `SyntaxError` — nas duas o
    # defeito só apareceu depois, porque `tests/test_notebooks.py` confere o arquivo em
    # disco e eu tinha gerado antes de rodar os testes. Falhar aqui é falhar cedo.
    for i, c in enumerate(celulas):
        if c["cell_type"] != "code":
            continue
        fonte = "".join(c["source"])
        try:
            compile(fonte, f"{spec['arquivo']}[{i}]", "exec")
        except SyntaxError as e:
            raise SyntaxError(
                f"{spec['arquivo']}: a célula {i} não compila ({e.msg}, linha {e.lineno}). "
                "Quase sempre é um `\\n` dentro de uma f-string do template — escreva "
                "`print()` numa linha separada em vez de escapar."
            ) from e

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
