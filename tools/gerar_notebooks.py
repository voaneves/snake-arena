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

Esta célula roda **40 iterações** da configuração escolhida e imprime o que precisa
estar são antes de valer a pena continuar. O que olhar:

| sinal | saudável | o que significa se sair errado |
|---|---|---|
| `perda_pi`, `perda_v`, `valor_raiz` | finitos, `perda_pi` abaixo de `ln 3 = 1,099` e caindo | `NaN`/`inf` = pare; `perda_pi` colada em 1,099 = o alvo não está ensinando nada |
| `entropia_alvo` | **acima de 0,3** | perto de zero é rótulo duro: a destilação joga fora a distribuição de visitas, que é justamente o que ela deveria aprender (ver o braço `alvo_cru`) |
| `visitas_no_argmax` | entre ~0,4 e ~0,9 | colado em 1,00 é a busca degenerada — todos os rollouts no mesmo filho (ver `docs/BUSCA_DEGENERADA.md`) |
| `‖∇v‖/‖∇π‖` no tronco | idealmente abaixo de ~10× | dezenas ou centenas = o tronco está sendo otimizado para o valor e a política não anda |
| `fome` | não colado em 100% | 100% de fome com score 0 é a cobra andando em círculo |

Ela **não** valida a configuração. São 40 iterações — ~20 mil passos de ambiente, 0,4% do
orçamento — num `resnet_tiny` com 32 ambientes, para caber em dois minutos. Herda do `cfg`
tudo que define o comportamento (`fpu`, `q_normalizado`, `valor_symlog`, `vf_coef`, temperatura,
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


BUSCA_MD = """## Veredito com busca — a coluna separada do contrato

A curva oficial mede a **rede pura**, greedy, sem nenhuma ajuda. É o que torna as curvas
comparáveis: a busca gasta `num_simulations` avaliações de rede e outros tantos passos de
simulador **por jogada**, contra 1 avaliação do PPO. Somar as duas no mesmo eixo diria "o
AlphaZero ganha do PPO" quando o que aconteceu foi gastar 32× mais computação na hora de
decidir — a mesma razão que manda o filtro de flood-fill para uma coluna própria.

Reportar, no entanto, é obrigação. Um agente que existe para buscar, avaliado só sem
buscar, é meia medição — e a busca é o que você levaria para jogar de verdade, já que em
Snake o simulador está disponível na hora de agir. Então: coluna separada, não coluna
proibida.

O protocolo é o mesmo do contrato — 1.000 episódios, greedy (argmax das visitas), semente
123 — e a busca roda com a **mesma** configuração do treino (`fpu`, `q_normalizado`,
`desempate`, `c_puct`, `gamma`). Dois orçamentos, para mostrar a curva computação ×
qualidade: quanto do resultado vem da rede e quanto vem do lookahead.

**Custo, e por que ele te pega de surpresa.** O laço roda até cada ambiente fechar a cota,
e **um agente bom faz episódios longos** — no AlphaZero de 5 M eles passam de 900 passos.
Com 64 ambientes são ~16 episódios cada, ou seja ~15 mil passos de ambiente, cada um
custando `num_simulations + 1` avaliações de rede. No MuZero é o dobro disso, porque cada
simulação chama `g` **e** `f`, e a árvore nunca poda nós terminais (o modelo aprendido não
prevê fim de episódio). Uma medição de 1.000 episódios com 32 simulações passa de uma hora,
e é exatamente esse o motivo de a primeira tentativa nesta célula ter sido cancelada.

Por isso ela vem com `MINUTOS_MAX`. Ao estourar, o que deu tempo de medir volta marcado
`completo=False` — o mesmo campo que o `validate()` usa para recusar avaliação parcial, de
modo que o número serve para você olhar e **não entra na arena por engano**. O progresso é
impresso a cada 30 s, com estimativa do que falta.

**Comece pequeno se for só olhar.** `EPISODIOS = 200` e **um** orçamento dão a ordem de
grandeza em poucos minutos. Subir `EPISODIOS` é o lever que importa; `AMBIENTES` mais alto
melhora o aproveitamento da GPU mas encarece o laço de árvore em Python na mesma proporção,
então costuma ser quase neutro.

**Mas o padrão é 1.000, porque é o que entra na arena.** O resultado vai para o campo
`busca` do registro — irmão de `final` e `melhor`, não um canto de `meta` —, e a coluna
*com busca* da tabela só aceita entradas com os 1.000 episódios do contrato e
`completo=True`. Uma medição de 200 episódios tem erro padrão ~2,2× o da oficial; uma que
estourou `MINUTOS_MAX` é uma amostra enviesada para episódios **curtos**, que são
justamente os ruins. As duas ficam gravadas e marcadas — o registro guarda o que você
mediu, a arena publica só o que cumpre a régua.
"""

BUSCA_CODE = '''import time

import numpy as np

# O contrato pede 1000. Menos que isso mede e grava, mas nao entra na arena: a coluna
# `com busca` da tabela so aceita o protocolo inteiro. 200/20min da a ordem de grandeza.
EPISODIOS_DO_CONTRATO = 1000
EPISODIOS = 1000              # @param {type:"integer"}
MINUTOS_MAX = 120             # @param {type:"integer"}
ORCAMENTOS = [cfg.sims_avaliacao]   # @param
AMBIENTES = 64                # @param {type:"integer"}
AVALIAR_MELHOR = False        # @param {type:"boolean"}

if not hasattr(agente, "avaliar_com_busca"):
    raise RuntimeError("este agente não busca na hora de agir — a coluna não se aplica")

# reaproveita o `resultado` da célula anterior: sem busca é barato, mas o filtro de
# flood-fill é laço Python e é a parte cara da avaliação
_base = globals().get("resultado")
if _base is None:
    _base = verdict(agente.politica(), episodes=EPISODIOS, com_filtro=False)
_tabela = {**_base, "linhas": list(_base["linhas"])}
_pura = _tabela["linhas"][1]["score_mean"]
_medidas = {}


def _mede_com_busca(_ag, modelo_nome):
    for _i, _sims in enumerate(ORCAMENTOS):
        _t0 = time.time()
        print(f"  {modelo_nome} · {_sims} sims ({EPISODIOS} episódios, teto "
              f"{MINUTOS_MAX} min)...", flush=True)
        _st = _ag.avaliar_com_busca(episodes=EPISODIOS, num_simulations=_sims,
                                    num_envs=AMBIENTES, max_segundos=MINUTOS_MAX * 60,
                                    verbose=True)
        _dt = time.time() - _t0
        # `checkpoint` e `num_simulations` viajam DENTRO da entrada: sem eles o
        # número não é interpretável, e ler a chave para descobrir de onde veio é um
        # contrato implícito que quebra na primeira vez que alguém renomeia a chave
        _st = {**_st, "checkpoint": modelo_nome}
        _medidas[f"{modelo_nome}_sims{_sims}"] = _st
        _rotulo = f"agente + busca ({_sims} sims)"
        if modelo_nome != "last":
            _rotulo += f" · {modelo_nome}"
        if not _st["completo"]:
            _rotulo += " ⚠ parcial"
        _tabela["linhas"].append({"regime": _rotulo, **_st})
        print(f"    score {_st['score_mean']:.2f} · cheio {_st['win_rate']:.1%} · "
              f"{_st['episodes']} episódios · {_dt / 60:.1f} min"
              + ("" if _st["completo"] else "  ⚠ TEMPO ESGOTADO: amostra parcial, "
                                            "`completo=False`, fora da arena"), flush=True)
        if _i == 0 and len(ORCAMENTOS) > 1:
            _resto = sum(ORCAMENTOS[1:]) / max(ORCAMENTOS[0], 1) * _dt
            print(f"     (os orçamentos restantes devem levar ~{_resto / 60:.0f} min)",
                  flush=True)


print(f"rede pura (a curva oficial): {_pura:.2f}   ·   piso {_tabela['piso']:.2f}")
print(f"medindo com busca, {EPISODIOS} episódios por orçamento...", flush=True)
_mede_com_busca(agente, "last")

if AVALIAR_MELHOR:
    # **Não** troque `agente.model` para medir o `best`, que é o que a célula do veredito
    # faz. Ali funciona porque `politica()` lê o atributo a cada chamada, em eager. Aqui a
    # busca passa por uma `tf.function` cujo traço **já capturou as variáveis** do modelo
    # atual: a troca seria silenciosamente ignorada e você mediria o `last` outra vez. No
    # MuZero é ainda mais silencioso — `model` é uma property com setter vazio. Um agente
    # novo tem o cache de traço vazio, e é a única forma honesta de medir o outro
    # checkpoint com busca.
    _ag_best = type(agente)(cfg)
    if _ag_best.retomar("best"):
        _mede_com_busca(_ag_best, "best")
    else:
        print("  sem checkpoint `best` — pulando")

print()
print(format_verdict(_tabela))
print()
for _nome, _st in _medidas.items():
    _s = _st["score_mean"]
    _linha = f"{_nome:>16}: busca {_s:>6.2f}  ·  rede pura {_pura:>6.2f}"
    # as razões só significam alguma coisa longe do zero; cedo no treino ambas são ~0 e
    # dividir uma pela outra imprime um número de sete dígitos que não quer dizer nada
    if _pura <= 0.5 or _s <= 0.5:
        _linha += "  ·  (razões omitidas: alguma das duas ainda está perto de zero)"
    elif _s > _pura:
        _linha += f"  ·  {_s / _pura:.2f}x  ·  a rede captura {_pura / _s:.0%} da busca"
    else:
        _linha += f"  ·  {_s / _pura:.2f}x  ·  a rede está À FRENTE da busca aqui"
    print(_linha)

# `busca` é campo de primeira classe do registro desde o schema 2 — irmão de `final` e
# `melhor`, e não um canto de `meta`. A diferença não é organizacional: o que mora em
# `meta` não passa por `validate()`, e isto **é um resultado**.
registro.record.busca = {**(registro.record.busca or {}), **_medidas}
_oficiais = [k for k, v in _medidas.items()
             if v.get("episodes") == EPISODIOS_DO_CONTRATO and v.get("completo")]
print()
print("gravado em `busca` de", registro.save())
if _oficiais:
    print(f"entram na arena: {', '.join(_oficiais)}")
else:
    print(f"NENHUMA entrada entra na arena — o contrato pede {EPISODIOS_DO_CONTRATO} "
          "episodios completos. O que foi medido fica gravado e marcado como espiada.")'''


ENSAIO_MZ_MD = """## Ensaio — 2 minutos antes de queimar 7 horas

Um treino de 5 M passos do MuZero custou **6,8 h** na execução `unroll5/seed0`. Esta célula
roda **40 iterações** da configuração escolhida, num `resnet_tiny` com 32 ambientes, e
imprime o que precisa estar são antes de valer a pena continuar.

O que olhar aqui é diferente do ensaio do AlphaZero, porque o modo de falha medido é
diferente:

| sinal | saudável | o que significa se sair errado |
|---|---|---|
| `frac_pi_0` | **~46% com `normaliza_unroll`, ~1/(K+1) sem** | é a fatia da perda de política que pertence ao passo 0 — o único que a métrica oficial mede. É o número que este notebook existe para mexer; se o braço escolhido liga o peso e ele continua em 1/(K+1), a chave não chegou no config |
| `perda_pi_0` | caindo, abaixo de `ln 3 = 1,099` | colada em 1,099 é o passo 0 não aprendendo nada da busca |
| `perda_r` | caindo para perto de zero | é a única âncora do estado oculto no mundo; se não cai, a dinâmica está inventando física |
| `perda_v` | finita e caindo | em `symlog`; o MCTS lê o valor na escala real |
| `v_busca` | positivo e crescendo devagar | é o que a árvore soma no backup |
| `fome` | não colado em 100% | 100% de fome com score 0 é a cobra andando em círculo |

Ela **não** valida a configuração — são ~20 mil passos de ambiente, 0,4% do orçamento, numa
rede menor. Serve para pegar o que é catastrófico e independente de arquitetura: `NaN`,
perda explodindo, âncora de recompensa morta, e a chave do braço não tendo efeito. Nada
aqui é gravado em `runs/`, e o agente do ensaio é descartado antes do treino começar.
"""

ENSAIO_MZ_CODE = '''import numpy as np

_cfg_ensaio = {**asdict(cfg)}
for _k in ("ckpt_dir", "runs_dir"):
    _cfg_ensaio.pop(_k, None)
_cfg_ensaio.update(net="resnet_tiny", num_envs=32, rollout=16, batch_size=256,
                   memory_size=20_000, total_steps=10**9, eval_every_steps=10**9,
                   log_every_steps=10**9, salvar_gif=False, salvar_grafico=False,
                   ckpt_dir="/tmp/ensaio_mz", runs_dir="/tmp/ensaio_mz")
_ens = MuZero(MuZeroConfig(**_cfg_ensaio))

_alvo = 1.0 / (cfg.unroll + 1)
print(f"unroll={cfg.unroll}  normaliza_unroll={cfg.normaliza_unroll}  "
      f"-> frac_pi_0 esperada: {'~46%' if cfg.normaliza_unroll else f'~{_alvo:.1%}'}")
print(f"{'iter':>5} {'busca':>7} {'perda_pi':>9} {'perda_pi_0':>11} {'frac_pi_0':>10} "
      f"{'perda_v':>8} {'perda_r':>8} {'v_busca':>8} {'fome':>6}")
_nan = float("nan")
for _i in range(1, 41):
    _st = _ens.iterate()
    if _i % 10:
        continue
    _r = _ens.resumo_janela()
    print(f"{_i:>5} {(_st.get('train_score_mean') or 0):>7.2f} "
          f"{_st.get('perda_pi', _nan):>9.4f} {_st.get('perda_pi_0', _nan):>11.4f} "
          f"{_st.get('frac_pi_0', _nan):>9.1%} {_st.get('perda_v', _nan):>8.4f} "
          f"{_st.get('perda_r', _nan):>8.4f} {_st.get('valor_busca', _nan):>8.3f} "
          f"{_r.get('frac_fome', _nan):>6.1%}")

_f = _st.get("frac_pi_0", _nan)
_esperado = 0.46 if cfg.normaliza_unroll else _alvo
print()
if abs(_f - _esperado) > 0.12:
    print(f"ATENCAO: frac_pi_0={_f:.1%} longe do esperado ({_esperado:.1%}). "
          "Confira se a chave do braco chegou no cfg.")
else:
    print(f"frac_pi_0={_f:.1%} bate com o esperado ({_esperado:.1%}).")

del _ens
'''


BRACOS_ABLACAO = [
    # os três mecanismos, um removido por vez — é a pergunta científica, em 3 execuções
    "sem_conserto_da_busca", "sem_conserto_do_tronco", "sem_conserto_do_alvo",
    "sem_correcoes",
    # o mesmo, botão a botão, para quem quiser atribuir dentro de um mecanismo
    "sem_fpu", "sem_q_normalizado", "sem_symlog", "vf_1", "sem_alvo_cru",
    "sem_temp_por_lance", "gradiente_1x", "sem_lr_decai", "dirichlet_05",
    "sem_desempate", "sem_bootstrap_janela",
    # o que a literatura sugere e **não** entrou no padrão
    "busca64", "gamma_995",
]

#: O braço pré-selecionado. Aqui não existe "só rodar": o agente oficial é o
#: `06_alphazero`, e este notebook é o menu de ablações. Vem em `sem_conserto_da_busca`
#: porque é a hipótese com o maior efeito esperado — a busca saiu de score 25 para 94,8
#: entre a execução sem consertos e a com.
BRACO_PADRAO = "sem_conserto_da_busca"

_PRE_CFG_ABLACAO = """BRACOS = {
    # ------------------------------------------ os tres mecanismos, um de cada vez
    # Cada um destes remove um conserto INTEIRO do padrao. Tres execucoes respondem
    # "qual dos tres mecanismos carregava o resultado", que e a pergunta que importa.
    # Comparar contra `06_alphazero` na MESMA semente.

    # §2.27 - o PUCT dava Q=0 a filho nao visitado; com valor positivo a busca so
    # confirmava o que a rede ja achava, em vez de discordar
    "sem_conserto_da_busca":  {"fpu": "zero", "q_normalizado": False},

    # §2.28 - o alvo de valor nao normalizado dominava o tronco compartilhado
    "sem_conserto_do_tronco": {"valor_symlog": False, "vf_coef": 1.0},

    # §2.29 - a temperatura transformava o alvo de politica em rotulo duro
    "sem_conserto_do_alvo":   {"temp_alvo": 0.0, "temp_passos": 0},

    # tudo desligado: o agente de antes. Ja existe na arena como
    # `alphazero/sims32_sem_correcoes/seed0` (10,62). Rode de novo so se quiser o
    # controle sob o codigo atual, com a assinatura atual.
    "sem_correcoes": {"fpu": "zero", "q_normalizado": False, "valor_symlog": False,
                      "vf_coef": 1.0, "epochs_por_iter": 1, "lr_final": 0.0,
                      "temp_alvo": 0.0, "temp_passos": 0, "dirichlet_alpha": 0.5,
                      "desempate": "ordem", "bootstrap_fim_janela": False},

    # ------------------------------------------------------- botao a botao, dentro
    "sem_fpu":              {"fpu": "zero"},
    "sem_q_normalizado":    {"q_normalizado": False},
    "sem_symlog":           {"valor_symlog": False},
    "vf_1":                 {"vf_coef": 1.0},
    "sem_alvo_cru":         {"temp_alvo": 0.0},
    "sem_temp_por_lance":   {"temp_passos": 0},
    "gradiente_1x":         {"epochs_por_iter": 1},
    "sem_lr_decai":         {"lr_final": 0.0},
    "dirichlet_05":         {"dirichlet_alpha": 0.5},
    "sem_desempate":        {"desempate": "ordem"},
    "sem_bootstrap_janela": {"bootstrap_fim_janela": False},

    # ------------------------- o que a literatura sugere e nao entrou no padrao
    # `busca64` DOBRA o tempo de parede (~16 h em vez de ~8 h por semente) e a medicao
    # diz que compra ~1,3 plies de profundidade, nao horizonte.
    "busca64":   {"num_simulations": 64, "sims_avaliacao": 64},
    "gamma_995": {"gamma": 0.995},
}
print(f"braco: {BRACO}  (o padrao e o 06_alphazero; aqui se remove uma coisa dele)")
for _k, _v in sorted(BRACOS[BRACO].items()):
    print(f"   {_k} = {_v!r}")

"""


# ---------------------------------------------------------------------------------
# MuZero — §2.31. Aqui a inversão do `93` não vale: o padrão do `07` **não** é a
# resposta, é o que produziu a oscilação. Estes braços **acrescentam** coisas.
# ---------------------------------------------------------------------------------
BRACOS_MUZERO = [
    # a hipótese principal e as suas variações
    "normaliza_unroll", "unroll10_normalizado", "unroll2_normalizado",
    # o regime de reúso: sair dele de graça, ou ficar nele com alvo fresco
    "reuso_do_paper", "reanalise_25", "reanalise_80", "reanalise_80_sims12",
    "normaliza_e_reanalise",
    # o que o Apêndice F e o Apêndice D mandam e o repositório não fazia
    "categorico", "transformacao_h", "categorico_h", "temp_de_treino", "priorizado",
    # o valor — este **contra** o paper
    "coef_valor_1",
    # a frescura do alvo
    "memoria_200k", "memoria_20k",
    # o que o instinto sugere, e que a aritmética diz ir para o lado errado
    "unroll10",
    # o que a literatura sugere quando o resultado oscila
    "sims32",
    # controle e soma
    "controle", "tudo",
]

BRACO_PADRAO_MUZERO = "normaliza_unroll"

_PRE_CFG_MUZERO = """BRACOS = {
    # ------------------------------------------------- a hipotese principal (§2.31)
    # `perda_pi` e uma SOMA CRUA sobre K+1 termos. O passo 0 - o unico que a metrica
    # oficial mede, porque `politica()` age sobre a observacao real - vale 14,5% dela
    # com unroll=5. O pseudocodigo do paper escala so os K passos imaginados por 1/K,
    # o que poe o passo 0 em ~46% qualquer que seja K. Medido em tools/diag_unroll.py.
    "normaliza_unroll":     {"normaliza_unroll": True},

    # o desenrolar longo SEM diluir o passo 0. E o `unroll10` feito do jeito certo.
    "unroll10_normalizado": {"unroll": 10, "normaliza_unroll": True},
    # o contraste: se o peso e o que importa, K curto com peso tambem anda - e custa
    # metade do tempo de parede por atualizacao
    "unroll2_normalizado":  {"unroll": 2, "normaliza_unroll": True},

    # ------------------------------------- a hipotese secundaria, CONTRA o paper
    # `perda_v` fica em ~0,19 em symlog, o que da uma banda de [6,7; 17,5] na escala
    # real - e e esse valor que o MCTS soma no backup. Mas 0,25 E O VALOR DO PAPER: o
    # Apendice H baixa o alvo de valor para 0,25 contra 1,0 de politica e recompensa,
    # exatamente para "avoid overfitting of the value function". Subir isto e ir CONTRA
    # o paper. O braco fica, porque a hipotese e testavel; a prioridade cai.
    "coef_valor_1":         {"coef_valor": 1.0},

    # ------------------------------------------ o regime de reuso do Reanalyse
    # 8 epocas x 256 / 1024 passos novos = 2,0 amostras por estado. O paper usa 0,1 no
    # MuZero puro e 2,0 no Reanalyse - e o Reanalyse REFAZ A BUSCA com a rede atual em
    # 80% das atualizacoes, mais rede alvo para o bootstrap.
    #
    # Duas saidas. `reuso_do_paper` volta ao reuso do MuZero puro, de graca, pagando em
    # orcamento de gradiente (§2.1). Os `reanalise_*` ficam com o reuso alto e trazem o
    # alvo fresco, pagando em busca. NAO rode nenhum antes de `normaliza_unroll` voltar:
    # se o conserto de graca resolver, isto aqui vira desnecessario.
    "reuso_do_paper":       {"epochs_por_iter": 1},

    # o alvo de politica do passo 0 refeito com a rede ATUAL, escrito de volta no
    # buffer. 0,8 e o numero do paper; 0,25 e o ponto em que o custo ainda cabe numa
    # noite. Custo medido em tools/diag_reanalise.py.
    "reanalise_25":         {"reanalise": 0.25},
    "reanalise_80":         {"reanalise": 0.80},
    # o Reanalyse com busca mais barata que a da coleta - desvio do paper, e o botao de
    # custo se 0,80 na busca cheia nao couber no tempo de parede
    "reanalise_80_sims12":  {"reanalise": 0.80, "reanalise_sims": 12},
    # a soma do que o Apendice G e o H mandam fazer e o repositorio nao fazia
    "normaliza_e_reanalise": {"normaliza_unroll": True, "reanalise": 0.80},

    # ------------------------------------------------------- a frescura do alvo
    # 50k transicoes sao ~49 iteracoes de atraso: o alvo de visitas veio de uma rede
    # `g` que ja nao existe. Nao ha Reanalyse aqui. As duas direcoes entram porque a
    # teoria nao decide: maior media mais versoes, menor traz alvos mais frescos.
    "memoria_200k":         {"memory_size": 200000},
    "memoria_20k":          {"memory_size": 20000},

    # -------------------------------------------- o que vai para o lado errado
    # PREVISAO PRE-REGISTRADA: fica igual ou pior que o controle. Sozinho, unroll=10
    # leva a fatia do passo 0 de 14,5% para 11,0%. Esta aqui para ser falsificavel.
    "unroll10":             {"unroll": 10},

    # PREVISAO PRE-REGISTRADA: nao ganha nada. O professor nao e o gargalo - o
    # `train_score` (que e o da busca) fica estavel em 58-60 de 2,5 M ate o fim,
    # enquanto a rede pura oscila entre 33 e 66. Melhorar o professor alarga o vao.
    "sims32":               {"num_simulations": 32, "sims_avaliacao": 32},

    # ------------------------------------------ Apendice F: cabeca categorica
    # O paper nao faz regressao escalar no valor nem na recompensa: projeta o alvo num
    # suporte discreto com two-hot e treina com entropia cruzada. Aqui a cabeca sempre
    # foi Dense(1) com erro quadratico. O suporte e dimensionado pelo dominio (teto 60
    # na escala real), e nao copiado do [-300,300] de Atari, que daria resolucao de ~3
    # pontos perto de zero num jogo cujo valor vive entre 0 e ~11. Ver §2.33.
    "categorico":           {"n_suporte": 121},
    # a transformacao do paper (R2D2) no lugar do symlog do DreamerV3 que esta la
    "transformacao_h":      {"transformacao": "h"},
    "categorico_h":         {"n_suporte": 121, "transformacao": "h"},

    # -------------------------------- Apendice D: o agendamento de Atari, nao o de
    # jogo de tabuleiro. Episodios aqui tem ~1200 lances, entao `temp_passos=30` poe
    # 97,5% do episodio em tau=0,25 desde a primeira iteracao. O paper, em Atari,
    # amostra o episodio INTEIRO com tau decaindo por passo de treino. Ver §2.34.
    "temp_de_treino":       {"temp_esquema": "treino"},

    # ---------------------------- Apendice G: replay priorizado, alpha=beta=1
    # P(i) ~ |v_i - z_i|^alpha: a distancia entre o valor que a BUSCA achou na raiz e o
    # retorno que o jogo entregou. Fixa na coleta - nao e o erro da rede, entao nao ha
    # atualizacao de prioridade como no PER do DQN. Hoje o sorteio e uniforme, que e o
    # que o paper faz nos jogos de tabuleiro; em Atari ele prioriza. Ver §2.35.
    "priorizado":           {"per": 1.0},

    # ---------------------------------------------------------- controle e soma
    # reproduz `muzero/unroll5/seed0` (final 49,26 / melhor 66,05) sob o codigo e a
    # assinatura atuais. So vale rodar se a assinatura do pacote tiver mudado.
    "controle": {},
    # tudo o que o paper manda e o repositorio nao fazia, mais o desenrolar longo.
    # `coef_valor` saiu daqui: 0,25 E o numero do Apendice H.
    "tudo": {"normaliza_unroll": True, "reanalise": 0.80, "unroll": 10,
             "memory_size": 200000},
}
print(f"braco: {BRACO}  (o padrao e o 07_muzero; aqui se ACRESCENTA uma coisa a ele)")
for _k, _v in sorted(BRACOS[BRACO].items()) or [("(nada)", "controle")]:
    print(f"   {_k} = {_v!r}")

"""

#: `param_braco` do spec aponta para uma destas entradas. Cada uma e a lista do
#: `@param`, o braco pre-selecionado e o dicionario que a celula de parametros carrega.
#: A conferencia em `monta_notebook` garante que a lista e o dicionario nao divirjam.
VARREDURA_MD = """## Varredura de `kl_max` — a medição principal

**Esta célula é a que responde à pergunta.** A dos braços, logo abaixo, é secundária.

A pergunta certa não é *"qual braço reduz a razão"* — isso é uma comparação de sete vias, de
uma estatística com cauda pesada, com mediadores que agem em silêncio. É:

> **A KL entregue responde à KL pedida?**

Isso se mede com **um agente só**, o padrão, variando `kl_max` por duas décadas. A resposta
é a inclinação da reta em log–log:

| inclinação | o que significa |
|---|---|
| **≈ 1** | a região de confiança funciona: pedir metade entrega metade |
| **≈ 0** | existe um **piso**, e `kl_max` é decorativo acima dele |
| entre 0 e 1 | responde em parte, com ganho sistemático |

Por que isto é robusto e a comparação de braços não era: é a resposta de um sistema ao
próprio input de controle, no mesmo agente, com todo o resto constante. Não depende de
comparar configurações diferentes, nem de acreditar em nenhuma hipótese sobre a causa.

**Duas colunas existem para a medição se autodenunciar.** `no teto` é a fração de
atualizações em que `η` bateu em `lr_start` — alto significa que o passo deixou de ser
governado por `kl_max` e a linha não vale. `clipado` é a fração de variáveis em que o
`clipnorm` mordeu — ele age **por variável, sobre a direção já pré-condicionada**, e medindo
aqui pela primeira vez ele aparece em ~33% das variáveis no padrão. Foram esses dois
mediadores que fizeram duas conclusões anteriores medirem outra coisa sem avisar.

**O aquecimento importa e agora é explícito.** O `baselines` usa `cold_iter = 100` antes de
confiar nos fatores do K-FAC; este repositório não tem cold start. Medindo com 60
atualizações o controle deu 7,4×; com 300, **1,2×**. `AQUECIMENTO` descarta as primeiras, e
a tabela por terços mostra o que sobra da decida.
"""

VARREDURA_CODE = '''import gc
import time

import numpy as np

KL_ALVOS = "2e-3, 5e-3, 1.5e-2, 5e-2, 1.5e-1"  # @param {type:"string"}
ITERS = 300         # @param {type:"integer"}
AQUECIMENTO = 100   # @param {type:"integer"}
ENVS = 512          # @param {type:"integer"}
SEMENTES = "0"      # @param {type:"string"}

_alvos = [float(x) for x in KL_ALVOS.split(",") if x.strip()]
_sems = [int(x) for x in SEMENTES.split(",") if x.strip()]

print(f"{ITERS} atualizacoes por alvo ({AQUECIMENTO} de aquecimento descartadas) · "
      f"{ENVS} ambientes · {REDE} · kl_calibrado=False")
print()
print(f"{'kl_max':>9} {'sem':>4} {'KL entregue':>12} {'IQR':>16} {'razao':>7} "
      f"{'no teto':>8} {'clipado':>8} {'s':>5}")

_pontos, _tercos = [], {}
for _sem in _sems:
    for _alvo in _alvos:
        _t0 = time.time()
        _c = ACKTRConfig(seed=_sem, net=REDE, num_envs=ENVS, kl_max=_alvo,
                         kl_calibrado=False, total_steps=10**9,
                         eval_every_steps=10**9, log_every_steps=10**9,
                         salvar_gif=False, salvar_grafico=False,
                         ckpt_dir="/tmp/diag_kl", runs_dir="/tmp/diag_kl")
        _ag = ACKTR(_c)
        _kl, _no_teto, _clip = [], [], []
        for _i in range(ITERS):
            _st = _ag.iterate()
            if not _st or _st.get("kl") is None:
                continue
            _kl.append(_st["kl"])
            _no_teto.append(1.0 if _st["lr"] >= _st["lr_teto"] * 0.999 else 0.0)
            _clip.append(_st.get("frac_clipado", 0.0))
        _v = np.array(_kl[AQUECIMENTO:] or _kl)
        _q1, _med, _q3 = np.quantile(_v, [0.25, 0.5, 0.75])
        _nt = float(np.mean(_no_teto[AQUECIMENTO:] or _no_teto))
        _cl = float(np.mean(_clip[AQUECIMENTO:] or _clip))
        _pontos.append({"kl_max": _alvo, "semente": _sem, "kl": float(_med),
                        "q1": float(_q1), "q3": float(_q3), "razao": float(_med / _alvo),
                        "no_teto": _nt, "clipado": _cl, "n": int(_v.size)})
        print(f"{_alvo:>9.4f} {_sem:>4} {_med:>12.5f} "
              f"[{_q1:>6.4f};{_q3:>7.4f}] {_med / _alvo:>6.1f}x "
              f"{_nt:>7.0%} {_cl:>7.0%} {time.time() - _t0:>4.0f}s", flush=True)
        # a queda do aquecimento, em tercos, para ela nao se esconder na mediana
        _n3 = max(len(_kl) // 3, 1)
        _tercos[(_sem, _alvo)] = [float(np.median(_kl[i * _n3:(i + 1) * _n3]))
                                  for i in range(3)]
        del _ag
        gc.collect()

# --- a estatistica que decide: inclinacao em log-log
print()
# A inclinacao SO pode ser lida sobre as linhas em que eta nao saturou. Incluir uma
# linha com `no teto` alto mistura a resposta ao alvo com a resposta ao TETO, e produz
# uma inclinacao intermediaria que nao existe. Aconteceu: com as cinco linhas deu 0,44,
# com as tres validas deu 0,08.
def _inclinacao(_ps):
    if len({p["kl_max"] for p in _ps}) < 2:
        return float("nan")
    return float(np.polyfit(np.log10([p["kl_max"] for p in _ps]),
                            np.log10([max(p["kl"], 1e-12) for p in _ps]), 1)[0])

_validos = [p for p in _pontos if p["no_teto"] < 0.10]
_incl_todos = _inclinacao(_pontos)
_incl = _inclinacao(_validos)
print(f"inclinacao log-log da KL entregue contra a pedida: {_incl:.2f}   "
      f"({len(_validos)} de {len(_pontos)} linhas)")
if len(_validos) < len(_pontos):
    print(f"  (com TODAS as linhas daria {_incl_todos:.2f}, e seria leitura errada: as")
    print(f"   linhas com `no teto` alto respondem ao TETO, nao ao alvo)")
if _incl > 0.8:
    print("  ~1 -> a regiao de confianca RESPONDE: pedir metade entrega metade.")
elif _incl < 0.25:
    print("  ~0 -> existe um PISO. `kl_max` e decorativo acima dele, e a 'razao' de")
    print("        qualquer braco mede a distancia ate esse piso, nao curvatura.")
else:
    print("  entre 0 e 1 -> responde em parte, com ganho sistematico.")
if len(_validos) < len(_pontos):
    _fora = [f"{p['kl_max']:g}" for p in _pontos if p["no_teto"] >= 0.10]
    print(f"  FORA DA LEITURA por `no teto` alto: kl_max = {', '.join(_fora)}. Ali o")
    print("  passo e governado por `lr_start`, nao por `kl_max`. Para medi-las, suba")
    print("  `lr_start` e repita — ou ignore-as, se o alvo real do treino for menor.")
if len(_validos) < 3:
    print("  POUCAS LINHAS VALIDAS: a inclinacao acima vale pouco. Acrescente alvos")
    print("  menores, que e onde eta nao satura.")
# a dispersao decide se as medianas sao distinguiveis entre si
_disp = np.median([(p["q3"] - p["q1"]) / max(p["kl"], 1e-12) for p in _validos]) \
    if _validos else float("nan")
_faixa = (max(p["kl"] for p in _validos) / min(p["kl"] for p in _validos)) \
    if _validos else float("nan")
print()
print(f"IQR/mediana tipico: {_disp:.1f}  ·  as entregues variam {_faixa:.2f}x enquanto os")
print(f"alvos variam {max(p['kl_max'] for p in _validos) / min(p['kl_max'] for p in _validos):.1f}x"
      if _validos else "")
if _validos and _faixa < 2.0 and _disp > 1.0:
    print("  A variacao entre as linhas e MENOR que a dispersao dentro de cada uma:")
    print("  as medianas nao sao distinguiveis, o que reforca o veredito de piso.")

print()
print("aquecimento (mediana da KL por terco da execucao):")
print(f"{'kl_max':>9} {'sem':>4} {'1o':>10} {'2o':>10} {'3o':>10}")
for (_sem, _alvo), _t in _tercos.items():
    print(f"{_alvo:>9.4f} {_sem:>4} {_t[0]:>10.5f} {_t[1]:>10.5f} {_t[2]:>10.5f}")

_DIAG = os.path.join(PASTA, "varredura_kl.json")
with open(_DIAG, "w", encoding="utf-8") as _f:
    json.dump({"rede": REDE, "envs": ENVS, "iters": ITERS,
               "aquecimento": AQUECIMENTO, "inclinacao_log_log": _incl,
               "inclinacao_todas_as_linhas": _incl_todos,
               "linhas_validas": [p["kl_max"] for p in _validos],
               "assinatura_pacote": ASSINATURA_PACOTE, "plataforma": detecta(),
               "pontos": _pontos,
               "tercos": {f"s{_s}_kl{_a}": _t for (_s, _a), _t in _tercos.items()}},
              _f, indent=1, ensure_ascii=False)
print()
print("gravado em", _DIAG)
entregar_arquivo(_DIAG)
'''


DIAG_KL_MD = """## Diagnóstico da KL — 15 minutos antes de gastar 3 horas

**Rode esta célula antes de escolher um braço.** A pergunta deste notebook é sobre a KL
**por atualização**, não sobre o score no fim — e ela se responde em ~3 min por braço numa
T4, contra 0,5 h de treino completo por braço. O treino serve para *confirmar* o que aqui
aparecer, e para ligar a KL ao score.

`escala_kl` devolve `η = √(2·kl_max / Δᵀ∇)`: o passo tal que **uma** atualização `ηΔ`
induz `kl_max`. A razão `KL_medida / KL_pedida` deveria ser ~1. A execução de 5 M mediu
**4,4× a 12,4×**, e a revisão atribuiu isso à Fisher aproximada — que é a premissa do
ACEKTR. Esta célula testa as duas explicações concorrentes que não são a Fisher.

`kl_calibrado = False` em todos os braços, e isso **não é detalhe**: ligado, ele mede a
razão e pede `kl_max/c`, de modo que a KL entregue converge para o alvo *qualquer que seja
a causa*. Medir com ele ligado responderia sempre "está calibrado".

| se a razão cair para ~1 em… | a causa é… |
|---|---|
| `sem_momento` | o momento — mas o braço joga fora a redução de variância junto |
| `momento_descontado` | o momento, com o conserto do `baselines` (`lr = η·(1−μ)`) |
| `sem_clip` | o `clipnorm` do PPO caindo sobre a direção já pré-condicionada |
| **nenhum** | a Fisher aproximada — a §2 sobrevive e o ACEKTR mantém a premissa |

**A forma importa.** Medido em CPU: na forma do contrato (512×5, `resnet_small`) o
`controle` dá 15,3×; numa forma reduzida (64×5, `resnet_tiny`) dá **0,9×** — o fenômeno
some. Lote 8× menor, estouro 17× menor, que é o que a hipótese do momento prevê e a da
Fisher não. Não encolha `ENVS` para economizar: você mediria outra coisa.

**E a forma mudou.** Tudo que está escrito acima e em `docs/diag_acktr_kl.json` — os 7,4×
do controle, o piso da KL em ~0,0186, os 15,3× do `sem_clip` — foi medido com
`rollout = 5`, que era o que o `ACKTRConfig` herdava do `A2CConfig` entre 21/08 e 01/09.
O `rollout` voltou a ser **16** (declarado, não herdado: ver a nota em `ACKTRConfig`), e
com ele o lote por atualização é 3,2× maior. Pela própria regra do parágrafo anterior,
lote maior desloca o fenômeno — então esta célula **remede**, não confirma. Se os números
vierem diferentes, o resultado novo é o que vale.

**Bracos de treino.** Os quatro últimos (`rollout5`, `calibrado_v1`, `calibrado_debias`,
`calibrado_debias_prior1`) não aparecem aqui: eles mexem na calibração, que esta célula
desliga em todos por construção. Eles se respondem treinando, e a célula de treino abaixo
é o lugar deles.
"""

DIAG_KL_CODE = '''import gc
import time

import numpy as np

ITERS = 60      # @param {type:"integer"}
ENVS = 512      # @param {type:"integer"}
SO_ESTES = ""   # @param {type:"string"}

# `kl_calibrado=False` entra aqui e nao vem do BRACOS: a calibracao esconde a causa.
# E por isso mesmo os bracos de treino ficam de fora: o que eles mexem E a calibracao,
# entao aqui eles virariam copias do `controle` - duplicados silenciosos na tabela.
_ARMS = {k: {**v, "kl_calibrado": False}
         for k, v in BRACOS.items() if k not in BRACOS_SO_TREINO}
if BRACOS_SO_TREINO:
    print(f"fora desta celula (so treino): {', '.join(BRACOS_SO_TREINO)}")
if SO_ESTES.strip():
    _ARMS = {k: _ARMS[k] for k in
             [x.strip() for x in SO_ESTES.split(",") if x.strip()]}
# `controle` primeiro, sempre: as fracoes do resumo sao contra ele, e uma sessao que caia
# no meio (Colab desconecta, a cota do Kaggle acaba) deixaria a tabela ilegivel sem ele.
if "controle" in _ARMS:
    _ARMS = {"controle": _ARMS.pop("controle"), **_ARMS}

print(f"{ITERS} atualizacoes por braco · {ENVS} ambientes · rollout={ACKTRConfig.rollout}"
      f" · {REDE} · kl_calibrado=False em todos")
if ACKTRConfig.rollout != 5:
    print("ATENCAO: os numeros gravados em `docs/diag_acktr_kl.json` (7,4x no controle, o")
    print("PISO da KL em ~0,0186) foram medidos com rollout=5, que era o default herdado")
    print("do A2C ate 01/09. Com o rollout restaurado o lote por atualizacao e 3,2x maior")
    print("e nada garante que eles se reproduzam - esta celula os REMEDE, nao os confirma.")
print()
print(f"{'braco':>24} {'KL mediana':>11} {'razao':>8} {'p90':>8} "
      f"{'no teto':>7} {'eta med':>9} {'s':>6}")
_linhas, _diag2 = {}, {}
for _nome, _extra in _ARMS.items():
    _t0 = time.time()
    _c = ACKTRConfig(seed=SEMENTE, net=REDE, num_envs=ENVS, total_steps=10**9,
                     eval_every_steps=10**9, log_every_steps=10**9,
                     salvar_gif=False, salvar_grafico=False,
                     ckpt_dir="/tmp/diag_kl", runs_dir="/tmp/diag_kl", **_extra)
    _ag = ACKTR(_c)
    _r, _eta, _teto = [], [], []
    for _i in range(ITERS):
        _st = _ag.iterate()
        if _st and _st.get("kl") is not None:
            _r.append(_st["kl"] / max(_st.get("kl_alvo_efetivo") or _c.kl_max, 1e-12))
            # `escala_kl` devolve `min(lr_max, sqrt(2*kl_max/quad))`. Quando o minimo
            # escolhe `lr_max`, a regiao de confianca esta SATURADA: o passo deixou de
            # ser governado por `kl_max` e a KL entregue nao responde mais ao alvo.
            # Quando eta e minusculo sem estar no teto, `quad` estourou - que e o unico
            # jeito de a KL sair exatamente zero.
            _eta.append(_st["lr"])
            _teto.append(_st["lr_teto"])
    # as 20 primeiras rodam com a media movel do K-FAC ainda crua (o `baselines` tem um
    # `cold_iter=100` que este repositorio nao tem) e ficam de fora da mediana
    _v = np.array(_r[20:] or _r)
    _e, _t = np.array(_eta[20:] or _eta), np.array(_teto[20:] or _teto)
    _linhas[_nome] = float(np.median(_v))
    _diag2[_nome] = {"frac_no_teto": float(np.mean(_e >= _t * 0.999)),
                     "eta_mediano": float(np.median(_e)),
                     "eta_min": float(_e.min()), "teto": float(np.median(_t))}
    print(f"{_nome:>24} {float(np.median(_v)) * _c.kl_max:>11.5f} "
          f"{_linhas[_nome]:>7.1f}x {np.quantile(_v, 0.9):>7.1f}x "
          f"{_diag2[_nome]['frac_no_teto']:>7.0%} {_diag2[_nome]['eta_mediano']:>9.2e} "
          f"{time.time() - _t0:>5.0f}s", flush=True)
    del _ag
    gc.collect()        # sete agentes em sequencia; o traco de cada um segura memoria

_base = _linhas.get("controle")
print()
if _base is None:
    print("sem o braco `controle`, nao ha contra o que comparar.")
elif _base < 1.5:
    # sem estouro no controle nao ha excesso a atribuir, e a divisao por (base-1)
    # imprimiria um numero absurdo. Quase sempre significa forma reduzida demais.
    print(f"SEM ESTOURO: o controle deu {_base:.1f}x — a regiao de confianca esta")
    print("entregando o que pede, e nao ha excesso a atribuir a braco nenhum.")
    print("Isso e um RESULTADO, nao uma falha da medicao. Com 60 atualizacoes o mesmo")
    print("controle deu 7,4x e com 300 deu 1,2x: o estouro era aquecimento do K-FAC.")
    print("Compare `KL mediana` entre `controle` e `kl_do_paper`: se as duas forem")
    print("parecidas apesar de os alvos diferirem 7,5x, o que existe e um PISO.")
else:
    print("fracao do EXCESSO removida (1,0x e o alvo; o excesso e `razao - 1`):")
    for _n, _q in _linhas.items():
        if _n == "controle":
            continue
        _f = 1.0 - max(_q - 1, 0) / (_base - 1)
        _lado = "ABAIXO do alvo" if _q < 0.8 else "acima" if _q > 1.25 else "no alvo"
        print(f"  {_n:>24}: {_f:6.1%}   (razao {_q:.1f}x, {_lado})")
    print()
    print("`no teto` e a fracao de atualizacoes em que eta bateu em `lr_start`. Alto")
    print("significa REGIAO DE CONFIANCA SATURADA: o passo deixou de ser governado por")
    print("`kl_max`, e a razao medida nao mede erro de Fisher nenhum - mede o teto.")
    print("`eta med` minusculo COM `no teto` em 0% e o caso oposto: `quad` estourou e o")
    print("passo virou nada, que e como a KL sai exatamente zero.")

# O numero precisa sobreviver ao fim da sessao. Este diagnostico nao cria um `registro`
# (nao ha treino nenhum aqui), entao ele nao entra no .zip da execucao la embaixo — e sem
# isto o resultado seria print no console, que some quando o Colab desconecta.
_DIAG = os.path.join(PASTA, "diag_acktr_kl.json")
with open(_DIAG, "w", encoding="utf-8") as _f:
    json.dump({"iters": ITERS, "envs": ENVS, "rede": REDE, "semente": SEMENTE,
               "rollout": ACKTRConfig.rollout,
               "assinatura_pacote": ASSINATURA_PACOTE,
               "plataforma": detecta(), "razoes": _linhas, "passo": _diag2,
               "bracos": {k: v for k, v in _ARMS.items()}}, _f,
              indent=1, ensure_ascii=False)
print()
print("gravado em", _DIAG)
entregar_arquivo(_DIAG)
'''


BRACOS_ACKTR = [
    # --- diagnostico da KL (calibracao desligada; rodam na celula curta) -------------
    # os dois suspeitos que nao sao a Fisher, e as combinacoes
    "sem_momento", "momento_descontado", "sem_clip", "sem_momento_sem_clip",
    # o alvo do paper, que o repositorio nao usa
    "kl_do_paper", "kl_do_paper_descontado",
    # controle
    "controle",
    # --- treino (calibracao LIGADA; a celula curta pula estes) ----------------------
    # a regressao do rollout, testada de frente
    "rollout5",
    # de onde vem o ganho da calibracao acordar mais cedo
    "calibrado_v1", "calibrado_debias", "calibrado_debias_prior1",
    # tudo o que se sustenta, junto, para UMA execucao
    "definitiva",
]

#: Bracos que so fazem sentido TREINANDO, porque o que eles mexem e a calibracao — e a
#: celula de diagnostico desliga a calibracao em todos, por construcao. Rodar
#: `calibrado_debias` la produziria uma linha identica ao `controle` e um duplicado
#: silencioso na tabela, que e a classe de erro que este notebook existe para evitar.
BRACOS_SO_TREINO_ACKTR = ("rollout5", "calibrado_v1", "calibrado_debias",
                          "calibrado_debias_prior1", "definitiva")

BRACO_PADRAO_ACKTR = "definitiva"

_PRE_CFG_ACKTR = """BRACOS = {
    # `kl_calibrado=False` em TODOS: ligado, ele mede a razao e pede `kl_max/c`, de modo
    # que a KL entregue converge para o alvo QUALQUER que seja a causa. Medir com ele
    # ligado responderia sempre "esta calibrado". Ver §2.36.

    # ------------------------------------------------- o suspeito numero um: o momento
    # `escala_kl` devolve eta tal que UM passo `eta*Delta` induz `kl_max`. Ele e
    # atribuido como learning_rate de um SGD(momentum=0.9, nesterov=True): em regime o
    # deslocamento e ate `eta*Delta/(1-mu)` = 10x isso, e a KL vai com o QUADRADO.
    # O baselines faz MomentumOptimizer(lr*(1-momentum), momentum) justamente por isso.

    # joga o momento fora - responde a pergunta, mas perde a reducao de variancia.
    # MEDIDO em GPU: 7,4x -> 3,8x. O momento amplifica ~2x, e nao e a causa.
    "sem_momento":           {"momento": 0.0, "kl_calibrado": False},
    # o conserto do baselines: desconta (1-mu) e mantem o momento. MEDIDO: leva a razao
    # a 0,2x - corrige DEMAIS. O (1-mu) supoe acumulo total (10x) e o acumulo real e ~2x.
    # `kl_calibrado`, que estima o fator por media movel, e a resposta certa.
    "momento_descontado":    {"descontar_momento": True, "kl_calibrado": False},

    # ------------------------------------------- o suspeito numero dois: o clipnorm
    # `max_grad_norm=0.5` vem do PPO e o Keras clipa POR VARIAVEL, dentro do
    # apply_gradients - sobre a direcao ja pre-condicionada. No baselines o clip nunca
    # toca a direcao natural.
    #
    # MEDIDO, e ao contrario do que se esperava: tirar o clip PIORA a razao (7,4x -> 15,3x
    # com momento; 3,8x -> 12,9x sem). Ele nao era a causa do estouro - era um FREIO
    # acidental, que encurtava a direcao e derrubava a KL junto. O problema dele e outro:
    # encurta de forma desigual entre camadas, desfazendo em silencio a razao de ser do
    # K-FAC, e faz `kl_max` significar coisas diferentes conforme a norma da direcao.
    "sem_clip":              {"max_grad_norm": 0.0, "kl_calibrado": False},
    "sem_momento_sem_clip":  {"momento": 0.0, "max_grad_norm": 0.0,
                              "kl_calibrado": False},

    # ------------------------------------------------ o alvo de KL do paper de verdade
    # Wu et al. usam 0,001-0,002; aqui `kl_max` e 1,5e-2, uma regiao 7-15x mais larga.
    # O `98_acktr_kl_nominal` media isso com a calibracao ligada e foi aposentado: era
    # exatamente este par de chaves, e o braco abaixo o substitui. Aqui a calibracao
    # esta desligada, e junto com o conserto do momento - que e o que o baselines roda.
    "kl_do_paper":             {"kl_max": 2e-3, "kl_calibrado": False},
    "kl_do_paper_descontado":  {"kl_max": 2e-3, "descontar_momento": True,
                                "kl_calibrado": False},

    # ------------------------------------------- o PISO da KL, que e o achado real
    # Com 300 atualizacoes, `controle` (kl_max=1,5e-2) entrega KL 0,01866 e
    # `kl_do_paper` (kl_max=2e-3) entrega 0,01848. Pedidos que diferem 7,5x, entregas
    # que diferem 1%. A KL entregue NAO responde ao alvo: e um PISO, nao um ganho.
    #



    # o de hoje, com a calibracao desligada. MEDIDO: 7,4x em GPU (15,3x na CPU do
    # assistente - a diferenca entre plataformas e do tamanho do efeito medido, que e o
    # mesmo fenomeno que o docstring do acktr.py ja registrava)
    "controle": {"kl_calibrado": False},

    # =============================================================== BRACOS DE TREINO
    # Daqui para baixo a calibracao fica LIGADA e a celula de diagnostico pula todos.
    # Nao sao perguntas sobre de onde vem o estouro da KL - sao perguntas sobre score.

    # ----------------------------------------- a regressao do rollout, de frente
    # `A2CConfig.rollout` foi de 16 para 5 em 21/08, um dia DEPOIS das tres sementes
    # gravadas de `acktr/resnet_small`, e o `ACKTRConfig` herdava o campo. As tres
    # rodaram com 16; a execucao do ACEKTR de 01/09, com 5. A restauracao (`rollout=16`
    # declarado no `ACKTRConfig`) veio de um argumento mecanico mais duas execucoes
    # confundidas - este braco e o teste de UMA variavel que faltava.
    #
    # A conta: com `gamma*lambda = 0,995*0,95 = 0,945`, a fracao do peso do GAE que sobra
    # no bootstrap `V(s_{t+T})` e `0,945^T` - 76% com T=5, 40% com T=16. Enquanto o
    # shaping segura a recompensa densa isso nao importa; depois de `shaping_frac`
    # (1,25 M de 5 M) a unica recompensa e comida e morte.
    #
    # PREVISAO: fecha na casa dos 70, nao dos 90. Se fechar nos 90, a restauracao do
    # rollout nao explica o que aconteceu com o ACEKTR e a leitura tem que ser refeita.
    "rollout5": {"rollout": 5},

    # ------------------------------- a calibracao acordando mais cedo, e a atribuicao
    # `_fator_kl` e uma media movel com `kl_cal_ema=0,98`: constante de tempo de ~50
    # atualizacoes num orcamento de 610. Partindo de 1,0 quando o fator verdadeiro e
    # 15-25, ela gasta ~8% do treino com o alvo efetivo ate 20x maior que o pedido -
    # exatamente quando a politica ainda e aleatoria.
    #
    # MEDIDO em 96 ambientes, mesma semente: com a EMA crua o fator ainda marcava 1,10 na
    # DECIMA atualizacao (ou seja, nao tinha comecado a corrigir) enquanto a entropia ja
    # havia caido de 1,06 para 0,29; por volta da 45a a politica estava morta (entropia
    # 0,0001, eta colado no teto). Com a media debiasada e prior 15, o fator assenta em
    # ~5 na quinta e a entropia segura em 0,22 ate a 50a.
    #
    # Os tres bracos separam DUAS mudancas que andam juntas no default do ACEKTR:
    "calibrado_v1":            {},                       # o ACKTR das 3 sementes gravadas
    "calibrado_debias":        {"kl_cal_debias": True,   # debias + prior medido
                                "kl_fator_inicial": 15.0},
    "calibrado_debias_prior1": {"kl_cal_debias": True,   # so o debias, sem o prior
                                "kl_fator_inicial": 1.0},
    # Se `calibrado_debias_prior1` ja recupera o ganho, o prior e cosmetico e o que
    # importava era a media nao levar 50 atualizacoes para acordar. Se so
    # `calibrado_debias` recupera, o estrago acontece nas PRIMEIRAS atualizacoes e nem o
    # debias chega a tempo - conclusoes diferentes, um braco de distancia.

    # ============================================ A EXECUCAO UNICA (o padrao do dropdown)
    # Tudo o que se sustenta, junto. E a lista do que ficou de FORA e metade do valor
    # deste braco, entao ela esta escrita abaixo em vez de ser sobreentendida.
    #
    # DENTRO:
    #   * `rollout = 16`, que nao aparece aqui porque virou o default do `ACKTRConfig`.
    #     E a restauracao: as tres sementes gravadas rodaram com 16, o 5 entrou herdado
    #     do `A2CConfig` em 21/08 e nao reproduz resultado nenhum do repositorio.
    #   * `kl_cal_debias` + `kl_fator_inicial = 15`. A media da calibracao tem constante
    #     de tempo de ~50 atualizacoes num orcamento de 610: partindo de 1,0 quando o
    #     fator real e 15-25, ela gasta 8% do treino pedindo um alvo ate 20x maior que o
    #     que quer. Medido em 96 ambientes: com a EMA crua o fator ainda marcava 1,10 na
    #     decima atualizacao enquanto a entropia caia de 1,06 para 0,29, e por volta da
    #     45a a politica estava morta (entropia 0,0001, eta colado no teto). Com o debias
    #     e o prior, o fator assenta em ~5 na quinta e a entropia segura em 0,22 ate a 50a.
    #   * `eval_every_steps = 125_000`: 40 avaliacoes em vez de 20. NAO gasta orcamento -
    #     `avaliar()` roda num `VecSnake` proprio e nao mexe no `global_step`. O que muda
    #     e a densidade da amostragem do `best`, e o contrato ja registra que em 8 das 21
    #     avaliacoes da primeira execucao longa havia um checkpoint anterior melhor que o
    #     modelo daquele momento - numa delas por 21,7 pontos. `avaliar_melhor()` remede o
    #     escolhido do zero com o mesmo protocolo, entao isto acha um pico melhor sem
    #     inflar o numero. Custa wall-clock: a avaliacao fica ~2x mais cara no total.
    #
    # FORA, e por que:
    #   * `max_grad_norm = 0`. E a distorcao conhecida que sobrou - o Keras clipa POR
    #     VARIAVEL sobre a direcao ja pre-condicionada, e ~30% das variaveis batem no
    #     teto, o que desfaz em silencio a razao de ser do K-FAC. Mas o braco `sem_clip`
    #     mediu o oposto do esperado: tirar o clip PIORA a razao da KL (7,4x -> 15,3x) e
    #     satura eta no teto do `lr`. Ele nao era so uma distorcao, era tambem um freio.
    #     Trocar `clipnorm` por `global_clipnorm` (que preserva a direcao) e o conserto
    #     certo, mas o limiar teria que ser recalibrado e ninguem mediu qual. Um palpite
    #     nao entra numa execucao unica.
    #   * `kl_max` maior. A evidencia so vai ate aqui: 0,002 deu 72,50 e 0,015 deu 89,78.
    #     Extrapolar seria tentador, mas a execucao do ACEKTR fechou a porta - ela
    #     acumulou `Sum sqrt(KL)` de 202 contra 57-73 do ACKTR e chegou 20 pontos abaixo.
    #     Movimento nao e o gargalo; direcao e.
    #   * `ent_coef_start` maior. A entropia cai para ~0,08 em 7% do treino em TODAS as
    #     execucoes, e isso e suspeito - mas o debias existe justamente para segurar essa
    #     queda. Mexer nos dois ao mesmo tempo tornaria o resultado inatribuivel.
    #   * `shaping_frac` maior. Adiar o fim do shaping so move o penhasco de lugar, e o
    #     shaping potencial decair a zero e o que garante que a politica otima final seja
    #     a do problema de verdade.
    #
    # O que este braco NAO e: uma aposta. Ele e a uniao do que tem medicao, e a razao de
    # a lista ser curta e que so tres coisas tem.
    "definitiva": {"kl_cal_debias": True, "kl_fator_inicial": 15.0,
                   "eval_every_steps": 125_000},
}

#: Os que a celula de diagnostico da KL pula. Ver o comentario la.
BRACOS_SO_TREINO = ("rollout5", "calibrado_v1", "calibrado_debias",
                    "calibrado_debias_prior1", "definitiva")

print(f"braco: {BRACO}  (o padrao e o 08_acktr; aqui se testa de onde vem o estouro)")
for _k, _v in sorted(BRACOS[BRACO].items()) or [("(nada)", "= o 08_acktr, pareado")]:
    print(f"   {_k} = {_v!r}")
print(f"   rollout = {ACKTRConfig.rollout}  (default restaurado; era 5 ate 01/09)")

"""


# ---------------------------------------------------------------------------------
# LBC — a populacao. O `10_lbc` mede o algoritmo; este mede se a POPULACAO vale o que
# custa, e com que eixo de `H`. Ver `docs/LBC.md` §2.12.
BRACOS_LBC = [
    "H_shaping", "H_shaping_pop5", "H_shaping_aleatoria", "pop1", "H_gamma_aleatoria",
]

#: `H_shaping` e o braco principal: e a hipotese de que a populacao da execucao anterior
#: era fraca por causa do eixo, e nao por causa da ideia.
BRACO_PADRAO_LBC = "H_shaping"

_PRE_CFG_LBC = """BRACOS = {
    # ---------------------------------------------- o eixo `RS` que faltava (§2.12)
    # No paper `h_i = (gamma_i, RS_i)`: desconto E metodo de reward shaping por
    # politica, pareados a moda do NGU (quem explora tem shaping alto e gamma baixo).
    # A implementacao daqui tinha reduzido `H` a gamma sozinho, e a medicao mostrou o
    # preco: as tres politicas otimizavam o MESMO objetivo, entao duas nao eram
    # "miope competente" e "paciente competente" -- eram so piores (V 7,62 e 10,99
    # contra 11,41 da avaliada), e concordavam no argmax em 31,8% dos estados, que e o
    # acaso com tres acoes.
    #
    # Aqui cada cabeca passa a maximizar um retorno diferente e legitimo:
    #   pi0 = (gamma 0,990, shaping 1,0 pela execucao inteira) miope e guiada
    #   pi1 = (gamma 0,995, shaping 0,5 zerando em 25%)        IDENTICA ao 01_ppo
    #   pi2 = (gamma 0,999, shaping 0,0 sempre)                paciente e esparsa
    # O `indice_alvo` continua apontando para pi1: e ela que entra na arena.
    "H_shaping": {
        "gammas": (0.99, 0.995, 0.999),
        "shapings": (1.0, 0.5, 0.0),
        "shaping_fracs": (1.0, 0.25, 0.0),
        "indice_alvo": 1,
    },

    # O mesmo eixo com cobertura maior. Cuidado: o §2.11 mediu que cada politica a mais
    # DOBRA o consumo do orcamento de gradiente (o tronco compartilhado recebe a soma),
    # entao cinco politicas nao sao "mais do mesmo" -- sao um regime diferente.
    "H_shaping_pop5": {
        "n_politicas": 5,
        "gammas": (0.99, 0.993, 0.995, 0.997, 0.999),
        "shapings": (1.0, 0.75, 0.5, 0.25, 0.0),
        "shaping_fracs": (1.0, 0.6, 0.25, 0.1, 0.0),
        "indice_alvo": 2,
    },

    # --------------------------------------------- as ablacoes da Fig. 5 do paper
    # Mesmo espaco de comportamento, escolhido no sorteio. Se empatar com `H_shaping`,
    # o merito estava no espaco e nao no bandit -- e a parte *learnable* do LBC nao fez
    # nada neste dominio, que e um resultado.
    "H_shaping_aleatoria": {
        "gammas": (0.99, 0.995, 0.999),
        "shapings": (1.0, 0.5, 0.0),
        "shaping_fracs": (1.0, 0.25, 0.0),
        "indice_alvo": 1,
        "selecao": "aleatoria",
    },

    # A mesma pergunta sobre o `10_lbc` padrao (H so de gamma), para saber se a
    # conclusao sobre o bandit depende do eixo de `H`.
    "H_gamma_aleatoria": {"selecao": "aleatoria"},

    # "Reducing H" da Fig. 5, levado ao extremo: populacao de UMA. O comportamento vem
    # so de `psi`. Na bancada reduzida este braco EMPATOU com o PPO enquanto a populacao
    # de tres ficava 4 pontos atras -- e o bandit do `10_lbc` escolheu
    # "tau alto, omega ~= pi1" em 96% das iteracoes, que e literalmente este braco.
    "pop1": {"n_politicas": 1, "gammas": (0.995,), "indice_alvo": 0},
}
print(f"braco: {BRACO}  (o controle e o 10_lbc, na MESMA semente)")
for _k, _v in sorted(BRACOS[BRACO].items()):
    print(f"   {_k} = {_v!r}")

"""


ABLACOES = {
    "alphazero": (BRACOS_ABLACAO, BRACO_PADRAO, _PRE_CFG_ABLACAO),
    "muzero": (BRACOS_MUZERO, BRACO_PADRAO_MUZERO, _PRE_CFG_MUZERO),
    "acktr": (BRACOS_ACKTR, BRACO_PADRAO_ACKTR, _PRE_CFG_ACKTR),
    "lbc": (BRACOS_LBC, BRACO_PADRAO_LBC, _PRE_CFG_LBC),
}

#: Braços que a célula de diagnóstico curta **não** roda, por ablação. Só o ACKTR tem —
#: nos outros dois notebooks a célula curta não desliga nada do que os braços mexem.
ABLACOES_SO_TREINO = {
    "acktr": BRACOS_SO_TREINO_ACKTR,
}


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
        "celulas_extra": [{"md": ENSAIO_MD, "codigo": ENSAIO_CODE, "titulo": "Ensaio"}],
        "celulas_pos_veredito": [{"md": BUSCA_MD, "codigo": BUSCA_CODE,
                                  "titulo": "Veredito com busca"}],
        "titulo": "AlphaZero — busca sobre o simulador real",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/agents/alphazero.py"],
        "agente": "AlphaZero",
        "config": "AlphaZeroConfig",
        "resumo": "Snake é determinístico e o simulador é rápido — então a árvore percorre "
                  "o jogo de verdade, sem precisar aprender um modelo do mundo. Abra e rode: "
                  "os padrões são os da versão consertada.\n\n"
                  "**O que mudou, e por quê.** A primeira execução de 5 M passos terminou em "
                  "**10,62** com **86,9% dos episódios morrendo de fome** — o pior número da "
                  "arena. A autópsia achou três defeitos somados, todos medidos em separado "
                  "(§2.27–§2.29 da revisão): o PUCT dava `Q = 0` a um filho não visitado e, "
                  "com o valor positivo que este jogo produz, a busca passava a **confirmar** "
                  "a rede em vez de discordar dela; o alvo de valor não normalizado dominava "
                  "o tronco compartilhado numa razão de gradiente de 71×; e a mesma "
                  "distribuição temperada que escolhia a ação virava o alvo de treino, "
                  "levando a entropia do alvo de 0,66 a 0,015 — rótulo duro, o oposto de "
                  "destilar a distribuição de visitas.\n\n"
                  "Os onze consertos são o padrão desde então. A execução anterior virou o "
                  "braço `sem_correcoes` do `93_alphazero_ablacoes`, que mede quanto cada um "
                  "valeu removendo-os um a um. Ver "
                  "[`docs/BUSCA_DEGENERADA.md`]"
                  "(https://github.com/voaneves/snake-arena/blob/main/docs/BUSCA_DEGENERADA.md).\n\n"
                  "**Duas colunas, de propósito.** A curva oficial mede a rede pura, greedy, "
                  "sem busca — é o que torna as curvas comparáveis, já que a busca gasta 33 "
                  "avaliações de rede por jogada contra 1 do PPO. A célula *Veredito com "
                  "busca* mede o agente como você o levaria para jogar, no mesmo protocolo, "
                  "e grava em `meta[\"com_busca\"]`.",
    },
    {
        "arquivo": "07_muzero.ipynb",
        "celulas_pos_veredito": [{"md": BUSCA_MD, "codigo": BUSCA_CODE,
                                  "titulo": "Veredito com busca"}],
        "titulo": "MuZero — a mesma busca, sobre um modelo aprendido",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/nets/muzero.py", "snakeai/agents/muzero.py"],
        "agente": "MuZero",
        "config": "MuZeroConfig",
        "resumo": "Deve perder para o AlphaZero — o simulador aqui é exato e gratuito. "
                  "O que se mede é quanto custa não tê-lo, e a comparação é limpa porque o "
                  "algoritmo de busca é **o mesmo objeto**: muda só o que a árvore "
                  "percorre.\n\n"
                  "E é exatamente por ser o mesmo objeto que os três defeitos achados na "
                  "primeira execução do AlphaZero (§2.27–§2.29) estavam aqui também: o "
                  "PUCT dando `Q = 0` a filho não visitado — com o valor positivo que este "
                  "jogo produz, a busca só confirmava a rede em vez de discordar dela —, o "
                  "alvo de valor não normalizado dominando o tronco, e a temperatura "
                  "transformando o alvo de política em rótulo duro. Os consertos já são o "
                  "padrão aqui: ao contrário do AlphaZero, o MuZero nunca rodou sob o "
                  "contrato, então não havia execução de controle a preservar. Ver "
                  "[`docs/BUSCA_DEGENERADA.md`]"
                  "(https://github.com/voaneves/snake-arena/blob/main/docs/BUSCA_DEGENERADA.md).\n\n"
                  "**Duas colunas.** A curva oficial mede a política pura de `h`+`f`, sem "
                  "busca. A célula *Veredito com busca* mede o agente como ele de fato "
                  "joga, no mesmo protocolo, e grava em `meta[\"com_busca\"]`.",
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
                  "calibrar é o braço `controle` do `98_acktr_ablacoes` (o antigo "
                  "`98_acktr_kl_nominal`, aposentado quando a ablação passou a cobrir "
                  "os dois eixos numa execução só).",
    },
    {
        "arquivo": "12_acektr.ipynb",
        "titulo": "ACEKTR — os autovalores medidos, não fatorados",
        "modulos": ["snakeai/kfac.py", "snakeai/agents/ppo.py", "snakeai/agents/a2c.py",
                    "snakeai/agents/acktr.py", "snakeai/agents/acektr.py"],
        "agente": "ACEKTR",
        "config": "ACEKTRConfig",
        # 40 avaliações em vez de 20. Não gasta orçamento — `avaliar()` roda num
        # `VecSnake` próprio e não toca no `global_step` — e o contrato registra que em 8
        # das 21 avaliações da primeira execução longa havia um checkpoint anterior
        # melhor que o modelo daquele momento, numa delas por 21,7 pontos. `melhor` é
        # remedido do zero por `avaliar_melhor()` com o mesmo protocolo, então amostrar
        # mais denso acha um pico melhor sem inflar o número. Custa wall-clock.
        "extra_cfg": "    eval_every_steps=125_000,",
        "resumo": "O `08_acktr` com o EK-FAC no lugar do K-FAC — e, desde 01/09, no "
                  "**regime que o paper propõe**.\n\n"
                  "De `A ⊗ G = (U_A ⊗ U_G)(S_A ⊗ S_G)(U_A ⊗ U_G)ᵀ` o K-FAC tira duas "
                  "coisas, e só uma se justifica: uma **base** de autovetores (defensável) "
                  "e uma **escala por eixo** obrigada a ter forma de produto, "
                  "`λ_A(j)·λ_G(i)` (que não vem de lugar nenhum além de ter saído junto). "
                  "O EK-FAC fica com a base e **mede** as escalas. Teorema 2: `s*` é a "
                  "melhor escala diagonal naquela base. Teorema 3: nunca é pior que o "
                  "K-FAC.\n\n"
                  "**Leia isto antes de rodar.** A execução de 01/09 fechou em 71,07 "
                  "contra 89,78 do ACKTR, e a leitura óbvia — o EK-FAC é pior — está "
                  "errada. As duas execuções não estavam pareadas: o `A2CConfig.rollout` "
                  "foi de 16 para 5 no commit `7cdfe2c`, um dia **depois** das três "
                  "sementes gravadas do ACKTR, e o `ACKTRConfig` herdava esse campo. Até "
                  "1,5 M de passos o ACEKTR estava no topo da faixa das três sementes; ele "
                  "separou por baixo logo depois de o shaping decair a zero em 1,25 M, que "
                  "é onde `γλ^T` decide quanto do crédito passa pelo GAE — 76% no "
                  "bootstrap com T=5 contra 40% com T=16. E não faltou passo: o ACEKTR "
                  "acumulou `Σ√KL` de 202 contra 57–73 do ACKTR. Andou 3,6× mais, chegou "
                  "20 pontos abaixo.\n\n"
                  "O que mudou aqui: `rollout=16` restaurado no `ACKTRConfig`; "
                  "`inv_every=50` com `ema_escalas=0.8` (base rara, escalas sempre — o "
                  "eixo de amortização do paper, que estava desligado); e "
                  "`kl_cal_debias=True` com `kl_fator_inicial=15`, porque a média móvel da "
                  "calibração tem constante de tempo de ~50 atualizações num orçamento de "
                  "610 e gastava 8% do treino com o alvo efetivo até 20× maior que o "
                  "pedido — tempo suficiente para a entropia cair de 1,06 para 0,29 e não "
                  "voltar. Ver `docs/EKFAC.md` §3.2, §3.3 e §5.1."},
    {
        "arquivo": "92_muzero_ablacoes.ipynb",
        "titulo": "MuZero — a oscila\u00e7\u00e3o e o peso do desenrolar",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/nets/muzero.py", "snakeai/agents/muzero.py"],
        "agente": "MuZero",
        "config": "MuZeroConfig",
        "param_braco": "muzero",
        "celulas_extra": [{"md": ENSAIO_MZ_MD, "codigo": ENSAIO_MZ_CODE,
                           "titulo": "Ensaio"}],
        "celulas_pos_veredito": [{"md": BUSCA_MD, "codigo": BUSCA_CODE,
                                  "titulo": "Veredito com busca"}],
        "resumo":
            "Este notebook **acrescenta** coisas ao padr\u00e3o, uma por vez — ao contr\u00e1rio do "
            "`93_alphazero_ablacoes`, que remove. A diferen\u00e7a n\u00e3o \u00e9 de estilo: l\u00e1 o padr\u00e3o "
            "\u00e9 a resposta e a pergunta \u00e9 quanto cada conserto valeu; aqui o padr\u00e3o \u00e9 o que "
            "produziu o problema.\n\n"
            "---\n\n"
            "### O que aconteceu em `muzero/unroll5/seed0`\n\n"
            "A execu\u00e7\u00e3o terminou em **49,26** com o melhor ponto em **66,05** (3,75 M) — "
            "16,8 pontos acima do final. \u00c9 tentador chamar isso de m\u00ednimo local. N\u00e3o \u00e9:\n\n"
            "| | 2,5 M | 3,0 M | 3,25 M | 3,75 M | 4,0 M | 5,0 M |\n"
            "|---|---|---|---|---|---|---|\n"
            "| **eval** (rede pura) | 60,44 | 33,25 | 31,74 | **66,05** | 48,05 | 49,26 |\n"
            "| **train** (a busca) | 58,30 | 60,34 | 60,16 | 59,79 | 59,34 | 58,02 |\n"
            "| `perda_pi` | 2,54 | 2,62 | 2,42 | 2,67 | 2,82 | **3,09** |\n\n"
            "O professor est\u00e1 est\u00e1vel. Quem oscila — entre 31,7 e 66,0, num protocolo de "
            "1000 epis\u00f3dios cujo erro padr\u00e3o \u00e9 de **0,9** — \u00e9 o aluno. E `perda_pi` **sobe** "
            "no \u00faltimo ter\u00e7o do or\u00e7amento **enquanto o `lr` desce**, o que descarta passo "
            "grande demais: n\u00e3o \u00e9 o otimizador passando do ponto, \u00e9 o alvo se afastando.\n\n"
            "O modo de falha dos pontos ruins tem assinatura: `fim_fome` \u00e9 **25,6%** no "
            "checkpoint final contra **5,8%** no melhor. A rede pura perde o impulso de ir "
            "atr\u00e1s da ma\u00e7\u00e3 no fim de jogo, e a busca resgata (`frac_fome` fica em ~0% no "
            "treino). Isso \u00e9 falha de **destila\u00e7\u00e3o**, n\u00e3o de busca.\n\n"
            "### A hip\u00f3tese principal, e por que `unroll=10` sozinho vai para o lado errado\n\n"
            "`perda_pi` \u00e9 uma **soma crua** sobre `K+1` termos: o passo 0, que sai de "
            "`f(h(o))` — a observa\u00e7\u00e3o **real**, o \u00fanico caminho que `politica()` usa na "
            "avalia\u00e7\u00e3o — e `K` passos imaginados, que saem de `f(g^k(...))`. Nenhum peso "
            "separa os dois. Medindo (`tools/diag_unroll.py`):\n\n"
            "| `unroll` | soma crua | com `normaliza_unroll` |\n"
            "|---|---|---|\n"
            "| 2 | 29,7% | 46,0% |\n"
            "| **5** (o padr\u00e3o) | **14,5%** | 46,0% |\n"
            "| **10** | **11,0%** | 55,2% |\n\n"
            "Ou seja: 85% do gradiente de pol\u00edtica treina um caminho que a m\u00e9trica oficial "
            "nunca percorre, e **aumentar o desenrolar dilui ainda mais o \u00fanico termo que "
            "produz o n\u00famero do contrato**. O paper n\u00e3o faz isso. O Ap\u00eandice G lista "
            "**duas** escalas de gradiente, textualmente: *\"we scale the loss of each "
            "head by 1/K\"* e *\"we scale the gradient at the start of the dynamics "
            "function by 1/2\"*. O reposit\u00f3rio tinha a segunda e n\u00e3o a primeira.\n\n"
            "Vale notar que a leitura literal da prosa \u2014 escalar os `K+1` termos, "
            "passo 0 inclu\u00eddo \u2014 n\u00e3o mudaria nada aqui: sob Adam, dividir a perda "
            "inteira por uma constante \u00e9 quase um no-op, porque o segundo momento "
            "normaliza; sobraria s\u00f3 um `clipnorm` mordendo menos. O que muda a "
            "**fatia** do passo 0 \u00e9 deix\u00e1-lo fora da escala, que \u00e9 o que o "
            "pseudoc\u00f3digo publicado faz e o que `normaliza_unroll` implementa. "
            "Ver \u00a72.31.\n\n"
            "### Como rodar, em ordem de prioridade\n\n"
            "Cada bra\u00e7o custa ~7 h. Escolha o `BRACO`, rode o ensaio (2 min — ele confere "
            "que `frac_pi_0` saiu no valor esperado, que \u00e9 o jeito de pegar uma chave que "
            "n\u00e3o chegou no `cfg`) e depois o treino. Compare com `07_muzero` **na mesma "
            "semente**; o `sufixo_variante` mantém os bra\u00e7os separados na arena.\n\n"
            "1. **`normaliza_unroll`** \u2014 a hip\u00f3tese principal, e a **\u00fanica coisa "
            "que o Ap\u00eandice G manda fazer e o reposit\u00f3rio n\u00e3o faz**. \u00c9 de gra\u00e7a: "
            "mesmo custo por atualiza\u00e7\u00e3o. Se voc\u00ea s\u00f3 tem uma execu\u00e7\u00e3o, \u00e9 esta.\n"
            "2. **`reuso_do_paper`** ou os **`reanalise_*`** \u2014 8 \u00e9pocas \u00d7 256 sobre 1.024 "
            "passos novos d\u00e3o **2,0 amostras por estado**. O paper usa 0,1 no MuZero puro "
            "e sobe para 2,0 s\u00f3 no **Reanalyse** (Ap\u00eandice H) \u2014 que refaz a busca com a "
            "rede atual em 80% das atualiza\u00e7\u00f5es e usa rede alvo para o bootstrap. "
            "Estamos no re\u00faso do Reanalyse **sem** o Reanalyse (§2.32). Duas sa\u00eddas: "
            "`reuso_do_paper` volta ao re\u00faso do MuZero puro, de gra\u00e7a, pagando em "
            "or\u00e7amento de gradiente; os `reanalise_*` ficam com o re\u00faso alto e trazem o "
            "alvo fresco, pagando em busca \u2014 **1,32\u00d7 a 1,57\u00d7** de tempo de parede, "
            "medido em `tools/diag_reanalise.py`.\n"
            "   O custo \u00e9 **sublinear na fra\u00e7\u00e3o**, e isso decide qual rodar: as buscas "
            "s\u00e3o em lote, ent\u00e3o o n\u00famero de la\u00e7os de \u00e1rvore em Python \u00e9 `epochs_por_iter` "
            "qualquer que seja a fra\u00e7\u00e3o \u2014 s\u00f3 a largura do lote cresce. Numa **GPU**, "
            "onde o la\u00e7o domina e a largura \u00e9 quase de gra\u00e7a, `reanalise_80` custa quase "
            "o mesmo que `reanalise_25`: v\u00e1 direto ao n\u00famero do paper. `reanalise_sims` \u00e9 "
            "o bot\u00e3o de custo l\u00e1, e \u00e9 desvio do paper.\n"
            "3. **`unroll10_normalizado`** \u2014 o desenrolar longo sem diluir o passo 0.\n\n"
            "**`coef_valor_1` caiu de prioridade, e a corre\u00e7\u00e3o \u00e9 minha.** Eu o propus "
            "argumentando que `perda_v \u2248 0,19` em `symlog` vira uma banda de "
            "`[6,7; 17,5]` na escala real, e que valor ruidoso produz alvo de visitas "
            "ruidoso. O Ap\u00eandice H diz o contr\u00e1rio: o paper baixa o alvo de valor "
            "**para 0,25** contra 1,0 de pol\u00edtica e recompensa, e diz por qu\u00ea \u2014 "
            "*\"avoid overfitting of the value function\"*. Ou seja, 0,25 j\u00e1 \u00e9 o valor "
            "do paper, e subir para 1,0 \u00e9 ir contra ele. O bra\u00e7o continua porque a "
            "hip\u00f3tese \u00e9 test\u00e1vel; deixou de ser a segunda coisa a rodar.\n\n"
            "Os bra\u00e7os `unroll10` e `sims32` t\u00eam **previs\u00e3o pr\u00e9-registrada de n\u00e3o "
            "ajudar** (est\u00e3o no dicion\u00e1rio com a previs\u00e3o escrita ao lado). Rode-os se "
            "quiser o registro da falsifica\u00e7\u00e3o; n\u00e3o os rode esperando ganho.\n\n"
            "**A curva oficial mede a rede pura**, greedy, sem busca. Olhe o `[eval]`, n\u00e3o "
            "o `score` do log de treino, que \u00e9 o da busca — a diferen\u00e7a entre os dois \u00e9 "
            "exatamente o v\u00e3o que este notebook tenta fechar.",
    },
    {
        "arquivo": "93_alphazero_ablacoes.ipynb",
        "titulo": "AlphaZero — quanto cada conserto valeu",
        "modulos": ["snakeai/search/dinamica.py", "snakeai/search/mcts.py",
                    "snakeai/agents/alphazero.py"],
        "agente": "AlphaZero",
        "config": "AlphaZeroConfig",
        "param_braco": "alphazero",
        "celulas_extra": [{"md": ENSAIO_MD, "codigo": ENSAIO_CODE, "titulo": "Ensaio"}],
        "celulas_pos_veredito": [{"md": BUSCA_MD, "codigo": BUSCA_CODE,
                                  "titulo": "Veredito com busca"}],
        "resumo":
            "Este notebook **remove** coisas do padrão, uma por vez. Não é aqui que se roda "
            "o AlphaZero — o agente oficial é o `06_alphazero`, e o padrão dele já é a "
            "versão consertada. É a mesma inversão que o ACKTR sem calibrar sofreu quando "
            "a calibração da região de confiança venceu a medição e virou o padrão do `08`: "
            "o braço que sobrevive é o *sem*.\n\n"
            "**Como rodar.** Escolha o `BRACO`, rode o ensaio (2 min, pega o que é "
            "catastrófico antes de você gastar ~8 h) e depois o treino. Compare com "
            "`06_alphazero` **na mesma semente**; o `sufixo_variante` mantém os braços "
            "separados na arena. E olhe o `[eval]`, que mede a rede pura — não o `score` do "
            "log de treino, que é o da busca.\n\n"
            "---\n\n"
            "### De onde vieram os consertos\n\n"
            "A primeira execução de 5 M passos terminou com a política pura em **10,62** "
            "(pico 13,03 em 3,0 M), **86,9% dos episódios por fome** e 0% de tabuleiro "
            "cheio — o pior número da arena, com folga. Ela está aqui como o braço "
            "`sem_correcoes` e na arena como `alphazero/sims32_sem_correcoes/seed0`. A "
            "autópsia achou três defeitos somados, cada um medido em separado:\n\n"
            "| § | o defeito | a evidência |\n"
            "|---|---|---|\n"
            "| **2.27** | o PUCT dá `Q = 0` a um filho não visitado — a convenção do "
            "AlphaZero, correta onde o valor é uma `tanh` em `[-1,1]`. Aqui a cabeça é "
            "linear e o `valor_raiz` medido vai de 0,26 a **3,5**, então o bônus "
            "`c_puct·P·√N` só cobre a diferença onde o prior já é alto: a busca passa a "
            "**confirmar** a rede em vez de discordar dela | somar uma constante ao valor "
            "da folha — sem mudar o ranking de estado nenhum — leva o score de 21,70 (100% "
            "colisão) a **0,00** (100% fome) |\n"
            "| **2.28** | o alvo de valor não é normalizado e domina o tronco "
            "compartilhado | na execução real, `perda_v/perda_pi` = **57,6×** depois de "
            "4 M — a `perda_pi` desabou para 0,016 enquanto a `perda_v` **subiu** de 0,34 "
            "para 1,0 e nunca convergiu. No `|z|` real, `symlog` leva o gradiente de 71× "
            "para 14×, e `vf_coef=0,5` para 7,0× |\n"
            "| **2.29** | a mesma π temperada escolhe a ação **e** vira o alvo de treino; "
            "com τ = 0,25 as visitas são elevadas à quarta potência | entropia do alvo de "
            "0,66 para **0,015** nas contagens de visita reais — da metade do treino em "
            "diante a rede aprende rótulo duro, que é o oposto de destilar a distribuição "
            "de visitas |\n\n"
            "Mais dois botões menores que entraram junto: o orçamento de gradiente "
            "(~4.900 atualizações contra as ~38.300 do PPO) e o decaimento de `lr` — este "
            "era o único agente do repositório sem, e a execução antiga oscilava entre 9,6 "
            "e 12,5 nos últimos 2 M, com o `best` 2,4 pontos acima do `last`.\n\n"
            "### Os braços\n\n"
            "Três deles removem um **mecanismo inteiro** e respondem a pergunta em três "
            "execuções em vez de onze — é por onde começar:\n\n"
            "* `sem_conserto_da_busca` — desliga `fpu` e `q_normalizado` (§2.27)\n"
            "* `sem_conserto_do_tronco` — desliga `valor_symlog` e volta `vf_coef=1` (§2.28)\n"
            "* `sem_conserto_do_alvo` — desliga `temp_alvo` e `temp_passos` (§2.29)\n"
            "* `sem_correcoes` — desliga tudo; é o agente de 10,62\n\n"
            "Os outros isolam botão a botão dentro de cada mecanismo. E há dois que a "
            "literatura sugere e que **não** entraram no padrão: `busca64` (dobra o tempo "
            "de parede e a medição diz que compra ~1,3 plies de profundidade, não "
            "horizonte) e `gamma_995` (alinhamento com o resto do repositório, não "
            "conserto).\n\n"
            "Tudo em [`docs/BUSCA_DEGENERADA.md`]"
            "(https://github.com/voaneves/snake-arena/blob/main/docs/BUSCA_DEGENERADA.md), "
            "com os scripts que regeneram as tabelas (`tools/diag_busca.py`, "
            "`tools/diag_balanco_perdas.py`).",
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
        "arquivo": "98_acktr_ablacoes.ipynb",
        "titulo": "ACKTR — de onde vem o estouro da região de confiança",
        "modulos": ["snakeai/kfac.py", "snakeai/agents/ppo.py", "snakeai/agents/a2c.py",
                    "snakeai/agents/acktr.py"],
        "agente": "ACKTR",
        "config": "ACKTRConfig",
        "param_braco": "acktr",
        "celulas_extra": [
            {"md": VARREDURA_MD, "codigo": VARREDURA_CODE,
             "titulo": "Varredura de kl_max"},
            {"md": DIAG_KL_MD, "codigo": DIAG_KL_CODE,
             "titulo": "Braços (secundário)"},
        ],
        "resumo":
            "**Se você veio rodar uma vez e ir embora: o braço padrão já é o certo.** "
"O dropdown abre em `definitiva`, que é a união do que tem medição — o "
"`rollout = 16` restaurado, a calibração da região de confiança debiasada com "
"prior 15, e 40 avaliações em vez de 20. Aperte *Executar tudo* e ignore o "
"resto desta página. A lista do que ficou **de fora** desse braço, e por quê, "
"está no comentário dele na célula de parâmetros — ela é metade do valor do "
"braço, porque três coisas terem medição e as outras não é o resultado.\n\n"
            "---\n\n"
            "O resto do notebook é a investigação que produziu esses três itens.\n\n"
            "**A medição principal é a célula *Varredura de `kl_max`*, e ela não usa "
            "braço nenhum.** A pergunta certa não é *qual braço reduz a razão* — "
            "isso é uma comparação de sete vias, de uma estatística de cauda pesada, com "
            "mediadores que agem em silêncio. É **se a KL entregue responde à KL "
            "pedida**, e isso se mede num agente só, varrendo `kl_max` por duas décadas: "
            "inclinação ~1 em log–log significa que a região de confiança funciona, ~0 "
            "significa que existe um **piso** e `kl_max` é decorativo acima dele.\n\n"
            "Os braços vêm depois, como secundários, e a curva oficial (o treino de 5 M) "
            "só depois deles.\n\n"
            "---\n\n"
            "### O problema\n\n"
            "`escala_kl` devolve `η = √(2·kl_max / Δᵀ∇)`: o passo tal que **uma** "
            "atualização `ηΔ` induz `kl_max`. A KL medida sai de **4,4× a 12,4×** disso "
            "(quintis de uma execução de 5 M). A revisão atribui a diferença à Fisher "
            "aproximada subestimar a curvatura — e **essa atribuição é a premissa do "
            "ACEKTR**, que existe para corrigi-la com autovalores exatos.\n\n"
            "Há dois outros suspeitos no mesmo lugar, e nenhum é a Fisher:\n\n"
            "1. **O momento.** `η` é atribuído como `learning_rate` de um "
            "`SGD(momentum=0.9, nesterov=True)`. Com momento, o deslocamento em regime é "
            "até `ηΔ/(1−μ) = 10·ηΔ`, e a KL vai com o **quadrado** do passo. Os "
            "4,4×–12,4× medidos correspondem a uma amplificação de 2,1×–3,5× — "
            "exatamente o que um momento de 0,9 sobre gradientes parcialmente "
            "correlacionados produz. O `baselines` original escreve "
            "`MomentumOptimizer(lr·(1−momentum), momentum)`, e o fator `(1−μ)` está lá "
            "justamente para cancelar isto.\n"
            "2. **O `clipnorm`.** O `max_grad_norm = 0,5` herdado do PPO é aplicado pelo "
            "Keras **por variável, dentro do `apply_gradients`** — sobre a direção já "
            "pré-condicionada. No `baselines` o clip só existe no caminho SGD *cold* e "
            "nunca toca a direção natural. Direções naturais com `damping = 1e-2` têm "
            "norma bem maior que as cruas, então o clip provavelmente age quase sempre: "
            "distorce a razão entre camadas que é a razão de ser do K-FAC.\n\n"
            "### O que decide entre eles\n\n"
            "O padrão temporal **não** decide: a revisão nota que o estouro é maior no "
            "começo e usa isso a favor da Fisher (\"o erro encolhe conforme a média "
            "móvel dos fatores amadurece\"), mas gradientes sucessivos também são muito "
            "mais correlacionados no começo do treino, quando a política se move numa "
            "direção só. Os dois mecanismos preveem a mesma curva. **Só a intervenção "
            "separa.**\n\n"
            "`kl_calibrado = False` em **todos** os braços, e isso não é detalhe: ligado, "
            "ele mede a razão e pede `kl_max/c`, de modo que a KL entregue converge para "
            "o alvo *qualquer que seja a causa*. Medir com ele ligado responderia sempre "
            "\"está calibrado\".\n\n"
            "### Como ler o resultado\n\n"
            "| se a razão `KL/alvo` cair para ~1 em… | a causa é… |\n"
            "|---|---|\n"
            "| `sem_momento` | o momento — mas este braço joga fora a redução de variância |\n"
            "| `momento_descontado` | o momento, e **este é o conserto certo** (o do `baselines`) |\n"
            "| `sem_clip` | o `clipnorm` sobre a direção natural |\n"
            "| `sem_momento_sem_clip` | os dois juntos |\n"
            "| **nenhum** | a Fisher aproximada — a §2 sobrevive e o ACEKTR mantém a premissa |\n\n"
            "O braço padrão é `momento_descontado` porque é o único que, se estiver "
            "certo, é também o conserto que se quer manter. Três sementes oficiais do "
            "ACKTR estão gravadas: **nada aqui vira padrão sem medição.**",
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
        "arquivo": "90_lbc_populacao.ipynb",
        "titulo": "LBC — a população vale o que custa?",
        "modulos": ["snakeai/bandit.py", "snakeai/agents/ppo.py",
                    "snakeai/agents/lbc.py"],
        "agente": "LBC",
        "config": "LBCConfig",
        "param_braco": "lbc",
        "resumo":
            "Não é aqui que se roda o LBC — o agente oficial é o `10_lbc`. Aqui se mede a "
            "peça de que ele mais depende e que a primeira execução corrigida mostrou ser "
            "a mais frágil: a **população de políticas**.\n\n"
            "**Como rodar.** Escolha o `BRACO` e compare com `10_lbc` **na mesma "
            "semente** — o `sufixo_variante` mantém os braços separados na arena.\n\n"
            "---\n\n"
            "### A pergunta\n\n"
            "O mapeamento híbrido do §4.1 mistura `N` políticas para obter um "
            "comportamento que nenhuma delas produz sozinha. Isso só funciona se as `N` "
            "forem **diferentes e cada uma competente**. Medido no modelo treinado da "
            "execução anterior, nenhuma das duas condições valia:\n\n"
            "| | π0 (γ=0,99) | π1 (γ=0,995, a avaliada) | π2 (γ=0,999) |\n"
            "|---|---|---|---|\n"
            "| entropia | 0,109 | 0,152 | 0,014 |\n"
            "| valor médio `V_i` | 7,62 | 11,41 | 10,99 |\n\n"
            "As três concordavam na ação de argmax em **31,8%** dos estados — com três "
            "ações, o acaso é 33%. Não era uma população de pares com estratégias "
            "distintas: era uma cabeça subtreinada, uma colapsada e a avaliada. E a "
            "mistura uniforme, comparada com \"só π1\", mudava a entropia (0,65 contra "
            "0,21) mas mantinha **95,1% do mesmo argmax** — ou seja, mudava o quanto a "
            "cobra hesita, não para onde ela vai. Isso é ε-greedy com passos extras, que "
            "é exatamente o que o mapeamento híbrido existe para superar.\n\n"
            "O meta-controlador já tinha chegado nessa conclusão: em **96,1%** das "
            "iterações ele escolheu o braço `τ∈[2,4] · ω≈π1` — *\"use π1 sozinha, quase "
            "gulosa\"*, que é o caso degenerado do Agent57 e literalmente o braço "
            "`pop1` daqui.\n\n"
            "### A hipótese que este notebook testa\n\n"
            "Que o problema era o **eixo**, não a ideia. O paper usa "
            "`h_i = (γ_i, RS_i)` — desconto **e** um método de *reward shaping* por "
            "política, pareados à moda do NGU. Esta implementação tinha reduzido `H` a γ "
            "sozinho (`docs/LBC.md` §2.2), então as três cabeças otimizavam o **mesmo** "
            "objetivo sobre um tronco compartilhado, e o que as separava era sobretudo "
            "ruído de treino. O braço `H_shaping` devolve o segundo eixo usando o "
            "potencial que o `VecSnake` já calcula.\n\n"
            "### Como ler o resultado\n\n"
            "| se… | a leitura é… |\n"
            "|---|---|\n"
            "| `H_shaping` > `10_lbc` | o eixo era o problema; `H` reduzido a γ não "
            "sustenta um mapeamento híbrido |\n"
            "| `pop1` ≥ `H_shaping` | a população não paga o que custa **neste domínio** "
            "— e o LBC aqui é o `ψ` sozinho |\n"
            "| `H_shaping_aleatoria` ≈ `H_shaping` | o mérito é do espaço de "
            "comportamento, não do bandit: a parte *learnable* não fez nada |\n"
            "| `H_shaping_pop5` < `H_shaping` | mais políticas custam mais do que "
            "rendem — cada uma dobra o consumo do orçamento de gradiente (§2.11) |\n\n"
            "Os dois números novos no registro respondem direto, sem forense de "
            "checkpoint: **`divergencia_populacao`** (KL médio entre as cabeças) e "
            "**`acordo_argmax`** (fração de estados em que todas concordam). Perto de 1 "
            "é população degenerada; perto de `1/3` é acaso, e aí elas estão resolvendo "
            "problemas diferentes em vez de o mesmo problema de jeitos diferentes. "
            "`valor_relativo_pior` diz se as outras cabeças são competentes ou só piores.",
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
        lista, padrao, pre_bracos = ABLACOES[spec["param_braco"]]
        _ns = {}
        exec(pre_bracos.split("print(")[0], _ns)                # noqa: S102
        if set(_ns["BRACOS"]) != set(lista):
            raise ValueError(
                f"a lista e o dicionário de braços do {spec['param_braco']} divergiram: "
                f"só na lista {sorted(set(lista) - set(_ns['BRACOS']))}, "
                f"só no dicionário {sorted(set(_ns['BRACOS']) - set(lista))}")
        if padrao not in lista:
            raise ValueError(f"braço padrão {padrao!r} não está na lista")
        # a mesma conferência para os braços que a célula de diagnóstico pula: um nome
        # errado ali não estoura, só faz o braço voltar a ser medido com a calibração
        # desligada e virar um duplicado silencioso do `controle` na tabela
        so_treino = ABLACOES_SO_TREINO.get(spec["param_braco"], ())
        if set(_ns.get("BRACOS_SO_TREINO", ())) != set(so_treino):
            raise ValueError(
                f"os braços só-de-treino do {spec['param_braco']} divergiram: "
                f"gerador {sorted(so_treino)} × notebook "
                f"{sorted(_ns.get('BRACOS_SO_TREINO', ()))}")
        if set(so_treino) - set(lista):
            raise ValueError(
                f"braço só-de-treino fora da lista: {sorted(set(so_treino) - set(lista))}")
        braco_param = ('\n' + f'BRACO = "{padrao}"  # @param ['
                       + ", ".join(f'"{k}"' for k in lista) + "]")
        pre = pre_bracos
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
        #: Células que entram DEPOIS do veredito — hoje só a coluna com busca, dos agentes
        #: que buscam na hora de agir. Ela lê `resultado` e `registro`, então tem que vir
        #: depois do veredito e antes do zip.
        *[c for extra in spec.get("celulas_pos_veredito", [])
          for c in (_md(extra["md"]), _code(extra["codigo"], extra.get("titulo")))],
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
        # `ensure_ascii=True` de propósito: o `.ipynb` sai **100% ASCII**, com os acentos
        # e os símbolos guardados como escapes JSON (`\u00e3`) em vez de bytes crus. O
        # conteúdo é idêntico — todo leitor de JSON desfaz o escape —, mas o arquivo passa
        # a atravessar intacto qualquer ferramenta que erre a codificação no caminho.
        # Foi o que aconteceu com o `06` ao subir para o Kaggle: o arquivo no repositório
        # estava correto (UTF-8 válido, sem BOM, sem mojibake) e chegou lá quebrado. Um
        # arquivo sem byte acima de 0x7F não tem como ser mal decodificado. Custa +5% de
        # tamanho, e estes arquivos são gerados — ninguém lê o diff deles.
        novo = json.dumps(nb, ensure_ascii=True, indent=1)

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
