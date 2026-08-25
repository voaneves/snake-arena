# Procedência: qual código produziu cada número

Este benchmark foi medido ao longo de três semanas, com o pacote mudando entre execuções.
Vinte e quatro execuções carregam cinco versões diferentes do código. Um artigo que publica
uma tabela dessas precisa responder duas perguntas antes que um revisor as faça: **qual
código produziu cada linha**, e **quais diferenças entre esses códigos tocam num número**.

Este documento é o método usado para responder às duas, e o registro dos casos concretos.
Ele é material direto para as seções de metodologia e de limitações.

## Por que não é `git rev-parse`

Quase todas as execuções nascem no Kaggle ou no Colab, onde não existe clone git: o
`meta["commit"]` sai `"desconhecido"` e a curva fica sem procedência. O substituto é a
**assinatura do pacote**: o `tools/gerar_notebooks.py` concatena o fonte dos módulos que
aquele notebook embute, tira o SHA-256 e injeta o resultado no bloco gerado como a constante
`ASSINATURA_PACOTE`. Em tempo de execução, `record._ambiente()` lê essa constante do
namespace do notebook e a grava em `meta["assinatura_pacote"]`.

Duas propriedades que precisam ficar explícitas, porque são contraintuitivas:

**Ela não é global.** Cada notebook embute uma fatia diferente do pacote, então `01_ppo` e
`08_acktr` nunca tiveram a mesma assinatura, mesmo no mesmo commit. Hoje o PPO grava
`40448b19b28116da` e o ACKTR `ca21410bf88c1c65`. Comparar assinaturas entre notebooks
diferentes não significa nada.

**Ela muda quando qualquer módulo embutido muda**, inclusive módulos que não participam do
treino. O `95_a2c_orcamento_esparso` embute quinze arquivos, entre eles `snakeai/plot.py`,
que é módulo de relatório. Uma troca de rótulo de coluna em `plot.py` muda a assinatura de
todos os notebooks sem tocar em um único número.

## O inventário

| execução | assinatura | atualizações | horas | s/atualização |
|---|---|---:|---:|---:|
| ppo/resnet_small/seed{0,1,2} | `40448b19b28116da` | 38.273 | 0,83 | 0,08 |
| ppo/resnet_small_esparso/seed{0,1,2} | — | ~2.424 | 0,41 | 0,62 |
| ppo/resnet_small_fome_esparso/seed{0,1,2} | — | ~2.424 | 0,40 | 0,60 |
| acktr/resnet_small/seed{0,1,2} | `ca21410bf88c1c65` | ~610 | 0,51 | 3,03 |
| acktr/resnet_small+kl0.002/seed0 | `ca21410bf88c1c65` | ~610 | 0,51 | 3,01 |
| acktr/resnet_small+kl_nominal+kl0.002/seed0 | `ca21410bf88c1c65` | ~610 | 0,51 | 3,01 |
| acktr/resnet_small_regua_antiga/seed0 | — | ~610 | 0,66 | 3,88 |
| a2c/resnet_small/seed{0,1,2} | `df6c8eb2b2ca2f58` | 1.954 | 0,31 | 0,57 |
| a2c/resnet_small_esparso/seed0 | `782a8b8aa4af004f` | 611 | 0,78 | **4,58** |
| a2c/resnet_small_esparso/seed{1,2} | `df6c8eb2b2ca2f58` | 611 | 0,25 | **1,47** |
| dqn/base/seed{0,1} | `75aa8ceb896d2cbc` | 38.908 | 1,85 | 0,17 |
| dqn/base_antigo/seed0 | — | — | 2,35 | — |

O travessão marca execuções anteriores ao mecanismo de assinatura. Todas elas já são
`comparable=False` por outros motivos, **exceto** `ppo/resnet_small_esparso`, que compete e
cuja procedência precisa ser reconstruída pelo commit em que os dados foram acrescentados.

## A fronteira de agosto: por que **todas** as assinaturas mudaram de uma vez

Três algoritmos entraram no repositório em sequência — LBC, SOAP e ACEKTR —, e cada um deles
acrescentou um construtor a `snakeai/nets/registry.py`. Esse módulo está no **núcleo** que o
gerador injeta em todo notebook, então a assinatura dos dezessete mudou três vezes, mesmo nos
notebooks que não usam nenhum dos construtores novos.

Isso precisa estar escrito porque um revisor vai perguntar, e a resposta certa é a que
tranquiliza: **a mudança é inerte**. O que entrou foram funções novas — nenhuma linha de
`build_actor_critic`, do `VecSnake`, do `evaluate` ou do `record` foi tocada. Uma execução de
PPO antes e depois da fronteira roda o mesmo código; o que mudou foi o hash do arquivo que o
contém.

Duas mudanças **não** são inertes e ficam registradas aqui:

| mudança | efeito num número |
|---|---|
| `env/render.py` passou a chamar `apos_passo` (§3.6 da revisão) | **nenhum número da arena**. Ela afeta só o GIF — mas todo GIF de DreamerV3 anterior à fronteira mostra uma política com o latente congelado |
| `agents/acktr.py` extraiu `_cria_precondicionador` do `__init__` | nenhum: o K-FAC continua sendo construído com os mesmos argumentos. É refatoração pura, e `test_ekfac.py` confere que o ACKTR e o ACEKTR só diferem nesse método |

O procedimento de auditoria da seção seguinte continua valendo sem alteração: a assinatura é
reprodutível a partir de qualquer commit, e é ela — não a data — que diz o que rodou.

## O método de auditoria

A assinatura é reprodutível a partir de qualquer commit: basta reconstruir a concatenação
com o fonte daquele commit e refazer o hash. Isso transforma "qual código rodou isto" de
memória em consulta.

```bash
python - <<'PY'
import subprocess, sys
sys.path.insert(0, "tools")
import gerar_notebooks as G

import json
nb = json.load(open("notebooks/95_a2c_orcamento_esparso.ipynb"))
MODULOS = nb["metadata"]["snake_arena"]["gerado_de"]   # a fatia daquele notebook

def git(*a):
    return subprocess.run(["git", "--no-optional-locks", *a],
                          capture_output=True, text=True, encoding="utf-8").stdout

def assinatura_em(commit):
    partes = ["from __future__ import annotations"]
    for m in MODULOS:
        txt = git("show", f"{commit}:{m}")
        if not txt:
            return None                      # o módulo não existia nesse commit
        partes.append(G._limpa(txt, m))
    return G._hash("\n\n".join(partes))

for linha in git("log", "--format=%h %s", "-30").strip().splitlines():
    curto, *msg = linha.split()
    print(f"{curto}  {assinatura_em(curto)}  {' '.join(msg)[:50]}")
PY
```

Rodado para `782a8b8aa4af004f`, ele localiza a faixa `187285e..b2ed0fd` — e mostra que a
assinatura passou a `df6c8eb2b2ca2f58` exatamente em `7cdfe2c`. A janela a inspecionar
deixa de ser "três semanas de commits" e vira um diff só.

## Caso 1 — A assinatura mudou no meio de um trio

`a2c/resnet_small_esparso` tem a semente 0 numa assinatura e as sementes 1 e 2 noutra. A
pergunta é se o trio ainda é um trio.

Dos quinze módulos que o notebook embute, **três** mudaram na janela `b2ed0fd..7cdfe2c`:

| módulo | o que mudou | toca num número? |
|---|---|---|
| `plot.py` | rótulos de coluna e o parágrafo explicativo da tabela | não — é módulo de relatório, chamado uma vez no fim para desenhar `curva.png` |
| `ppo.py` | escalares passados como `tf.constant` em vez de float Python | não — mesmo valor entra no grafo; o que muda é o retracing |
| `a2c.py` | o mesmo, mais `rollout` padrão de 16 para 5 e o `A2CConfig.esparso()` novo | não neste braço — ver abaixo |

A mudança de padrão `rollout` 16 → 5 parece perigosa e não é: a execução antiga pegava 16 do
padrão, e `A2CConfig.esparso()` fixa 16 explicitamente. O teste decisivo é comparar campo a
campo o que `esparso()` produz hoje contra o `config` gravado na execução de 20/08:

```bash
python - <<'PY'
import ast, json

def defaults(path, cls):
    for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            out = {}
            for s in n.body:
                if isinstance(s, ast.AnnAssign) and s.value is not None:
                    try:
                        out[s.target.id] = ast.literal_eval(s.value)
                    except Exception:
                        out[s.target.id] = ast.unparse(s.value)
            return out
    return {}

hoje = {**defaults("snakeai/agents/base.py", "AgentConfig"),
        **defaults("snakeai/agents/ppo.py", "PPOConfig"),
        **defaults("snakeai/agents/a2c.py", "A2CConfig"),
        "rollout": 16, "sufixo_variante": "esparso"}     # o que esparso() fixa
run = json.load(open("runs/a2c/resnet_small_esparso/seed0/history.json"))["config"]
print([(k, run[k], hoje[k]) for k in set(run) & set(hoje) if run[k] != hoje[k]] or "idênticos")
PY
```

Zero divergências. **O trio é um trio**, e as três sementes entram na mesma média.

Uma observação que um revisor vai fazer, então melhor antecipar: a semente na assinatura
antiga é a que marcou **mais** (55,47 contra 53,60 e 47,59). Isso é variação entre sementes —
o desvio padrão do braço é 4,11 — e não efeito de código, porque as três diferenças acima
são numericamente neutras por inspeção.

## Caso 2 — O tempo de parede não atravessa a fronteira

O que a correção de retracing muda é o custo, e o `a2c/resnet_small_esparso` mede isso num
experimento controlado que ninguém planejou: **mesma configuração, mesmo número de
atualizações, mesmo lote, só o código diferente**.

| semente | assinatura | atualizações | s/atualização | horas |
|---|---|---:|---:|---:|
| seed0 | `782a8b8aa4af004f` | 611 | **4,58** | 0,78 |
| seed1 | `df6c8eb2b2ca2f58` | 611 | 1,53 | 0,26 |
| seed2 | `df6c8eb2b2ca2f58` | 611 | 1,40 | 0,24 |

Três segundos por atualização de recompilação de grafo. A causa é conhecida e está em
`REVISAO_ALGORITMOS.md` §2.6: escalares Python entram na assinatura da `tf.function`, e o
`ent_coef` muda a cada iteração, então cada iteração retraçava o grafo inteiro. O
`reduce_retracing=True` relaxa formas de tensor, não escalares Python.

**Consequência para a tabela:** a coluna `horas` é comparável entre execuções da mesma
assinatura e não é comparável através da fronteira. Isso vale dentro do próprio trio esparso
do A2C, e vale entre o A2C e o PPO/ACKTR, cujas execuções são anteriores à correção. Onde o
artigo usar tempo de parede — e o gráfico `arena_tempo` usa —, a ressalva tem que estar na
legenda.

## Caso 3 — A pasta e a identidade podem discordar

`load_all()` agrupa execuções por `(algo, variant, seed)` lido de dentro do `history.json`,
**não** pelo caminho; `plot.py` agrega as sementes da arena pelo mesmo par. Uma execução
gravada com um `variant` e movida para outra pasta à mão fica com a identidade da pasta
antiga: some do grupo onde o caminho diz que está e reaparece, em silêncio, dentro do grupo
de outra configuração.

Aconteceu duas vezes neste repositório:

* **`a2c/resnet_small_esparso/seed0`** — rodada antes de `A2CConfig.esparso()` existir, saiu
  com `sufixo_variante=""`, virou `a2c/resnet_small/seed0` e foi renomeada à mão. A colisão
  só apareceria quando o braço denso — que é `a2c/resnet_small/seed0` por direito — fosse
  medido, e apareceria como uma mediana de duas sementes onde só existe uma, com orçamentos
  de gradiente diferentes.
* **`dqn/base_antigo/seed0`** — a execução pré-correção, deslocada quando o DQN corrigido
  tomou o lugar em `dqn/base/`. Continuava se identificando como `dqn/base/seed0`, ou seja,
  uma execução `comparable=False` compartilhando identidade com o resultado oficial.

Nos dois casos o `history.json` foi corrigido no lugar: `variant`, `config.sufixo_variante`
e os caminhos gravados em `meta["artefatos"]`. Curva, `final` e proveniência não foram
tocados.

O que impede a terceira vez é um teste, `test_every_recorded_run_sits_where_its_identity_says`
em `tests/test_record.py`, que varre `runs/` e exige
`runs/<algo>/<variant>/seed<N>/history.json` igual à tripla de dentro do arquivo. Ele pegou o
caso 2 sozinho, cinco dias depois de ter sido escrito para o caso 1.

## O que isto obriga no artigo

1. **Publicar a assinatura por execução**, não só o algoritmo e a semente. Ela já está em
   `meta["assinatura_pacote"]` de cada `history.json`.
2. **Declarar as três execuções cuja procedência é anterior ao mecanismo** — o trio
   `ppo/resnet_small_esparso` é o único caso que compete na arena sem assinatura.
3. **Marcar a coluna de tempo de parede como não comparável entre assinaturas**, com o
   número da correção de retracing como justificativa quantitativa.
4. **Descrever o método de auditoria**, não só o resultado. A afirmação forte deste
   documento não é "as diferenças eram inócuas" — é que qualquer um pode refazer a
   verificação em dois comandos.
