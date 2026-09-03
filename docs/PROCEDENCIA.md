# Procedência: qual código produziu cada número

Este benchmark foi medido ao longo de três semanas, com o pacote mudando entre execuções.
São **56 execuções gravadas** em 55 identidades `(algo, variant, seed)` — a 56ª é a colisão
do LBC descrita no rodapé desta seção —, e elas carregam **25 assinaturas distintas**, mais
oito execuções anteriores ao mecanismo de assinatura. Um artigo que publica
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
| a2c/resnet_small/seed{0,1,2} | `df6c8eb2b2ca2f58` | 1.954 | 0,31 | 0,57 |
| a2c/resnet_small_esparso/seed0 | `782a8b8aa4af004f` | 611 | 0,78 | 4,58 |
| a2c/resnet_small_esparso/seed{1,2} | `df6c8eb2b2ca2f58` | 611 | 0,25 | 1,46 |
| acektr/resnet_small/seed{0,1,2} | `849d16d28efc2d5d` | — | 0,29 | — |
| acektr/resnet_small+base50+s_ema/seed0 | `d3b3680cad73bfb8` | — | 0,39 | — |
| acektr/resnet_small+kl_cal_v1+s_ema_T5/seed0 | `67fb85327b6cc0c7` | — | 0,40 | — |
| acer/resnet_small/seed{0,1} | `027b5bbfc345ca18` | — | 1,43 | — |
| acer/resnet_small/seed2 | `a185b0e84e0f6066` | — | 0,60 | — |
| acktr/resnet_small/seed{0,1,2} | `ca21410bf88c1c65` | — | 0,51 | — |
| acktr/resnet_small+kl0.002/seed0 | `ca21410bf88c1c65` | — | 0,51 | — |
| acktr/resnet_small+kl_cal_debias_definitiva/seed{0,1} | `a8ed01298eb66b12` | — | 0,26 | — |
| acktr/resnet_small+kl_nominal+kl0.002/seed0 | `ca21410bf88c1c65` | — | 0,51 | — |
| acktr/resnet_small+kl_nominal_momento_descontado/seed0 | `88e94000953da52b` | — | 0,20 | — |
| acktr/resnet_small_regua_antiga/seed0 | — | — | 0,66 | — |
| alphazero/sims32/seed0 | `13560a9422c146ad` | 39.064 | 7,16 | 0,66 |
| alphazero/sims32/seed{1,2} | `f1e812f31407d2a2` | 39.064 | 7,04 | 0,65 |
| alphazero/sims32_sem_correcoes/seed0 | `03504eb9b222dcf6` | — | 7,54 | — |
| dqn/base/seed{0,1,2} | `75aa8ceb896d2cbc` | 38.908 | 1,85 | 0,17 |
| dqn/base_antigo/seed0 | — | — | 2,35 | — |
| lbc/resnet_small/seed0 | `3c00ec03887e3d23` | 3.524 | 0,16 | 0,16 |
| lbc/resnet_small+H_shaping/seed0 | `f0315ec8b316f379` | 9.523 | 0,22 | 0,08 |
| lbc/resnet_small+H_shaping+conc49_bala_de_prata/seed0 **(duas vezes)** | `f6ac5a2ec858e414` | 15.782 · 18.399 | 0,26 · 0,30 | 0,06 |
| lbc/resnet_small_antes_das_correcoes/seed0 | `3d2d21a62b187954` | 39.168 | 0,36 | 0,03 |
| muzero/unroll10+num_simulations32/seed0 | `cec3a247a15263a3` | 39.064 | 9,67 | 0,89 |
| muzero/unroll5/seed0 | `cec3a247a15263a3` | 39.064 | 6,77 | 0,62 |
| muzero/unroll5_normaliza_unroll/seed0 | `d5ad07023295c02e` | 39.064 | 6,65 | 0,61 |
| ppo/resnet_small/seed{0,1,2} | `40448b19b28116da` | 38.274 | 0,83 | 0,08 |
| ppo/resnet_small_esparso/seed{0,1,2} | — | — | 0,41 | — |
| ppo/resnet_small_fome_esparso/seed{0,1,2} | — | — | 0,40 | — |
| rainbow/completo/seed{0,1} | `b6be2f8e874d7644` | 19.454 | 4,26 | 0,79 |
| rainbow/completo/seed2 | `b26e37e2d9d82a27` | 19.454 | 2,83 | 0,52 |
| rainbow/completo+n3/seed0 | `88e54feead9b01a9` | 19.454 | 2,61 | 0,48 |
| rainbow/completo+n3+sem_noisy+eps_greedy/seed0 | `88e54feead9b01a9` | 19.454 | 8,02 | 1,48 |
| soap/resnet_small/seed{0,1,2} | `2464fce3786fd31a` | 36.906 | 0,36 | 0,04 |

O travessão na coluna de assinatura marca execuções anteriores ao mecanismo. Todas elas já
são `comparable=False` por outros motivos, **exceto** `ppo/resnet_small_esparso`, que compete
e cuja procedência precisa ser reconstruída pelo commit em que os dados foram acrescentados.
O travessão em `atualizações` é outra coisa: `meta["atualizacoes"]` passou a ser gravado
depois, e onde ele falta o `s/atualização` não pode ser calculado sem inventar o
denominador.

**Três configurações têm sementes de assinaturas diferentes**, e a tabela agora as separa em
linhas em vez de escondê-las numa média. Além do `a2c/resnet_small_esparso`, que já estava
documentado, aparecem `acer/resnet_small` (a semente 2 rodou com `a185b0e84e0f6066` e em
menos da metade do tempo de parede das outras duas), `alphazero/sims32` (a semente 0 com
`13560a9422c146ad`) e `rainbow/completo` (a semente 2 com `b26e37e2d9d82a27`). Nenhuma delas
tem a análise que o Caso 4 fez para o `n_steps` do Rainbow, e as três competem hoje na arena
como se as sementes fossem intercambiáveis — é dívida de procedência aberta, não um problema
resolvido.

**A linha de duas execuções.** `lbc/resnet_small+H_shaping+conc49_bala_de_prata/seed0` foi
rodado duas vezes com a **mesma** configuração e a **mesma** assinatura, e as duas ficaram
gravadas: 24,82 pontos às 14:06 e 61,35 às 16:30 de 03/09. Elas dividem a identidade
`(algo, variant, seed)`, então `load_all` as junta numa curva só e a arena reporta amplitude
±36,53 onde deveria haver uma execução. Está aqui de propósito, enquanto a causa da
diferença é investigada: o que a tabela não pode fazer é fingir que a segunda substituiu a
primeira. `tests/test_record.py::test_every_recorded_run_sits_where_its_identity_says` fica
vermelho enquanto as duas conviverem — é o teste fazendo o trabalho dele.

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

## Caso 4 — Os dois braços do `n_steps` do Rainbow não têm a mesma assinatura

O resultado mais violento do repositório — `n_steps=3` em **0,57** contra `n_steps=20` em
**65,43** (§2.24 da revisão) — é uma comparação de duas execuções cujas assinaturas
**diferem**: `88e54feead9b01a9` e `b6be2f8e874d7644`, separadas por 2 h 40 do mesmo dia.

Confrontando os dois `config`, a diferença é de **duas** chaves e não de uma:

| chave | `completo+n3` | `completo` |
|---|---|---|
| `n_steps` | 3 | 20 |
| `ruido_por_ambiente` | ausente | `False` |

A segunda é o campo que o `por_amostra` da `NoisyDense` trouxe, e ele entrou **desligado**.
Ou seja: o comportamento provavelmente é idêntico, e a assinatura mudou porque o *arquivo*
mudou, não porque o que roda mudou.

"Provavelmente" é o ponto desta seção. O procedimento deste documento existe para não
transformar "provavelmente idêntico" em "idêntico" no meio de um parágrafo, e a régua é a
mesma do Caso 1: **a diferença fica declarada, e a conclusão é dimensionada pelo que a
evidência sustenta.** Aqui ela sustenta bastante — a diferença entre os dois braços não é de
grau, é qualitativa (100% de fome contra 87,8% de colisão), e nenhum mecanismo plausível
liga um campo desligado de ruído a esse desfecho. O que ela **não** sustenta é um tamanho de
efeito: "n=20 vale 64,9 pontos" é uma frase que precisa de três sementes de cada lado na
mesma assinatura, e o `94_rainbow_nstep3` existe para produzi-las.

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

Aconteceu quatro vezes neste repositório:

* **`a2c/resnet_small_esparso/seed0`** — rodada antes de `A2CConfig.esparso()` existir, saiu
  com `sufixo_variante=""`, virou `a2c/resnet_small/seed0` e foi renomeada à mão. A colisão
  só apareceria quando o braço denso — que é `a2c/resnet_small/seed0` por direito — fosse
  medido, e apareceria como uma mediana de duas sementes onde só existe uma, com orçamentos
  de gradiente diferentes.
* **`dqn/base_antigo/seed0`** — a execução pré-correção, deslocada quando o DQN corrigido
  tomou o lugar em `dqn/base/`. Continuava se identificando como `dqn/base/seed0`, ou seja,
  uma execução `comparable=False` compartilhando identidade com o resultado oficial.
* **`muzero/unroll10+num_simulations32/seed0`** — o `MuZeroAgent` deriva a variante de
  `unroll` e de mais nada, então a execução saiu carimbada `unroll10` mesmo tendo subido
  `num_simulations` de 24 para 32. A pasta foi nomeada à mão com a marca certa e o JSON ficou
  para trás. Aqui a colisão ainda não existia — não há um `unroll10` puro — mas ela seria
  **inevitável** no dia em que houvesse, e o nome que descrevia a configuração era o da pasta.
* **`lbc/resnet_small_antes_das_correcoes/seed0`** — a primeira execução do LBC (§2.10 do
  `LBC.md`), copiada para uma pasta `runs/_falhas/` que não existe em lugar nenhum do código.
  Este é o caso mais caro dos quatro, e por dois motivos: `load_all` faz `os.walk` em `runs/`
  inteiro e **não sabe o que é uma pasta de quarentena**, então a cópia era carregada como
  qualquer outra execução; e como as duas cópias eram byte a byte idênticas, a arena
  registrava `lbc/resnet_small/seed0` **duas vezes** — a mesma execução, contada duas vezes,
  como se fossem duas sementes.

Nos quatro casos o `history.json` foi corrigido no lugar: `variant`, `config.sufixo_variante`
e os caminhos gravados em `meta["artefatos"]`. Curva, `final` e proveniência não foram
tocados.

O que impede a quinta vez é um teste, `test_every_recorded_run_sits_where_its_identity_says`
em `tests/test_record.py`, que varre `runs/` e exige
`runs/<algo>/<variant>/seed<N>/history.json` igual à tripla de dentro do arquivo. Ele pegou o
caso 2 sozinho, cinco dias depois de ter sido escrito para o caso 1, e pegou os casos 3 e 4
juntos.

**O corolário do caso 4, que é o único realmente novo:** não existe pasta de quarentena
dentro de `runs/`. Uma execução que não deve competir se declara `comparable=False` com o
motivo escrito e **continua morando no endereço da própria identidade** — é assim que
`dqn/base_antigo` e `acktr/resnet_small_regua_antiga` são mantidos, e é o que a arena sabe
ler: ela lista as execuções fora da arena com o motivo de cada uma, em vez de fingir que não
existem. Um prefixo `_` numa pasta é uma convenção que só existe na cabeça de quem a criou;
`os.walk` não a respeita, e o resultado é uma execução fantasma numa mediana.

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
