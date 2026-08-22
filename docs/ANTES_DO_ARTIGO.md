# O que falta antes de congelar os resultados do artigo

Revisão do estado do repositório com uma pergunta só: **o que hoje entra num artigo e o
que não entra**. É uma lista de decisões, não de tarefas de código.

*Revisado em 21/08, segunda passagem. A versão de 18/08 listava três decisões antes de
gastar GPU; as três estão fechadas. A primeira passagem de 21/08 dizia que o que sobrava era
fila de execução — o A2C fechou os dois braços e produziu um resultado que **derruba duas
afirmações publicadas neste repositório**, e o DQN corrigido abriu uma pergunta nova. O
histórico está no fim, em "O que mudou desde a revisão anterior", porque a decisão importa
mais do que o número que ela substituiu.*

## Resposta curta

**Já não há decisão metodológica travando o artigo.** Dá para escrever agora tudo que não
depende da tabela final: introdução, trabalhos relacionados, o ambiente, o contrato de
comparabilidade, o protocolo de avaliação, a metodologia, e — inteiras — as duas ablações
fechadas: o canal de fome e o orçamento de gradiente.

O que falta é **fila de GPU**: cinco algoritmos ainda sem nenhuma semente na régua atual,
mais a terceira do DQN. Nenhum deles depende de uma escolha que ainda não foi feita.

Há **uma** pergunta nova, e ela é barata: a correção do DQN mudou duas coisas ao mesmo tempo
e o efeito líquido foi de −9,8 pontos. Uma ablação de 1,85 h separa as duas. Ela não trava
nada, mas trava a redação daquele parágrafo.

## O que está fechado

| experimento | sementes | estado |
|---|---:|---|
| PPO, configuração oficial (densa) | 3 | ✅ 81,50 / 78,87 / 82,32 — média 80,90, dp 1,80 |
| ACKTR, região de confiança calibrada | 3 | ✅ 89,78 / 70,67 / 78,13 — média 79,52, dp 9,63 |
| A2C, `t_max = 5` canônico | 3 | ✅ 75,44 / 69,61 / 67,73 — média 70,93, dp 4,02 |
| Ablação do orçamento de gradiente, PPO | 3 + 3 | ✅ `ORCAMENTO_DE_GRADIENTE.md` |
| Ablação do orçamento de gradiente, **réplica no A2C** | 3 + 3 | ✅ previsão pré-registrada **falhou** |
| Ablação do canal de fome | 3 | ✅ `CANAL_DE_FOME.md` |
| Auditoria de procedência do corpus | — | ✅ `PROCEDENCIA.md` |

Três achados já sustentam parágrafo próprio no artigo.

**PPO e ACKTR: mesma média, cinco vezes a dispersão.** Empatados dentro do ruído (80,90
contra 79,52), com desvio entre sementes de 1,80 contra 9,63 — o ACKTR tem a melhor semente
do repositório (89,78) e também a pior das duas famílias (70,67). A média sozinha esconderia
os dois lados.

**O ACKTR chega lá com 1,6% do orçamento de gradiente do PPO.** 610 atualizações contra
38.273, e já saturado: a inclinação no terço final é +0,77 pontos por milhão de passos. É a
observação de maior densidade do repositório, e está desenvolvida em
`ORCAMENTO_DE_GRADIENTE.md`, seção "O eixo entre famílias".

**A previsão registrada para a réplica do orçamento falhou.** O efeito no A2C deveria ser
proporcionalmente menor (razão de 3,2× entre os braços contra 16× no PPO) e saiu idêntico:
+18,70 contra +18,71. Além disso a explicação publicada para o colapso de dispersão **não
replicou** — no A2C o desvio não se mexe (4,11 → 4,02). As duas afirmações derrubadas ficam
no documento, marcadas, em vez de apagadas.

## O que falta, e é só execução

Cinco algoritmos × 3 sementes, mais a terceira semente do DQN. Nenhum depende de decisão
pendente — os quatro sequenciais já têm o truncamento por fome corrigido, e o DQN já conta a
rede alvo em atualizações de gradiente em vez de passos de ambiente.

| ordem | algoritmo | notebook | estado | custo medido |
|---:|---|---|---|---|
| ~~1~~ | ~~A2C~~ | `04_a2c` | ✅ **feito**, nos dois braços | 0,31 h/semente |
| 2 | DQN | `02_dqn` | 🔄 2 de 3 sementes | 1,85 h/semente |
| 3 | Rainbow | `03_rainbow` | pendente | desconhecido |
| 4 | ACER | `05_acer` | pendente | desconhecido |
| 5 | DreamerV3 | `09_dreamerv3` | pendente | desconhecido |
| 6 | AlphaZero | `06_alphazero` | pendente | desconhecido — busca em árvore |
| 7 | MuZero | `07_muzero` | pendente | desconhecido |

O A2C saiu **muito** mais barato que a estimativa de 0,7 h: 0,31 h por semente, o menor
custo do repositório. A correção de retracing (`PROCEDENCIA.md`, caso 2) responde por boa
parte disso.

Antes de 4–7, uma **sondagem de tempo**: uma semente parcial de cada um dos quatro só para
medir passos/s e projetar a fila. Quatro execuções que estouram o limite de sessão do
Kaggle descobertas uma a uma custam mais do que quatro sondagens de dez minutos.

Faltam ainda, e são opcionais:

* 2 sementes para cada uma das duas ablações da região de confiança do ACKTR
  (`+kl0.002` e `+kl_nominal+kl0.002`) — hoje têm 1, e a arena avisa. São as **únicas** duas
  configurações que ainda disparam esse aviso;
* ~~3 sementes de `95_a2c_orcamento_esparso`~~ — **feito**, e o resultado derrubou a
  previsão registrada.

## O ponto novo em aberto: a correção do DQN custou 9,8 pontos

A execução pré-correção do DQN marcava **57,43**. As duas sementes da versão corrigida
marcam **48,24 e 47,11** — média 47,67, desvio 0,80. A correção tornou o número **pior**, e
isso precisa estar no artigo com essas palavras, não escondido atrás de "reexecutado com o
pacote corrigido".

O que a correção arrumou funcionou como previsto: a morte por inanição caiu de **34,3% para
2,2%**, que era exatamente o sintoma que o bug do truncamento por fome produzia (a fome
entrava no buffer como terminação e o `next_obs` gravado era o do episódio seguinte —
§1.1 da revisão).

**Mas duas coisas mudaram ao mesmo tempo**, e por isso os 9,8 pontos não são atribuíveis:

| | pré-correção | corrigido |
|---|---|---|
| truncamento por fome | bugado | corrigido |
| `target_update` | 2.000 | 250 |
| score | 57,43 | 47,67 *(n=2)* |
| morte por fome | 34,3% | 2,2% |
| `meta["atualizacoes"]` | não gravado | 38.908 |

Três leituras possíveis, e os dados atuais não separam: o número antigo era inflado pelo bug;
a recalibração de `target_update` custou desempenho; ou as duas coisas em direções
diferentes. **Uma ablação de uma variável resolve** — refazer uma semente com o truncamento
corrigido e `target_update = 2.000` custa 1,85 h e fecha a pergunta. Enquanto não for feita,
a redação honesta é "a correção mudou duas coisas e o efeito líquido foi −9,8 pontos".

Vale notar o que isso não é: não é motivo para preferir o número antigo. A execução antiga
tem um bug de treino documentado e permanece `comparable=False`. Um resultado melhor obtido
com um bootstrap errado não é um resultado melhor.

### E o par que este resultado libera

O DQN corrigido grava **38.908** atualizações de gradiente; o PPO denso grava **38.273**.
Uma diferença de 1,7%. É o **único par do repositório com o orçamento de gradiente casado**,
e portanto a única comparação entre dois algoritmos que este benchmark oferece hoje sem o
confundidor descrito em `ORCAMENTO_DE_GRADIENTE.md`: 80,90 contra 47,67.

Duas ressalvas para não sobrevender: o DQN aqui é o **baseline de 2015** puro
(`double`, `dueling`, `per`, `noisy` desligados, `n_steps=1`, sem C51), então o par mede
PPO contra DQN-vanilla, não contra o melhor método baseado em valor; e o Rainbow, que é essa
comparação justa, é o próximo da fila. Além disso o DQN ainda tem duas sementes.

## Procedência do corpus

A seção que antes ficava aqui — sobre a assinatura do pacote ter mudado no meio do corpo de
execuções — virou documento próprio: [`PROCEDENCIA.md`](PROCEDENCIA.md). Ele cobre o que a
assinatura é e o que ela não é, o inventário de qual código produziu cada uma das 24
execuções, o método de auditoria (reconstruir a assinatura em qualquer commit, em dois
comandos), e os três casos concretos: o trio do A2C esparso partido entre duas assinaturas,
a medição controlada do custo de retracing, e as duas vezes em que a pasta e a identidade
gravada discordaram.

O resumo operacional, para quem só precisa da conclusão: **nenhuma diferença de código
dentro do corpus toca num número de score**; a coluna de **tempo de parede não é comparável
entre assinaturas**; e o trio `ppo/resnet_small_esparso` é o único que compete na arena sem
assinatura gravada.

## Uma correção de rótulo que vale um parágrafo do artigo

A coluna oficial da arena chamava-se **"score médio (last)"** e entregava a **mediana entre
as sementes**. As duas coisas convivem sem problema — a mediana é a estatística que o
gráfico já desenhava como linha, e com três sementes ela é o que uma semente divergente não
arrasta —, mas o rótulo dizia outra coisa, e o resultado era que a arena e os documentos de
ablação davam números diferentes para as mesmas execuções: PPO 81,50 numa e 80,90 na outra,
ACKTR 78,13 e 79,53. Nenhum dos dois estava errado; faltava dizer qual era qual.

A coluna passou a se chamar **"score (last)"**, o texto abaixo da tabela diz que agrega
sementes por mediana, e a coluna que antes se chamava "mediana" (que é a mediana entre
*episódios*, não entre sementes) virou **"mediana/ep"**. Um teste prende o rótulo à
estatística para não voltarem a divergir.

Para o artigo a recomendação é reportar **os dois**: mediana entre sementes na tabela de
ranking, média e desvio no texto de cada ablação. Com n = 3 nenhum dos dois é suficiente
sozinho, e a diferença entre eles é informação — para o ACKTR, ela é literalmente o achado.

## O que já dá para escrever

O ambiente e o contrato; a discussão de comparabilidade (é a contribuição metodológica mais
forte do trabalho, e independe dos números); o protocolo de avaliação, incluindo a correção
do viés de amostra que moveu o piso de 1,08 para 1,21; as duas ablações inteiras, com
gráfico; o eixo do orçamento de gradiente como confundidor declarado; e a seção de
limitações — para a qual a revisão dos algoritmos é matéria-prima direta.

## O que mudou desde a revisão anterior

Todos os itens que a versão de 18/08 listava como bloqueio estão fechados:

* **ACKTR com 1 semente e régua antiga** → 3 sementes na régua atual. A execução antiga
  virou `acktr/resnet_small_regua_antiga/seed0`, `comparable=False`, com o motivo escrito
  no registro em vez de na memória de alguém.
* **DQN pré-correção entrando na arena como resultado oficial** → marcada
  `comparable=False` e listada na seção "execuções que não entraram", com o sintoma
  (34,3% de morte por fome) preservado como evidência do "antes".
* **Decisão do orçamento de gradiente** → medida com 3 sementes de cada lado e **efetivada**:
  a configuração densa é o padrão do `PPOConfig`, e a antiga virou a ablação
  `96_ppo_orcamento_esparso`. O mesmo tratamento foi dado ao ACKTR (região calibrada é o
  padrão, nominal virou `98_acktr_kl_nominal`) e ao A2C (`t_max=5` canônico é o padrão,
  `rollout=16` virou `95_a2c_orcamento_esparso`).
* **Truncamento por fome nos sequenciais** → corrigido em ACER, DreamerV3, AlphaZero e
  MuZero, com testes que verificam o bootstrap no passo truncado.
* **Congelar o commit** → `ASSINATURA_PACOTE` é injetada pelo gerador e gravada em
  `meta["assinatura_pacote"]`, resolvendo o `commit = "desconhecido"` do Kaggle.
* **Rainbow com a rede alvo congelada** (achado novo, 21/08) → `target_update` passou a
  ser contado em atualizações de gradiente em §2.4, o DQN teve o valor recalculado junto
  (2.000 → 250) e o Rainbow **não**: os 8.000 canônicos viraram 41% do orçamento, ou duas
  sincronizações no treino inteiro. Corrigido para 1.000, com um teste que exige pelo menos
  dez sincronizações de qualquer agente de valor. Pegou a tempo: nenhuma semente de Rainbow
  tinha sido gasta ainda.
* **As quatro emendas pequenas** → todas feitas: `validate()` exige as chaves de causa de
  fim, a arena avisa quando `n < 3` **e revalida cada registro na hora de montar** (em vez
  de confiar no carimbo gravado no dia do treino), a assinatura vai para o registro, e o
  `.gitattributes` com `eol=lf` está no repositório.
* **A ablação do canal de fome sumindo da arena** → execuções `comparable=False` agora
  aparecem na seção "execuções que não entraram", com o motivo.

* **A2C fechado nos dois braços** (21/08) → 3 sementes densas (média 70,93, dp 4,02) e 3
  esparsas (52,22, dp 4,11). A previsão pré-registrada para o tamanho do efeito **falhou**, e
  a explicação publicada para o colapso de dispersão **não replicou**. As duas ficam no
  `ORCAMENTO_DE_GRADIENTE.md`, marcadas.
* **DQN reexecutado com o pacote corrigido** (21/08) → a morte por fome caiu de 34,3% para
  2,2% e o score caiu de 57,43 para 47,67. A execução antiga virou `dqn/base_antigo/seed0`,
  `comparable=False`. Duas mudanças ao mesmo tempo; a atribuição está em aberto.
* **Procedência auditável** (21/08) → `PROCEDENCIA.md`, com o inventário das 24 execuções e
  o método para reconstruir a assinatura em qualquer commit.
* **Identidade travada por teste** (21/08) → `test_every_recorded_run_sits_where_its_identity_says`.
  Escrito para o A2C esparso, pegou sozinho o `dqn/base_antigo` cinco dias depois.

Duas coisas ficaram registradas como hipóteses **não** confirmadas, e é importante que
fiquem assim: o `vf_clip` travando o crítico (§2.2) não se confirmou no orçamento denso —
a variância explicada ficou entre 0,88 e 0,96 —, e a ideia de um ótimo interior de KL no
ACKTR foi **retirada** depois que a semente 2 a contradisse.

## Higiene de repositório pendente (não bloqueia nada)

Coisas que só o Victor pode apagar, porque a ponte com o dispositivo não deleta arquivos:

* `notebooks/98_acktr_kl_max_corrigido.ipynb` — substituído por `98_acktr_kl_nominal.ipynb`
  (`git rm`). **Pendente**: é um arquivo rastreado, e a ponte não remove do índice.
* `_to_delete/` na raiz — pastas vazias já movidas para fora de `runs/` e o `bundle.tgz` de
  transporte. Está no `.gitignore`, então é limpeza de disco: `rmdir /s /q _to_delete`.
* ~~pastas vazias em `runs/`~~ — **feito em 21/08**. As seis (`ppo/resnet_small_denso`,
  `ppo/resnet_small_fome`, `acktr/resnet_small+klcal`, `acktr/resnet_small+klcal15`,
  `acktr/resnet_small+sondagem` e `runs/_mudanca_temporaria`) saíram de `runs/`, que hoje
  não tem nenhum diretório vazio. Eram sobras da renomeação; como o git ignora pasta vazia,
  foi limpeza de disco e não aparece no histórico.
