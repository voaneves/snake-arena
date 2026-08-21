# O que falta antes de congelar os resultados do artigo

Revisão do estado do repositório com uma pergunta só: **o que hoje entra num artigo e o
que não entra**. É uma lista de decisões, não de tarefas de código.

*Revisado em 21/08. A versão de 18/08 listava três decisões antes de gastar GPU; as três
estão fechadas, e o que sobrou é fila de execução. O histórico delas está no fim, em
"O que mudou desde a revisão anterior", porque a decisão importa mais do que o número que
ela substituiu.*

## Resposta curta

**Já não há decisão metodológica travando o artigo.** Dá para escrever agora tudo que não
depende da tabela final: introdução, trabalhos relacionados, o ambiente, o contrato de
comparabilidade, o protocolo de avaliação, a metodologia, e — inteiras — as duas ablações
fechadas: o canal de fome e o orçamento de gradiente.

O que falta é **fila de GPU**: sete algoritmos ainda sem as três sementes na régua atual.
Nenhum deles depende de uma escolha que ainda não foi feita.

## O que está fechado

| experimento | sementes | estado |
|---|---:|---|
| PPO, configuração oficial (densa) | 3 | ✅ 81,50 / 78,87 / 82,32 — média 80,90, dp 1,80 |
| ACKTR, região de confiança calibrada | 3 | ✅ 89,78 / 70,67 / 78,13 — média 79,53, dp 9,63 |
| Ablação do orçamento de gradiente (PPO denso × esparso) | 3 + 3 | ✅ `ORCAMENTO_DE_GRADIENTE.md` |
| Ablação do canal de fome | 3 | ✅ `CANAL_DE_FOME.md` |

As duas primeiras linhas são o par mais interessante do repositório: **mesma média, cinco
vezes a dispersão**. PPO e ACKTR terminam empatados dentro do ruído (80,90 contra 79,53),
mas o desvio entre sementes é 1,80 contra 9,63 — o ACKTR tem a melhor semente do
repositório (89,78) e também a pior das duas famílias (70,67). Isso é um achado, não um
empate: a média sozinha esconderia os dois lados.

## O que falta, e é só execução

Sete algoritmos × 3 sementes. Nenhum depende de decisão pendente — os quatro sequenciais já
têm o truncamento por fome corrigido, e o DQN já conta a rede alvo em atualizações de
gradiente em vez de passos de ambiente.

| ordem | algoritmo | notebook | por que nesta posição |
|---:|---|---|---|
| 1 | A2C | `04_a2c` | é o controle experimental do PPO; barato (~0,7 h) e fecha a família de gradiente de política |
| 2 | DQN | `02_dqn` | a execução atual é pré-correção e está fora da arena; ~2,4 h por semente |
| 3 | Rainbow | `03_rainbow` | mesma família, e responde "os seis componentes valem?"; a rede alvo dele acabou de ser recalibrada — ver abaixo |
| 4 | ACER | `05_acer` | daqui em diante o custo por semente é desconhecido |
| 5 | DreamerV3 | `09_dreamerv3` | idem |
| 6 | AlphaZero | `06_alphazero` | idem — busca em árvore, o passo custa muito mais |
| 7 | MuZero | `07_muzero` | idem |

Antes de 4–7, uma **sondagem de tempo**: uma semente parcial de cada um dos quatro só para
medir passos/s e projetar a fila. Quatro execuções que estouram o limite de sessão do
Kaggle descobertas uma a uma custam mais do que quatro sondagens de dez minutos.

Faltam ainda, e são opcionais:

* 2 sementes para cada uma das duas ablações da região de confiança do ACKTR
  (`+kl0.002` e `+kl_nominal+kl0.002`) — hoje têm 1, e a arena avisa;
* 3 sementes de `95_a2c_orcamento_esparso`, se a réplica do eixo de orçamento numa
  segunda família entrar no artigo.

## O único ponto novo em aberto

**A assinatura do pacote mudou no meio do corpo de execuções.** `snakeai/plot.py` foi
editado (o rótulo da coluna oficial da tabela — ver abaixo), e isso muda a assinatura de
todos os notebooks. As execuções que já existem carregam a assinatura antiga; as ~21 que
faltam carregarão a nova.

Antes de tratar isso como problema, vale lembrar o que a assinatura é: ela identifica **a
fatia do pacote embutida naquele notebook**, então `01_ppo` e `08_acktr` nunca tiveram a
mesma — o PPO hoje grava `40448b19b28116da` e o ACKTR `ca21410bf88c1c65`. Ela nunca foi um
identificador global do repositório; é o substituto do `commit`, que o Kaggle não fornece.
O que muda agora é que o **mesmo** notebook passa a ter duas assinaturas ao longo do tempo,
e para o DQN, o A2C e os quatro sequenciais isso é irrelevante porque nenhuma semente deles
foi feita ainda.

E a diferença é inócua, o que vale dizer no artigo em vez de esperarem a pergunta:
`plot.py` é módulo de **relatório**. Não é importado pelo laço de treino nem pelo protocolo
de avaliação, e a tabela é regenerada a partir dos `history.json` na hora de publicar — a
versão de `plot.py` que estava na sessão de treino não toca em nenhum número. As mudanças
que tocariam (PPO, A2C, ACKTR) foram verificadas bit a bit contra a versão anterior antes
de entrar, e estão registradas em `ORCAMENTO_DE_GRADIENTE.md` e `REVISAO_ALGORITMOS.md`.

Um detalhe menor do mesmo tipo: as execuções de ACKTR não gravam `meta["atualizacoes"]`
(o contador entrou depois), e as do PPO esparso não gravam nem assinatura nem contador.
Para essas, o número de atualizações sai da configuração — ~610 no ACKTR, ~2.424 no PPO
esparso — e é analítico, não medido. Vale escrever assim.

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

Duas coisas ficaram registradas como hipóteses **não** confirmadas, e é importante que
fiquem assim: o `vf_clip` travando o crítico (§2.2) não se confirmou no orçamento denso —
a variância explicada ficou entre 0,88 e 0,96 —, e a ideia de um ótimo interior de KL no
ACKTR foi **retirada** depois que a semente 2 a contradisse.

## Higiene de repositório pendente (não bloqueia nada)

Coisas que só o Victor pode apagar, porque a ponte com o dispositivo não deleta arquivos:

* `notebooks/98_acktr_kl_max_corrigido.ipynb` — substituído por `98_acktr_kl_nominal.ipynb`
  (`git rm`);
* pastas vazias em `runs/`: `ppo/resnet_small_denso`, `ppo/resnet_small_fome`,
  `acktr/resnet_small+klcal`, `acktr/resnet_small+klcal15`, `acktr/resnet_small+sondagem`,
  e `runs/_mudanca_temporaria` — sobras da renomeação; o git ignora pasta vazia, então isso
  é limpeza de disco, não do histórico.
