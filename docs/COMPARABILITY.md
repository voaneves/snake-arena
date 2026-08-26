# O contrato de comparabilidade

Este é o documento que dá sentido ao repositório. **Nenhum resultado entra no gráfico da
arena se não obedecer ao que está aqui** — não por burocracia, mas porque um número
incomparável dentro de um gráfico comparativo parece legítimo, e isso é pior do que não
ter o número.

O contrato é código: `snakeai/record.py` o define como constante, `validate()` o aplica a
cada execução, e `tests/test_record.py` trava cada cláusula.

## As regras

| Item | Valor fixado | Onde vive |
|---|---|---|
| Ambiente | `VecSnake` | `CONTRATO["env"]` |
| Tabuleiro | 10 × 10 | `CONTRATO["board_size"]` |
| Fome | `starve_base = 100` passos desde a última comida | `CONTRATO["starve_base"]` |
| Observação | 5 canais egocêntricos `(B, B, 5)` | `CONTRATO["n_channels"]`, `["obs"]` |
| Ações | 3 relativas, com máscara de morte imediata | `CONTRATO["n_actions"]` |
| Recompensa | `+1` comer · `−1` morrer · `0` passo | `CONTRATO["reward_food"]`, `["reward_death"]` |
| **Métrica** | `score` = comida comida, começando em **0** | `CONTRATO["metric"]` |
| **Orçamento** | **5.000.000** passos de ambiente | `ORCAMENTO_OFICIAL` |
| Avaliação | 1.000 episódios, greedy, `seed=123`, **sem** filtro de segurança | `CONTRATO["eval_*"]` |
| Sementes | 3 por configuração (0, 1, 2) | convenção da arena |
| Piso | política aleatória **com máscara** = **1,21 ± 0,06** | medido |
| Teto | **97** — score perfeito num 10 × 10 | `SCORE_PERFEITO` |

## Por que cada regra existe

**Score, nunca comprimento.** O repositório antigo registrava `snake.length`, que começa em
3. Um "16" daquelas curvas é um score 13. Foi a primeira e mais silenciosa fonte de
incomparabilidade: dois números na mesma unidade aparente, medindo coisas diferentes.

**Passos de ambiente, não episódios.** Com centenas de ambientes em paralelo, "episódio"
deixa de ser unidade de tempo — e encolhe conforme o agente melhora: no começo são ~200
passos por episódio, com score ~50 já são ~700. Medir em episódios **premia quem morre
rápido**. O número de episódios vai junto no registro, para quem quiser a leitura antiga.

**O mesmo orçamento para todos.** Comparar um algoritmo que treinou 5 M passos com outro
que treinou 500 mil não mede algoritmo, mede paciência. Validado por execução e conferido
de novo no conjunto, quando a arena é montada.

**A amostra de avaliação não pode ser "os primeiros a terminar".** Episódios curtos
terminam primeiro; se a coleta parar ao atingir 1.000, eles dominam e **o agente é
subestimado** — quanto melhor o agente, maior o viés. Cada ambiente contribui com o mesmo
número de episódios. Foi essa correção que moveu o piso medido de 1,08 para 1,21.

**Greedy, sem exploração.** O número honesto é o da política, não o da política com sorte.
Por isso as noisy nets desligam o ruído em `training=False`: sem isso, o mesmo modelo daria
resultados diferentes a cada avaliação.

**O filtro de segurança fica de fora da curva.** O flood-fill é pós-processamento de
inferência, não política aprendida. Vale como coluna separada da tabela — e a mesma regra
se aplica ao **MCTS na hora de jogar**: busca é computação extra no momento da inferência,
então AlphaZero e MuZero são medidos na curva pela **rede pura**, e com busca numa coluna à
parte.

O DreamerV3 passa por essa regra sem asterisco, e vale dizer por quê: ele também tem modelo
do mundo, mas não o usa para agir. O modelo serve para **treinar** o ator em rollouts
imaginados; na hora de jogar, a rede olha o estado latente e escolhe. Não há computação de
inferência extra a descontar. O que ele tem de diferente é **memória**: a política é
recorrente, e por isso `evaluate` avisa a política de onde cada episódio terminou
(`apos_passo`). Sem esse aviso, o latente atravessaria a morte da cobra e o Dreamer seria
medido **para baixo** — um defeito de medição que pareceria um resultado.

**Três sementes.** Uma curva de RL de execução única não é resultado, é anedota. A arena
mostra a mediana com faixa interquartil.

## Uma regra que é de leitura, não de medição

**Oito é o limite de uma paleta categórica.** Com nove algoritmos, a nona cor seria
indistinguível de alguma das oito sob daltonismo — e uma cor ambígua num gráfico
comparativo é a mesma classe de erro que um eixo x fabricado: parece informação e não é.
`cores_por_algoritmo` **levanta exceção** no nono em vez de gerar matiz nova.

A saída é mudar a forma, não a paleta: `arena_figure` troca sozinha para *small multiples*,
um painel por família (política · valor · modelo do mundo), com as demais curvas em cinza na
mesma escala para que a comparação entre famílias não se perca. O painel do legado continua
com o eixo x próprio, em episódios.

## O último modelo e o melhor checkpoint

RL profundo **não melhora monotonicamente**. A garantia de melhora monotônica é da
*policy iteration* tabular e morre quando a tabela vira uma rede; no DQN é pior, porque
aproximação + bootstrapping + off-policy é a tríade sem prova de convergência nenhuma.
Medido aqui: na primeira execução longa do ACKTR, **8 das 21 avaliações** tinham um
checkpoint anterior melhor que o modelo daquele momento — numa delas, 21,7 pontos melhor.

Por isso cada execução guarda dois resultados:

| campo | o que é | onde aparece |
|---|---|---|
| `final` | o modelo do **último** passo | a curva e o **número oficial** da arena |
| `melhor` | o **melhor checkpoint**, com o passo em que apareceu | coluna à parte na tabela |

O oficial é o `final`, e a razão é a mesma que mantém a busca do AlphaZero fora da curva:
escolher o melhor entre N avaliações **premia quem foi medido mais vezes**. Com avaliação
ruidosa, mais medições significa maior chance de uma sair alta por acaso, e aí a régua
passaria a depender de `eval_every_steps`. O `final` mede o que o algoritmo entrega,
instabilidade inclusa — e instabilidade é um resultado, não um detalhe a esconder.

O `melhor` fica registrado porque é a resposta de outra pergunta legítima: *qual modelo eu
levo para o jogo?* As duas convivem, rotuladas, em vez de uma virar a outra em silêncio.

Os dois `.keras` vão para `runs/<algo>/<variante>/seed<N>/modelos/`, junto com a curva e os
GIFs: a pasta de uma execução tem que ser autossuficiente. Um `history.json` que afirma um
score sem o modelo que o produziu é, num repositório feito para tornar resultados
comparáveis, exatamente o que não serve.

## Políticas com memória, e o que o protocolo exige delas

O contrato diz "1.000 episódios, greedy, `seed=123`". Para um agente sem estado interno isso
é tudo. Três agentes daqui **têm** estado interno — o DreamerV3 carrega o latente do modelo do
mundo, o SOAP carrega a crença sobre a opção corrente, o LBC carrega o comportamento sorteado
por episódio — e para eles a frase esconde uma segunda metade.

A política que `snakeai/eval.py` consome pode expor um método a mais:

```python
politica(obs, mask) -> logits          # obrigatório
politica.apos_passo(acoes, done)       # opcional; quem tem memória precisa
```

`apos_passo` recebe **o que de fato aconteceu**: a ação escolhida — que pode não ser o
argmax, se o filtro de segurança agiu — e onde o episódio terminou, para zerar o estado
interno ali. Sem a segunda metade, duas coisas quebram, e nenhuma levanta exceção:

* o estado interno **congela no valor inicial**, e o número publicado é de uma política mais
  fraca que a treinada. A conclusão "modelo do mundo não ajuda aqui" viria do defeito de
  medição, não do algoritmo;
* o estado **atravessa a morte da cobra** e leva a crença de uma partida para dentro da
  próxima.

Isto é regra de contrato e não detalhe de implementação, porque muda o número oficial. A
prova de que não é hipotético está na §3.6 da revisão: o renderizador de GIF **não** chamava
`apos_passo`, e todos os GIFs de DreamerV3 gerados até a correção mostram uma política que
nunca existiu.

Uma consequência que também é de contrato: **a exportação para TFLite não afirma paridade de
ação** para esses agentes. Um `.tflite` que recebe só a observação não consegue reproduzir
uma política cuja ação depende de estado interno; os arquivos continuam sendo gerados e
medidos, e o que não se afirma é a paridade.

E onde a paridade **é** afirmada, ela precisa ser uma medição de verdade. Até a §2.26 da
revisão não era: a conferência reduzia a saída da rede a "um escore por ação" só do lado
Keras, então quebrava no Rainbow — cuja saída é `(lote, ações, átomos)` — e, pior, entregava
número sem quebrar no LBC e no ACER, onde a saída de política do `.tflite` era escolhida por
uma heurística que casava o crítico. Os valores publicados eram 0,315 e 0,210, que é o acaso
com três ações. A regra que ficou: a mesma redução nos dois lados, a saída do `.tflite`
casada pela forma do tensor do Keras, e desempate pelo valor quando duas saídas têm a mesma
forma. Um relatório de exportação anterior a essa correção não sustenta a linha
`acoes_iguais` — os arquivos continuam válidos, a afirmação sobre eles não.

## A capacidade **não** está igualada

O contrato iguala o orçamento de passos de ambiente. Ele não iguala o tamanho da rede — e
a variação entre os doze é de **22×**. Isso não é defeito: cada algoritmo precisa do que
precisa, e obrigar o DreamerV3 a caber no orçamento de parâmetros do PPO seria mutilar o
que ele é. O defeito seria não dizer. Até aqui o número só existia **depois** da execução,
na coluna `params` do `RESULTADOS.md`, então "o ACER tem quase o dobro do A2C?" custava 5 M
de passos para ser respondido.

A tabela sai dos mesmos construtores que os agentes chamam, na configuração padrão de cada
notebook — `python tools/tabela_parametros.py` a regera, e um teste falha se ela divergir:

| notebook | algoritmo | tronco | `model` | extras | total | × PPO |
|---|---|---|---:|---:|---:|---:|
| `01_ppo` | ppo | `resnet_small` | 180.464 | — | 180.464 | 1.00× |
| `02_dqn` | dqn | `resnet_small` | 333.475 | — | 333.475 | 1.85× |
| `03_rainbow` | rainbow | `resnet_small` | 1.196.648 | — | 1.196.648 | 6.63× |
| `04_a2c` | a2c | `resnet_small` | 180.464 | — | 180.464 | 1.00× |
| `05_acer` | acer | `resnet_small` | 334.878 | — | 334.878 | 1.86× |
| `06_alphazero` | alphazero | `resnet_small` | 180.464 | — | 180.464 | 1.00× |
| `07_muzero` | muzero | `resnet_small` | 154.608 | 118.485 | 273.093 | 1.51× |
| `08_acktr` | acktr | `resnet_small` | 180.464 | — | 180.464 | 1.00× |
| `09_dreamerv3` | dreamerv3 | `dreamer_small` | 198.403 | 3.794.054 | 3.992.457 | 22.12× |
| `10_lbc` | lbc | `resnet_small` | 286.896 | — | 286.896 | 1.59× |
| `11_soap` | soap | `resnet_small` | 300.036 | — | 300.036 | 1.66× |
| `12_acektr` | acektr | `resnet_small` | 180.464 | — | 180.464 | 1.00× |

Como ler:

* **seis dos doze são a mesma rede**, 180.464 parâmetros de `build_actor_critic` sobre o
  `resnet_small`: PPO, A2C, ACKTR, ACEKTR e AlphaZero. As comparações dentro desse grupo —
  que são as do §"as três perguntas" e as três ablações de orçamento — têm capacidade
  igualada por construção, e são as únicas que têm;
* **ACER, 1,86×**, e a diferença inteira está numa linha sem justificativa escrita.
  `build_policy_q` projeta o tronco com `Conv2D(8, 1)` antes de achatar, enquanto
  `build_actor_critic` usa `Conv2D(2, 1)`: 800 features entrando na `Dense(256)` em vez de
  200, ou 205.056 parâmetros contra 51.456. O que o algoritmo **de fato** exige — o crítico
  devolver `Q(s,·)` por ação em vez de um escalar, que é o que o Retrace consome — custa 771
  contra 257. A assimetria é do `8`, não do Retrace;
* **DQN 1,85× e Rainbow 6,63×** — a mesma projeção de 8 filtros, mais o que o Rainbow soma
  em cima: `dueling` duplica a corrente densa, `n_atoms=121` multiplica a saída por 121, e
  cada `NoisyDense` guarda μ **e** σ. A cabeça sozinha tem 1.069.000 parâmetros, oito vezes
  o tronco que a alimenta;
* **MuZero** é o único que fica **abaixo** do PPO no `.keras` (0,86×) e acima no total: o
  `model` é o composto `h`+`f`, e a dinâmica `g` — 118.485 parâmetros que a busca usa a cada
  simulação — mora fora dele;
* **DreamerV3, 22×**, é o modelo do mundo: encoder, GRU, prior, posterior, decoder e as
  cabeças. O ator, que é o que joga, tem 198.403 — a mesma ordem do PPO. Comparar o Dreamer
  pelo tamanho do `.keras` do ator seria comparar a ponta do iceberg;
* **LBC 1,59× e SOAP 1,66×** carregam três políticas e quatro opções sobre um tronco
  **compartilhado**. O desvio está declarado em `LBC.md`: o paper trata cada política como
  um modelo independente, e três ResNets separadas triplicariam o custo por passo.

A regra de leitura que sai disto: **quando duas curvas diferem e a capacidade também, o
efeito não está isolado** — o par mede algoritmo *mais* tamanho de rede, e a conclusão tem
de dizer isso. Os pares limpos são os seis de 180.464 entre si, e cada ablação contra o seu
próprio braço de controle, que por construção compartilha a rede.

## As três perguntas, separadas

O gráfico principal responde **uma** pergunta: *quem vai mais longe com os mesmos dados?*
Ele é o oficial, e é o padrão da literatura. Mas duas outras perguntas legítimas moram nos
mesmos dados, e juntá-las numa única resposta seria fingir que são a mesma coisa.

| pergunta | como se lê | onde está |
|---|---|---|
| quem vai mais longe com os mesmos dados | curva na **vertical**, num x fixo | gráfico e coluna `score médio (last)` |
| quem chega antes a um nível dado | curva na **horizontal**, num y fixo | coluna `passos até 40` |
| quem entrega mais por hora de GPU | outro eixo x | `arena_tempo`, painel separado |

**Passos até o limiar.** Sem interpolação: a resolução é a cadência de avaliação, e
interpolar inventaria precisão que a amostragem não tem. Sementes que não chegaram ficam
**fora** da mediana, com o `(k/n)` denunciando — incluí-las como um número grande
arbitrário seria pior que omitir.

**Horas de GPU.** O eixo oficial iguala os *dados vistos*; ele não iguala o *esforço*. Um
passo de AlphaZero é uma busca em árvore inteira, um de DQN é uma passada de rede — a mesma
posição no eixo x custa ordens de grandeza diferentes. Isso não é defeito do contrato, é o
que "passos de ambiente" significa; mas é meia verdade, e a outra metade agora está
desenhada.

Esse painel só vale **dentro do mesmo hardware**. Uma curva de P100 ao lado de uma de T4
compara aceleradores, não algoritmos, e nada no gráfico denunciaria — por isso o registro
guarda plataforma e GPU, e o painel escreve o aviso na figura quando elas divergem.

## Como uma execução é reprovada

O `Recorder` grava **sempre** — perder a curva no fim de um treino de horas seria o pior
desfecho possível. O que acontece é outro: as violações vão para
`meta["contract_violations"]`, `RunRecord.oficial` passa a ser `False`, e a arena lista a
execução como excluída, **com o motivo**. Excluir em silêncio seria pior que incluir.

```
$ python -m snakeai.arena --all
0 execuções oficiais, 6 curvas históricas
  [fora da arena] acer/resnet_tiny/seed0: avaliação final com 300 episódios, contrato exige 1000
```

## As curvas históricas

Os seis CSVs de 10.000 episódios do `colab-rl` entram por uma porta lateral:
`comparable=False` mais um `caveat` obrigatório explicando por quê. Elas aparecem **num
painel próprio**, com o eixo em episódios, em cinza tracejado — nunca no mesmo eixo x das
execuções novas. Plotá-las juntas fabricaria um eixo comum que não existe, que é o mesmo
pecado do gráfico de dois eixos y com outra roupa.

## Mudar o contrato

É permitido, e deve ser consciente. `tests/test_record.py::test_contract_constant_is_the_documented_one`
quebra quando alguém mexe em `CONTRATO`, obrigando a mudança a ser deliberada — e a mudança
**invalida o histórico**: execuções gravadas sob o contrato antigo deixam de ser comparáveis
com as novas, exatamente como as de 2019.
