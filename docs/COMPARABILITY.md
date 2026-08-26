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
