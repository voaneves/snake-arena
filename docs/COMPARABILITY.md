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

**Três sementes.** Uma curva de RL de execução única não é resultado, é anedota. A arena
mostra a mediana com faixa interquartil.

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
