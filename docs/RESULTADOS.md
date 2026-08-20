# Resultados

Gerado por `python -m snakeai.arena --all`. Não editar à mão.

![arena](../assets/arena_light.png)

| algoritmo | rede | params | sementes | passos | score médio (last) | melhor ckpt | passos até 40 | horas | amplitude | mediana | máx | cheio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _piso aleatório_ | — | — | — | 0 | **1,21** | — | — | — | — | 1 | — | 0% |
| ppo · resnet_small_denso | `resnet_small` | 180,464 | 3 | 5,013,504 | **81.50** | 81.98 | 802,816 | 0.9 | ±3.45 | 97 | 97 | 61.4% |
| acktr · resnet_small+klcal | `resnet_small` | 180,464 | 1 | 5,005,312 | **72.50** | 74.19 | 2,547,712 | 0.5 | ±0.00 | 75 | 97 | 28.9% |
| ppo · resnet_small | `resnet_small` | 180,464 | 3 | 5,013,504 | **64.56** | 62.72 | 2,703,360 | 0.4 | ±19.15 | 71 | 97 | 0.0% |
| acktr · resnet_small | `resnet_small` | 180,464 | 1 | 5,005,312 | **64.53** | 84.92 | 1,277,952 | 0.5 | ±0.00 | 69 | 97 | 26.7% |

Score perfeito no 10×10: **97**.

**passos até 40** é a curva lida na horizontal em vez da vertical: em vez de *quanto marcou no fim*, *quantos passos precisou para chegar lá*. Sai dos mesmos dados e responde à outra pergunta — menor é melhor. A resolução é a cadência de avaliação, e não há interpolação: o passo mostrado é um em que a medição de fato aconteceu. `(k/n)` significa que só `k` das `n` sementes chegaram, e as que não chegaram ficam **fora** da mediana em vez de entrar como um número inventado.

**horas** é tempo de parede da execução inteira, útil só entre execuções do mesmo hardware. O eixo de passos iguala os *dados vistos*; ele não iguala o *esforço*, e a diferença entre os dois é enorme para quem faz busca em árvore.

A coluna **score médio (last)** é o número oficial: o modelo do último passo, que é o estado final do algoritmo. **melhor ckpt** é o melhor que aquela execução produziu em algum momento — fica à parte porque premia quem foi medido mais vezes, pela mesma razão que a busca do AlphaZero e o filtro de flood-fill ficam fora da curva.

## O mesmo resultado, no eixo do custo

O gráfico acima iguala os **dados vistos**. Este iguala o **esforço gasto** — e a
ordem muda, porque um passo de AlphaZero custa uma busca em árvore inteira e um
de DQN custa uma passada de rede. São duas perguntas diferentes, e nenhuma das
duas é a resposta da outra.

![arena por tempo](../assets/arena_tempo_light.png)

## Configurações com menos de 3 sementes

Entram no gráfico, mas **não sustentam comparação**: a amplitude entre
sementes do PPO neste ambiente é de 19 pontos, maior que quase toda
diferença entre algoritmos que a tabela mostra.

- `acktr/resnet_small`: 1 de 3 — faltam 2
- `acktr/resnet_small+klcal`: 1 de 3 — faltam 2

## Execuções que não entraram na arena

Estão registradas em `runs/`, com curva e artefatos, mas não competem. O
motivo é conferido na hora de montar a arena, com esta versão do código —
não é o carimbo que ficou gravado no dia do treino. Execuções marcadas
`comparable=False` também aparecem aqui: elas não competem por construção,
e some-las seria pior do que incluí-las.

- `acktr/resnet_small_regua_antiga/seed0`: comparable=False: medida com o protocolo de avaliação anterior a 14/08: sem as chaves de causa de fim, com a maçã do episódio vencedor faltando (score_max 96 num tabuleiro cujo perfeito é 97) e com `win_rate` de outra fórmula. Mantida como registro histórico; a semente 0 na régua atual está em `acktr/resnet_small/seed0`.
- `dqn/base/seed0`: comparable=False: treinada antes da correção do truncamento por fome (§1.1 da revisão): a fome entrava no buffer como terminação e o `next_obs` gravado era o do episódio seguinte. 34,3% dos episódios finais terminaram por inanição — o sintoma que a correção ataca. Mantida como registro do 'antes'; refazer com o pacote corrigido.
- `ppo/resnet_small_fome/seed0`: comparable=False: observação com 6 canais (fome), fora do contrato de 5
- `ppo/resnet_small_fome/seed1`: comparable=False: observação com 6 canais (fome), fora do contrato de 5
- `ppo/resnet_small_fome/seed2`: comparable=False: observação com 6 canais (fome), fora do contrato de 5
