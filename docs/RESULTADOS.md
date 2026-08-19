# Resultados

Gerado por `python -m snakeai.arena --all`. Não editar à mão.

![arena](../assets/arena_light.png)

| algoritmo | rede | params | sementes | passos | score médio (last) | melhor ckpt | passos até 40 | horas | amplitude | mediana | máx | cheio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _piso aleatório_ | — | — | — | 0 | **1,21** | — | — | — | — | 1 | — | 0% |
| acktr · resnet_small | `resnet_small` | 180,464 | 1 | 5,005,312 | **83.91** | — | 1,531,904 | 0.7 | ±0.00 | 96 | 96 | 67.2% |
| ppo · resnet_small | `resnet_small` | 180,464 | 3 | 5,013,504 | **64.56** | 62.72 | 2,703,360 | 0.4 | ±19.15 | 71 | 97 | 0.0% |

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
