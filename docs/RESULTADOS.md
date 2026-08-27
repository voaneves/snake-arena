# Resultados

Gerado por `python -m snakeai.arena --all`. Não editar à mão.

![arena](../assets/arena_light.png)

O gráfico mostra o **braço principal** de cada algoritmo — o que o notebook roda na
configuração padrão. A tabela abaixo mostra **tudo**, ablações inclusive: a figura
responde *quem vai mais longe com os mesmos dados*, e uma ablação desenhada ao lado
do próprio controle, na mesma cor, responde outra pergunta.

| algoritmo | rede | params | sementes | passos | score (last) | melhor ckpt | passos até 40 | horas | amplitude | mediana/ep | máx | cheio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _piso aleatório_ | — | — | — | 0 | **1,21** | — | — | — | — | 1 | — | 0% |
| acer · resnet_small | `resnet_small` | 334,878 | 2 | 5,001,216 | **83.96** | 88.25 | 1,251,328 | 1.4 | ±12.24 | 91 | 97 | 47.3% |
| ppo · resnet_small | `resnet_small` | 180,464 | 3 | 5,013,504 | **81.50** | 81.98 | 802,816 | 0.9 | ±3.45 | 97 | 97 | 61.4% |
| acktr · resnet_small | `resnet_small` | 180,464 | 3 | 5,005,312 | **78.13** | 85.84 | 1,277,952 | 0.5 | ±19.11 | 97 | 97 | 60.7% |
| acktr · resnet_small+kl0.002 | `resnet_small` | 180,464 | 1 | 5,005,312 | **72.50** | 74.19 | 2,547,712 | 0.5 | ±0.00 | 75 | 97 | 28.9% |
| a2c · resnet_small | `resnet_small` | 180,464 | 3 | 5,002,240 | **69.61** | 72.94 | 2,501,120 | 0.3 | ±7.72 | 77 | 97 | 2.2% |
| ppo · resnet_small_esparso | `resnet_small` | 180,464 | 3 | 5,013,504 | **64.56** | 62.72 | 2,703,360 | 0.4 | ±19.15 | 71 | 97 | 0.0% |
| acktr · resnet_small+kl_nominal+kl0.002 | `resnet_small` | 180,464 | 1 | 5,005,312 | **64.53** | 84.92 | 1,277,952 | 0.5 | ±0.00 | 69 | 97 | 26.7% |
| rainbow · completo | `resnet_small` | 1,196,648 | 2 | 5,000,192 | **54.46** | 75.78 | 1,875,200 | 4.3 | ±21.93 | 44 | 97 | 19.9% |
| a2c · resnet_small_esparso | `resnet_small` | 180,464 | 3 | 5,005,312 | **53.60** | 53.60 | 4,005,888 | 0.3 | ±7.87 | 59 | 78 | 0.0% |
| rainbow · completo+n3+sem_noisy+eps_greedy | `resnet_small` | 662,148 | 1 | 5,000,192 | **49.17** | 49.97 | 4,000,000 | 8.0 | ±0.00 | 49 | 90 | 0.0% |
| dqn · base | `resnet_small` | 333,475 | 3 | 5,000,192 | **47.11** | 51.79 | 3,500,032 | 1.9 | ±2.86 | 49 | 89 | 0.0% |
| rainbow · completo+n3 | `resnet_small` | 1,196,648 | 1 | 5,000,192 | **0.57** | 0.78 | não chegou | 2.6 | ±0.00 | 0 | 6 | 0.0% |

Score perfeito no 10×10: **97**.

**passos até 40** é a curva lida na horizontal em vez da vertical: em vez de *quanto marcou no fim*, *quantos passos precisou para chegar lá*. Sai dos mesmos dados e responde à outra pergunta — menor é melhor. A resolução é a cadência de avaliação, e não há interpolação: o passo mostrado é um em que a medição de fato aconteceu. `(k/n)` significa que só `k` das `n` sementes chegaram, e as que não chegaram ficam **fora** da mediana em vez de entrar como um número inventado.

**horas** é tempo de parede da execução inteira, útil só entre execuções do mesmo hardware. O eixo de passos iguala os *dados vistos*; ele não iguala o *esforço*, e a diferença entre os dois é enorme para quem faz busca em árvore.

A coluna **score (last)** é o número oficial: o modelo do último passo, que é o estado final do algoritmo. O valor é a **mediana entre as sementes** do score médio de cada uma — não a média entre elas. É a mesma estatística que o gráfico desenha como linha, com o intervalo entre sementes como faixa, e com três sementes ela é o que uma semente divergente não consegue arrastar. Os documentos de ablação (`ORCAMENTO_DE_GRADIENTE.md`, `CANAL_DE_FOME.md`) reportam **média e desvio**, porque lá a pergunta é o tamanho de um efeito, não a ordem de um ranking: os dois números convivem, e cada um diz qual é. **mediana/ep** é outra coisa ainda — a mediana entre *episódios*, não entre sementes. **melhor ckpt** é o melhor que aquela execução produziu em algum momento — fica à parte porque premia quem foi medido mais vezes, pela mesma razão que a busca do AlphaZero e o filtro de flood-fill ficam fora da curva.

## O mesmo resultado, no eixo do custo

O gráfico acima iguala os **dados vistos**. Este iguala o **esforço gasto** — e a
ordem muda, porque um passo de AlphaZero custa uma busca em árvore inteira e um
de DQN custa uma passada de rede. São duas perguntas diferentes, e nenhuma das
duas é a resposta da outra.

![arena por tempo](../assets/arena_tempo_light.png)

## E no eixo de quem fecha o tabuleiro

A média e a taxa de vitória são dois funcionais da **mesma** distribuição —
`E[X]` e `P(X = 97)` — e não têm obrigação de concordar. O limiar joga fora tudo
abaixo do teto: um episódio de 96 conta igual a um de 3. A média joga fora o
formato: não distingue "sempre 78" de "metade perfeito, metade zero". **Quando
a ordem desta figura difere da ordem do gráfico oficial, é exatamente isso que
está acontecendo** — e nenhuma das duas é "a qualidade do modelo".

Por isso a barra não é a taxa de vitória sozinha: é a repartição inteira das
causas de fim, com a vitória como primeiro segmento. Ela mostra o que nenhum dos
dois números mostra — perder por fome e perder por colisão são fracassos
diferentes, e a curva de score é idêntica nos dois casos.

![quem fecha o tabuleiro](../assets/arena_vitorias_light.png)

## Configurações com menos de 3 sementes

Entram no gráfico, mas **não sustentam comparação**: a amplitude entre
sementes do PPO neste ambiente é de 19 pontos, maior que quase toda
diferença entre algoritmos que a tabela mostra.

- `acer/resnet_small`: 2 de 3 — faltam 1
- `acktr/resnet_small+kl0.002`: 1 de 3 — faltam 2
- `acktr/resnet_small+kl_nominal+kl0.002`: 1 de 3 — faltam 2
- `rainbow/completo`: 2 de 3 — faltam 1
- `rainbow/completo+n3`: 1 de 3 — faltam 2
- `rainbow/completo+n3+sem_noisy+eps_greedy`: 1 de 3 — faltam 2

## Execuções que não entraram na arena

Estão registradas em `runs/`, com curva e artefatos, mas não competem. O
motivo é conferido na hora de montar a arena, com esta versão do código —
não é o carimbo que ficou gravado no dia do treino. Execuções marcadas
`comparable=False` também aparecem aqui: elas não competem por construção,
e some-las seria pior do que incluí-las.

- `acktr/resnet_small_regua_antiga/seed0`: comparable=False: medida com o protocolo de avaliação anterior a 14/08: sem as chaves de causa de fim, com a maçã do episódio vencedor faltando (score_max 96 num tabuleiro cujo perfeito é 97) e com `win_rate` de outra fórmula. Mantida como registro histórico; a semente 0 na régua atual está em `acktr/resnet_small/seed0`.
- `dqn/base_antigo/seed0`: comparable=False: treinada antes da correção do truncamento por fome (§1.1 da revisão): a fome entrava no buffer como terminação e o `next_obs` gravado era o do episódio seguinte. 34,3% dos episódios finais terminaram por inanição — o sintoma que a correção ataca. Mantida como registro do 'antes'; refazer com o pacote corrigido.
- `ppo/resnet_small_fome_esparso/seed0`: comparable=False: observação com 6 canais (fome), fora do contrato de 5; orçamento de gradiente esparso (~2.400 atualizações), que era o padrão quando a ablação foi medida
- `ppo/resnet_small_fome_esparso/seed1`: comparable=False: observação com 6 canais (fome), fora do contrato de 5; orçamento de gradiente esparso (~2.400 atualizações), que era o padrão quando a ablação foi medida
- `ppo/resnet_small_fome_esparso/seed2`: comparable=False: observação com 6 canais (fome), fora do contrato de 5; orçamento de gradiente esparso (~2.400 atualizações), que era o padrão quando a ablação foi medida
