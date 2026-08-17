<a name="topo"></a>

<div align="center">

# 🐍 snake-arena

**Um benchmark reprodutível de algoritmos de Reinforcement Learning no Snake.**

Todos em Keras 3. Todos no mesmo ambiente. Todos no mesmo gráfico.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Keras 3](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/status-em%20constru%C3%A7%C3%A3o-yellow?style=for-the-badge)](#roteiro)

</div>

> **Estado atual: em construção.** A estrutura e o contrato de comparabilidade estão definidos; os
> algoritmos estão sendo portados. Este README não contém nenhum número inventado — a tabela de
> resultados só será preenchida com execuções reais, e as células vazias são vazias de propósito.

---

## Sumário

- [Por que este repositório existe](#por-que)
- [Relação com os repositórios antigos](#antigos)
- [O contrato de comparabilidade](#contrato)
- [Algoritmos](#algoritmos)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura)
- [Como usar](#como-usar)
- [Notas de porte para Keras 3](#keras3)
- [Roteiro](#roteiro)
- [Créditos e licença](#creditos)

---

<a name="por-que"></a>
## Por que este repositório existe

Entre 2018 e 2023 eu acumulei treze notebooks de Snake com RL espalhados por duas pastas chamadas
`Working` e `Not Working`. Eles tinham DQN com todas as variações do Rainbow, ACER, experimentos com
o otimizador K-FAC — e nenhum deles era comparável com nenhum outro. Cada um media uma coisa
diferente, num ambiente ligeiramente diferente, e onze dos treze terminavam com uma exceção na
última execução salva.

O gatilho foi reescrever o agente em **PPO com Keras 3**, sobre um ambiente vetorizado em NumPy.
Ele funcionou — e aí veio a pergunta óbvia: *funcionou melhor que o quê?* Não dava para responder.
As curvas antigas do DQN registravam o **comprimento da cobra** (que começa em 3), a curva nova
registrava o **score** (que começa em 0), e o jogo por baixo das duas não era o mesmo.

Este repositório é a resposta: um lugar onde cada algoritmo é implementado em Keras 3, roda no mesmo
ambiente, sob o mesmo orçamento, é avaliado pelo mesmo protocolo — e as curvas podem, finalmente,
ser postas lado a lado sem asterisco.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="antigos"></a>
## Relação com os repositórios antigos

Este repositório **não apaga** os anteriores. Ele os sucede, e carrega junto o que eles produziram
de aproveitável.

| Repositório | O que era | O que acontece com ele |
|---|---|---|
| [**voaneves/snake-on-pygame**](https://github.com/voaneves/snake-on-pygame) | O jogo em pygame, jogável por humanos, com uma API no estilo gym para agentes e um leaderboard humano em `resources/scores.json` | **Continua vivo** como o jogo. Vira dependência opcional daqui: renderização e o duelo humano × agente. O leaderboard passa a usar a mesma métrica do benchmark |
| [**voaneves/colab-rl**](https://github.com/voaneves/colab-rl) | Implementações Keras de DQN e família (TF1/Keras 1.x) + 13 notebooks de Snake + modelos de benchmark | **A parte de RL migra para cá.** Os 13 notebooks são preservados em [`legacy/`](legacy/) com o motivo exato da falha documentado; os CSVs de treino viram séries históricas em [`results/legacy/`](results/legacy/). O repo original permanece como acervo de notebooks diversos |

### O que foi resgatado do acervo

Seis CSVs de **10.000 épocas** de treino de DQN sobreviveram, cobrindo `experience replay` e
`prioritized experience replay`, com e sem `double`, `dueling` e retornos de `n` passos. Eles não são
comparáveis com o benchmark novo — foram medidos em outro ambiente — mas são convertidos
(`score = comprimento − 3`) e aparecem no gráfico da arena como **linhas tracejadas cinza**, com a
ressalva no rótulo. História merece ser plotada, só não merece competir.

### O que foi aprendido ao autopsiar o código antigo

Três defeitos no ambiente original ajudam a explicar por que aquelas curvas saturavam cedo. Estão
listados aqui porque são a justificativa técnica do ambiente novo:

1. **A cabeça sumia do estado.** Em `state()`, a posição da cabeça era escrita e logo em seguida
   sobrescrita pelo laço que desenha o corpo. A rede via um blob indistinto — nunca soube para onde
   a cobra estava olhando.
2. **A recompensa não era estacionária.** Comer devolvia `+comprimento` (4, 5, 6, … 30) contra `−1`
   de morte. A escala do alvo de Q crescia dentro do próprio episódio.
3. **A dica de segurança destruía o tabuleiro.** A heurística de perigo escrevia seus três booleanos
   na **última linha** do canvas, apagando o conteúdo real daquelas células e colocando informação
   não-espacial na entrada de uma convolução.

Some a isso um canal único com codificação ordinal (`comida=1 < corpo=2 < cabeça=3`), que faz a
convolução ler categoria como magnitude, e o platô deixa de ser mistério.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="contrato"></a>
## O contrato de comparabilidade

Esta é a regra central do repositório: **nenhum resultado entra no gráfico se não obedecer a esta
tabela.** Há um validador em `snakeai/record.py` e um teste que o aplica a todo `history.json`.

| Item | Valor fixado |
|---|---|
| Ambiente | `VecSnake` — N tabuleiros em paralelo, NumPy puro, sem pygame |
| Tabuleiro | 10 × 10, `starve_base = 100` passos desde a última comida |
| Observação | 5 canais egocêntricos `(B, B, 5)`: corpo, cabeça, decaimento de cauda, comida, comprimento |
| Ações | 3 relativas (esquerda, reto, direita) com máscara de morte imediata |
| Recompensa | `+1` comer · `−1` morrer · `0` passo · shaping potencial decaindo a zero em 25% do treino |
| **Métrica** | `score` = comida comida, começando em **0**. Nunca comprimento |
| Piso de referência | política aleatória **com máscara** = **1,21** (medido, 1.000 episódios × 5 sementes, desvio 0,06) |
| Teto | **97** — score perfeito num 10 × 10 |
| Orçamento | o **mesmo número de passos de ambiente** para todos. Episódios são reportados, nunca usados como orçamento |
| Avaliação | 1.000 episódios, política greedy, `seed=123`, **sem** filtro de segurança |
| Filtro de segurança | coluna separada da tabela — é pós-processamento, não política aprendida |
| Sementes | 3 por configuração. Curva = mediana, banda = intervalo interquartil |
| Registro | `runs/<algo>/<variante>/seed<N>/history.json`, esquema único |

**Por que o piso subiu de 1,08 para 1,21.** O número antigo veio da forma ingênua de avaliar:
rodar N ambientes e parar assim que 1.000 episódios terminassem. Isso **subestima qualquer agente**,
porque episódios curtos terminam primeiro e entram na amostra, enquanto os longos — que são os bons —
ainda estão correndo quando a contagem fecha. Quanto melhor o agente, maior o viés. O `evaluate`
deste repositório faz cada ambiente contribuir com o mesmo número de episódios, o que elimina a
distorção. O piso corrigido é **1,21 ± 0,06** (5 sementes), e a mesma correção vale para todos os
algoritmos — que é o ponto.

**Por que passos e não episódios.** O repositório antigo media em épocas (`NB_EPOCH = 10000`). Com
centenas de ambientes em paralelo, "episódio" deixa de ser unidade de tempo — e pior, ele encolhe
conforme o agente melhora: no início são ~200 passos por episódio, com score ~50 já são ~700. Medir
em episódios premia quem morre rápido. O eixo oficial é passo de ambiente; o número de episódios vai
junto no registro, para quem quiser a leitura antiga.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="algoritmos"></a>
## Algoritmos

Todos reimplementados em **Keras 3**, sobre o mesmo ambiente e a mesma API de agente.

| Algoritmo | Paper | Notebook | Origem | Estado |
|---|:---:|---|---|---|
| **PPO** — clipping, GAE(λ), value clipping, early stop por KL | [📎](https://arxiv.org/abs/1707.06347) | `01_ppo.ipynb` | novo, é a referência | ✅ implementado, 19 testes |
| **DQN** — família unificada: ER/PER, double, dueling, n-step, noisy, C51 | [📎](https://arxiv.org/abs/1312.5602) | `02_dqn.ipynb` | 6 notebooks do `colab-rl` | ✅ implementado, 8 variantes testadas |
| **Rainbow** — os seis componentes juntos | [📎](https://arxiv.org/abs/1710.02298) | `03_rainbow.ipynb` | novo | ✅ algoritmo próprio, com linha própria na arena |
| **A2C** — actor-critic síncrono, o controle experimental do PPO | [📎](https://arxiv.org/abs/1602.01783) | `04_a2c.ipynb` | prometido no `colab-rl`, nunca escrito | ✅ implementado, herda o rollout do PPO |
| **ACER** — Retrace(λ), IS truncado com correção de viés, região de confiança | [📎](https://arxiv.org/abs/1611.01224) | `05_acer.ipynb` | 2 notebooks quebrados | ✅ reescrito e **convergindo** (16,8 em 151k passos) |
| **AlphaZero** — MCTS sobre o simulador real | [📎](https://arxiv.org/abs/1712.01815) | `06_alphazero.ipynb` | novo | ✅ implementado; a busca sozinha faz **30,3** |
| **MuZero** — a mesma busca, sobre um modelo aprendido | [📎](https://arxiv.org/abs/1911.08265) | `07_muzero.ipynb` | novo | ✅ implementado |
| **ACKTR** — A2C com gradiente natural via K-FAC e região de confiança | [📎](https://arxiv.org/abs/1708.05144) | `08_acktr.ipynb` | 4 notebooks quebrados | ✅ K-FAC reimplementado em Keras 3, 19 testes de curvatura |
| **DreamerV3** — modelo do mundo, ator treinado no sonho | [📎](https://arxiv.org/abs/2301.04104) | `09_dreamerv3.ipynb` | novo | ✅ RSSM categórico, symlog, two-hot, 28 testes |
| ↳ **ACKTR calibrado** — a KL entregue converge para a pedida | — | `98_acktr_kl_max_corrigido.ipynb` | — | ✅ mesmo agente, `kl_calibrado=True`; é ablação, pode piorar |
| ↳ **PPO com o sexto canal** — a observação passa a ver o relógio da fome | — | `97_ppo_canal_de_fome.ipynb` | — | ⚠️ `comparable=False`: muda a entrada da rede, não divide eixo com as curvas de 5 canais |
| ↳ **eixo de otimizadores** (primeira ordem) | — | `99_ablacoes.ipynb` | — | ✅ Adam, AdamW, RMSprop, Lion e SGD como ablação medida |

<details>
<summary><b>Referências completas</b> — e as peças que não têm linha própria na tabela</summary>

Cada 📎 acima aponta para o paper que **define** o algoritmo. Várias implementações daqui
compõem mais de um trabalho, e as linhas `↳` são ablações deste repositório, não algoritmos
publicados — por isso não têm paper. A lista completa:

| Peça | Onde é usada | Paper |
|---|---|---|
| PPO | `01`, e o rollout que o A2C e o ACKTR herdam | Schulman et al., 2017 — *Proximal Policy Optimization Algorithms* [📎](https://arxiv.org/abs/1707.06347) |
| GAE(λ) | vantagem do PPO, A2C, ACKTR | Schulman et al., 2015 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation* [📎](https://arxiv.org/abs/1506.02438) |
| DQN | `02`, base do Rainbow | Mnih et al., 2013 — *Playing Atari with Deep Reinforcement Learning* [📎](https://arxiv.org/abs/1312.5602) · versão Nature 2015 [📎](https://www.nature.com/articles/nature14236) |
| Double Q-learning | flag `double` | van Hasselt et al., 2015 — *Deep Reinforcement Learning with Double Q-learning* [📎](https://arxiv.org/abs/1509.06461) |
| Dueling | flag `dueling` | Wang et al., 2015 — *Dueling Network Architectures for Deep Reinforcement Learning* [📎](https://arxiv.org/abs/1511.06581) |
| Prioritized replay | flag `per` | Schaul et al., 2015 — *Prioritized Experience Replay* [📎](https://arxiv.org/abs/1511.05952) |
| C51 (RL distribucional) | flag `c51`, e o mesmo motivo do two-hot do Dreamer | Bellemare et al., 2017 — *A Distributional Perspective on Reinforcement Learning* [📎](https://arxiv.org/abs/1707.06887) |
| NoisyNet | flag `noisy` | Fortunato et al., 2017 — *Noisy Networks for Exploration* [📎](https://arxiv.org/abs/1706.10295) |
| Rainbow | `03` — a composição canônica das seis | Hessel et al., 2017 — *Rainbow: Combining Improvements in Deep Reinforcement Learning* [📎](https://arxiv.org/abs/1710.02298) |
| A3C / A2C | `04` | Mnih et al., 2016 — *Asynchronous Methods for Deep Reinforcement Learning* [📎](https://arxiv.org/abs/1602.01783) |
| ACER | `05` | Wang et al., 2016 — *Sample Efficient Actor-Critic with Experience Replay* [📎](https://arxiv.org/abs/1611.01224) |
| Retrace(λ) | o estimador off-policy do ACER | Munos et al., 2016 — *Safe and Efficient Off-Policy Reinforcement Learning* [📎](https://arxiv.org/abs/1606.02647) |
| AlphaZero | `06` | Silver et al., 2017 — *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* [📎](https://arxiv.org/abs/1712.01815) |
| MuZero | `07` | Schrittwieser et al., 2019 — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* [📎](https://arxiv.org/abs/1911.08265) |
| ACKTR | `08`, `98` | Wu et al., 2017 — *Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation* [📎](https://arxiv.org/abs/1708.05144) |
| K-FAC | `snakeai/kfac.py` — as camadas densas | Martens & Grosse, 2015 — *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* [📎](https://arxiv.org/abs/1503.05671) |
| KFC | `snakeai/kfac.py` — as convoluções | Grosse & Martens, 2016 — *A Kronecker-factored Approximate Fisher Matrix for Convolution Layers* [📎](https://arxiv.org/abs/1602.01407) |
| DreamerV3 | `09` | Hafner et al., 2023 — *Mastering Diverse Domains through World Models* [📎](https://arxiv.org/abs/2301.04104) |

</details>

**Sobre o Rainbow.** Ele não é um algoritmo novo — é o `DQN` deste repositório com as seis
flags ligadas. Existe como classe própria por duas razões, as duas sobre honestidade do
gráfico: teria a cor do DQN e um rótulo `dqn · double+dueling+per+noisy+3steps+c51` que
ninguém lê; e a composição canônica do paper fica no código em vez de depender de quem
configura acertar seis argumentos.

**Sobre o K-FAC.** Quatro notebooks de 2019 tentaram e nenhum roda: dependiam de
`tensorflow.contrib.kfac`, removido no TF2. Ele foi reimplementado do zero em Keras 3
(`snakeai/kfac.py`) — captura de ativações e gradientes de pré-ativação por camada,
amortecimento de Tikhonov fatorado, Cholesky nos dois fatores, e o KFC de Grosse & Martens
para as convoluções. A implementação canônica do Google (`tensorflow/kfac`) foi **arquivada
em 19/04/2026** e usa `tensorflow.compat.v1`, então serviu de referência, não de dependência.

Ele **não** entra no eixo `cfg.optimizer`, e por um motivo estrutural: um
`keras.optimizers.Optimizer` só recebe pares `(gradiente, variável)`, e o K-FAC precisa das
ativações de entrada de cada camada. Ele vive onde o uso é historicamente correto — dentro
do **ACKTR**, que é o `A2C` deste repositório com uma única troca. A diferença entre as duas
curvas na arena é a resposta medida para "vale a pena aproximar a curvatura?", com todo o
resto congelado: mesmo rollout, mesmo GAE, mesma rede.

O que trava a corretude é `tests/test_kfac.py`: com a Fisher exata, **um** passo de tamanho
1 aterrissa no ótimo de mínimos quadrados — o que nenhum método de primeira ordem faz, com
learning rate nenhum. É a diferença entre "implementei K-FAC" e "implementei algo que
pré-condiciona".

**Sobre o DreamerV3.** É o único dos nove que não busca nada na hora de agir: AlphaZero e
MuZero gastam computação de inferência em MCTS, o Dreamer usa o modelo só para **treinar**,
em rollouts imaginados. Por isso o número dele na curva é o da política pura, sem asterisco
— a mesma regra que mantém a busca do AlphaZero numa coluna à parte.

**Por que busca.** Snake é determinístico, de informação perfeita, tem 3 ações e o
`VecSnake` faz ~286 mil passos/s. Isso torna planejamento com o simulador **real** a jogada
de maior retorno — e torna desnecessária a parte cara do MuZero, que é aprender um modelo
do mundo que aqui já existe exato. Medido, com um valor heurístico bobo (distância de
Manhattan até a comida) e **nenhum treino**:

| | score médio | máx |
|---|---|---|
| aleatório com máscara | 0,67 | 4 |
| MCTS 8 simulações | **24,15** | 37 |
| MCTS 24 simulações | **30,33** | 46 |

O MuZero está aqui para responder a pergunta oposta: **quanto custa não ter o simulador?**
Ele deveria perder para o AlphaZero neste domínio, e é justamente esse o resultado
interessante.

### Redes como eixo de comparação

A arquitetura não é detalhe de implementação aqui — é uma variável medida. Qualquer agente aceita
qualquer tronco por string:

| Tronco | Origem | Notas |
|---|---|---|
| `cnn2`, `cnn3`, `cnn4` | portadas de `colab-rl/models/utilities/networks.py` | corrigidas (`CNN1` e `CNN2` faziam `return model` com o nome indefinido — nunca funcionaram) e convertidas para `channels_last` |
| `resnet_tiny` (~40k), `resnet_small` (~135k), `resnet_base` (~320k) | do notebook de PPO | ResNet totalmente convolucional com GroupNorm |

Cabeças `dueling`, `noisy` e `c51` encaixam em qualquer tronco. O notebook
`99_ablacoes.ipynb` fixa o algoritmo e varre as redes — assim "qual arquitetura é melhor" vira
medida, não folclore.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="resultados"></a>
## Resultados

![arena](assets/arena_light.png)

A tabela completa está em [`docs/RESULTADOS.md`](docs/RESULTADOS.md), gerada por
`python -m snakeai.arena --all`. **O painel da esquerda está vazio de propósito** — nenhuma
execução no orçamento oficial foi feita ainda, e o repositório prefere um gráfico honesto e
vazio a um gráfico bonito com números de execuções curtas.

O painel da direita é o acervo de 2019, no eixo dele: episódios, não passos de ambiente.

O que já foi medido, fora do contrato e portanto fora da arena:

| | score | onde |
|---|---|---|
| MCTS 24 sims + valor heurístico, sem treino | **30,3** | `tests/test_search.py` |
| melhor DQN de 2019 (treino, ambiente antigo) | 18,3 | `results/legacy/` |
| ACER, 151 mil passos, rede `tiny`, CPU | 16,8 | execução de fumaça |
| piso aleatório com máscara | 1,21 | contrato |

**Os últimos modelos treinados moram neste repositório**, em [`models/`](models/) — `.keras`
para retomar treino e TFLite fp16/int8 para embarcar no jogo.

<a name="estrutura"></a>
## Estrutura do projeto

```
snake-arena/
├── snakeai/                  # o pacote — fonte única de verdade
│   ├── env/                  # VecSnake, wrapper de env único, renderização
│   ├── nets/                 # troncos (cnn2/3/4, resnet) + cabeças + registry
│   ├── agents/               # ppo, dqn, rainbow, a2c, acer sobre uma base comum
│   ├── memory/               # replay uniforme e PER com sum-tree
│   ├── eval.py               # evaluate, verdict, piso aleatório, filtro de segurança
│   ├── record.py             # esquema do history.json + validador do contrato
│   ├── plot.py               # o gráfico comparativo
│   └── export.py             # .keras + TFLite + medição de latência
├── notebooks/                # um .ipynb por algoritmo (9) + as ablações
├── runs/                     # history.json de cada execução (versionado)
├── models/                   # os melhores checkpoints, por algoritmo
├── legacy/                   # os 13 notebooks antigos, congelados e anotados
├── results/legacy/           # os CSVs históricos, normalizados
└── tests/                    # invariantes do ambiente, GAE, sum-tree, formatos
```

Cada notebook roda sozinho no Colab, mas a primeira célula clona este repositório e importa
`snakeai` em vez de redefinir o ambiente. Você mantém N notebooks independentes; uma correção de bug
no ambiente vale para todos, em vez de exigir a mesma edição em seis cópias.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="como-usar"></a>
## Como usar

> As instruções abaixo descrevem a interface alvo. Elas passam a valer conforme cada fase do
> [roteiro](#roteiro) é concluída.

```bash
git clone https://github.com/voaneves/snake-arena
cd snake-arena
pip install -r requirements.txt
```

Treinar um algoritmo com o orçamento oficial:

```bash
python -m snakeai.train --algo ppo --net resnet_small --steps 5_000_000 --seed 0
```

Avaliar um modelo pelo protocolo do contrato:

```bash
python -m snakeai.eval --model models/ppo/best.keras --episodes 1000
```

Regenerar o gráfico e a tabela a partir de tudo que está em `runs/`:

```bash
python -m snakeai.arena --all
```

### Os notebooks rodam no Colab, no Kaggle e na sua máquina

São o ambiente de execução de primeira classe do projeto — a linha de comando acima existe para
CI e para quem tem GPU local, mas **todo notebook abre e roda do zero, sem nada instalado**, e é
**o mesmo arquivo** nas três plataformas: `snakeai/plataforma.py` detecta onde está e escolhe a
pasta que persiste em cada uma.

Manter um `.ipynb` por serviço traria de volta exatamente o problema que este repositório existe
para consertar — duas cópias do mesmo notebook que divergem em silêncio.

No **Kaggle** vale usar *Save Version → Save & Run All*: roda headless, sem aba aberta, e a saída
vira artefato versionado. Para continuar de onde parou, anexe a saída anterior em
*Add Input → Your Work → Notebook Output* — a célula de parâmetros recupera os checkpoints
sozinha.

| Notebook | Abrir |
|---|---|
| PPO — a referência | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/01_ppo.ipynb) |
| DQN — a família inteira | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/02_dqn.ipynb) |
| Rainbow | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/03_rainbow.ipynb) |
| A2C — o controle do PPO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/04_a2c.ipynb) |
| ACER | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/05_acer.ipynb) |
| AlphaZero | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/06_alphazero.ipynb) |
| MuZero | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/07_muzero.ipynb) |
| ACKTR — K-FAC | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/08_acktr.ipynb) |
| DreamerV3 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/09_dreamerv3.ipynb) |
| ACKTR — região de confiança calibrada | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/98_acktr_kl_max_corrigido.ipynb) |
| PPO — sexto canal (fome) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/97_ppo_canal_de_fome.ipynb) |
| Ablações — rede e otimizador | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/99_ablacoes.ipynb) |

O que todo notebook garante, por construção:

- **Roda em `Runtime → GPU (T4)`**, o nível gratuito. Nada aqui exige A100.
- **Primeira célula = clone + `pip install`**, com checagem de versão de TF/Keras e de GPU visível.
- **Sem `pygame`.** O ambiente é NumPy puro; a visualização sai como GIF, porque o Colab não tem
  display.
- **Treino retomável.** A sessão do Colab cai — é uma questão de quando, não de se. Basta rodar a
  célula de novo e ela continua do último checkpoint. Com `USE_DRIVE = True` os checkpoints vão para
  o seu Google Drive e sobrevivem à queda da máquina.
- **Parâmetros via `# @param`**, então dá para ajustar orçamento, rede e semente pelos widgets, sem
  editar código.
- **Uma célula "▶ rodar tudo"** no fim: treina, avalia os 1.000 episódios, exporta os modelos e grava
  o GIF.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="keras3"></a>
## Notas de porte para Keras 3

O alvo é **Keras 3 com backend TensorFlow**, fixado explicitamente — nada de depender do padrão do
ambiente. Toda entrada do projeto (pacote e notebooks) começa com:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
```

O código é escrito em `keras.ops` sempre que não custa nada, então trocar de backend um dia é
possível — mas o backend suportado, testado no CI e usado em todos os números publicados é o
TensorFlow (≥ 2.20). A conversão para TFLite e a medição de latência dependem dele.

Todo código legado passa por este porte. Não há exceção: nada entra no pacote ainda dependendo de
TF1 ou de APIs removidas do Keras.

| Padrão antigo | Substituição |
|---|---|
| `tf.Session`, `tf.global_variables_initializer`, `sess.run` | execução eager; `@tf.function` só onde medimos ganho |
| `K.set_image_dim_ordering('th')` — removido do Keras | `channels_last` em todo o projeto; observações já nascem `(B, B, C)` |
| `keras.backend` para grafo, `tf.losses.huber_loss` | `keras.ops` e `keras.losses.Huber` |
| Otimizadores customizados herdando da API antiga (`COCOB`, `SMORMS3`, `Yogi`, `Nadamax`, `Radamax`, `AdamDelta`) | reescritos sobre `keras.optimizers.Optimizer` — ou aposentados, se não justificarem a manutenção |
| `NoisyDense` mexendo em internals de `Dense` | camada própria com `add_weight` e `keras.random` |
| `tensorflow.contrib.kfac` | não existe, e `tensorflow/kfac` foi arquivado em 19/04/2026. Reimplementado em `snakeai/kfac.py` sobre Keras 3 puro |
| Salvar em `.h5` | `.keras` para retomar treino, TFLite para embarcar |
| Modelo funcional recebendo `KerasTensor` em API TF sem dispatch (o erro que matava o ACER) | operação encapsulada em `keras.Layer` própria |

**Uma armadilha silenciosa, registrada aqui para ninguém repetir:** no Keras 3, converter para TFLite
precisa passar por um SavedModel. `TFLiteConverter.from_concrete_functions(...)` compila sem erro,
gera um arquivo minúsculo — e **não captura os pesos**. A inferência devolve NaN sem nenhum aviso.
Use `model.export(dir, format="tf_saved_model")` e converta a partir dele.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="roteiro"></a>
## Roteiro

- [x] **0** — Criar o repositório, o contrato e o README
- [x] **1** — Núcleo `snakeai/`: ambiente, redes, avaliação, registro, gráfico, testes
- [x] **2** — PPO refatorado para o pacote
- [x] **3** — DQN unificado (substitui os 6 notebooks antigos por um)
- [x] **4** — A2C
- [x] **5** — Rainbow (C51 + double + dueling + PER + n-step + noisy)
- [x] **6** — ACER reescrito — e converge
- [x] **7** — AlphaZero e MuZero, com o MCTS compartilhado
- [x] **8** — Arena: `python -m snakeai.arena --all`
- [x] **9** — Notebooks do Colab, gerados a partir do pacote
- [ ] **10** — **Treinar de verdade**: 3 sementes × orçamento oficial, numa GPU
- [ ] **11** — Modelos exportados, `MODELS.md`, integração com o leaderboard humano
- [ ] **12** — Verificação final: reprodutibilidade, paridade `.keras` × TFLite, CI

O passo 10 é o que falta para o gráfico deixar de estar vazio. Ele não cabe numa CPU: o
orçamento oficial de 5 M passos leva ~3,7 h por semente só no PPO. É para isso que os
notebooks existem.

O plano detalhado, com o diagnóstico completo dos treze notebooks e a lista de bugs encontrados,
está em [`docs/PLANO.md`](docs/PLANO.md).

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="creditos"></a>
## Créditos e licença

- [**@farizrahman4u**](https://github.com/farizrahman4u) — o código de Snake do `qlearning4k` foi a
  base original do jogo.
- [**@chuyangliu**](https://github.com/chuyangliu) — a ideia das ações relativas.
- [**@Kaixhin**](https://github.com/Kaixhin) — a `CNN3` veio da implementação dele do paper do Rainbow.
- *The 37 Implementation Details of PPO* — a lista de detalhes que decide se um PPO aprende ou vira ruído.

Licenciado sob a [Licença MIT](LICENSE).

<div align="center">

Feito por [**@voaneves**](https://github.com/voaneves) · [LinkedIn](https://linkedin.com/in/voaneves)

</div>
