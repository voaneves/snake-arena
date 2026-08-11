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

| Algoritmo | Notebook | Origem | Estado |
|---|---|---|---|
| **PPO** — clipping, GAE(λ), value clipping, early stop por KL, entropia decrescente | `01_ppo.ipynb` | novo, é a referência | 🚧 portando para o pacote |
| **DQN** — família unificada: ER/PER, double, dueling, n-step, noisy | `02_dqn.ipynb` | 6 notebooks do `colab-rl` | 📋 planejado |
| **Rainbow** — C51 sobre a família acima | `03_rainbow.ipynb` | novo | 📋 planejado |
| **A2C** — actor-critic síncrono, o controle experimental do PPO | `04_a2c.ipynb` | prometido no `colab-rl`, nunca escrito | 📋 planejado |
| **ACER** — Retrace(λ), IS truncado com correção de viés, replay ratio | `05_acer.ipynb` | 2 notebooks quebrados | 📋 planejado, risco alto |
| **K-FAC** | — | 2 notebooks quebrados | ⚰️ aposentado — dependia de `tensorflow.contrib`, que não existe desde o TF2. A pergunta original ("o otimizador importa?") vira uma ablação de RMSprop × Adam × AdamW, que roda |

### Redes como eixo de comparação

A arquitetura não é detalhe de implementação aqui — é uma variável medida. Qualquer agente aceita
qualquer tronco por string:

| Tronco | Origem | Notas |
|---|---|---|
| `cnn2`, `cnn3`, `cnn4` | portadas de `colab-rl/models/utilities/networks.py` | corrigidas (`CNN1` e `CNN2` faziam `return model` com o nome indefinido — nunca funcionaram) e convertidas para `channels_last` |
| `resnet_tiny` (~40k), `resnet_small` (~135k), `resnet_base` (~320k) | do notebook de PPO | ResNet totalmente convolucional com GroupNorm |

Cabeças `dueling`, `noisy` e `c51` encaixam em qualquer tronco. O notebook
`99_ablation_redes.ipynb` fixa o algoritmo e varre as redes — assim "qual arquitetura é melhor" vira
medida, não folclore.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

<a name="resultados"></a>
## Resultados

<div align="center">

*O gráfico da arena aparece aqui quando as primeiras execuções oficiais terminarem.*

</div>

| Algoritmo | Rede | Params | Passos | Score médio | Mediana | p95 | Máx | Tabuleiro cheio | Com filtro | ms/inf |
|---|---|---|---|---|---|---|---|---|---|---|
| aleatório + máscara | — | — | 0 | **1,21** | 1 | 4 | — | 0% | — | — |
| PPO | | | | | | | | | | |
| DQN | | | | | | | | | | |
| Rainbow | | | | | | | | | | |
| A2C | | | | | | | | | | |
| ACER | | | | | | | | | | |

**Os últimos modelos treinados moram neste repositório**, em [`models/`](models/) — `.keras` para
retomar treino e TFLite fp16/int8 para embarcar no jogo. O [`MODELS.md`](MODELS.md) registra, para
cada arquivo: score, orçamento, semente e o hash do commit que o produziu.

<p align="right">(<a href="#topo">voltar ao topo</a>)</p>

---

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
├── notebooks/                # um .ipynb por algoritmo + a arena
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

### Os notebooks são feitos para o Google Colab

Este é o ambiente de execução de primeira classe do projeto — a linha de comando acima existe para
CI e para quem tem GPU local, mas **todo notebook é escrito para abrir e rodar no Colab**, do zero,
sem nada instalado.

| Notebook | Abrir |
|---|---|
| Arena — treina/agrega tudo e gera o gráfico | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/00_arena.ipynb) |
| PPO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/01_ppo.ipynb) |
| DQN | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/02_dqn.ipynb) |
| Rainbow | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/03_rainbow.ipynb) |
| A2C | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/04_a2c.ipynb) |
| ACER | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/05_acer.ipynb) |
| Ablação de redes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voaneves/snake-arena/blob/main/notebooks/99_ablation_redes.ipynb) |

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
| `tensorflow.contrib.kfac` | não existe. Aposentado (ver [Algoritmos](#algoritmos)) |
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
- [ ] **1** — Núcleo `snakeai/`: ambiente, redes, avaliação, registro, gráfico, testes
- [ ] **2** — PPO refatorado para o pacote, 3 sementes, primeiro `history.json` oficial
- [ ] **3** — DQN unificado (substitui os 6 notebooks antigos por um)
- [ ] **4** — A2C
- [ ] **5** — Rainbow
- [ ] **6** — ACER reescrito
- [ ] **7** — Ablação de otimizadores (sucessora do K-FAC)
- [ ] **8** — Arena: gráfico comparativo, tabela final, painel de tempo de parede
- [ ] **9** — Modelos exportados, `MODELS.md`, integração com o leaderboard humano
- [ ] **10** — Verificação: reprodutibilidade, paridade `.keras` × TFLite, CI

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
