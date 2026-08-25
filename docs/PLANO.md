# Plano de reestruturação — snake-arena

**Repositório alvo:** `voaneves/snake-arena` (novo)
**Repositórios de origem:** `voaneves/snake-on-pygame` (jogo) e `voaneves/colab-rl` (algoritmos e notebooks)
**Ambiente único:** `VecSnake` (o do notebook de PPO)
**Stack:** Keras 3 com backend TensorFlow (≥ 2.20), sem exceção
**Execução:** todo `.ipynb` é feito para o Google Colab, GPU T4, treino retomável
**Escopo:** todos os algoritmos, incluindo os quebrados
**Entrega:** um `.ipynb` por algoritmo + arena comparativa + modelos versionados no repo

> Revisado em 2026-08-11: o alvo deixou de ser o `snake-on-pygame` e passou a ser um repositório
> novo, o `snake-arena`, que sucede os dois antigos e os referencia. As fases abaixo valem como
> estavam; só mudou o endereço.

---

## 0. Diagnóstico — o que existe hoje

### 0.1 Inventário

| Onde | O que | Estado |
|---|---|---|
| `snake-on-pygame` | `snake.py` (jogo pygame, 37 kB), `utilities/`, `resources/scores.json` (leaderboard humano) | Funciona para humanos; a API de agente é lenta e tem bugs de estado |
| `colab-rl` | `models/dqn.py`, `memory.py`, `utilities/{networks,policy,optimizers,noisy_dense,sum_tree}.py` | TF1 / Keras 1.x. Não roda hoje |
| `colab-rl/notebooks/SnakeAI/Working 2.x/` | 6 notebooks DQN (Adam-CNN3, RMSProp-CNN3, RMSprop-CNN2-KL, RMSprop-CNN4 ×2, RMSprop-PER-Dueling-CNN4) | 5 de 6 terminam em erro |
| `colab-rl/notebooks/SnakeAI/Working 1.x/` | ACER, DQN KFAC-CNN3, DQN KFAC-KL-CNN3 | 2 de 3 em erro |
| `colab-rl/notebooks/SnakeAI/Not Working/` | ACER, Keras, KFAC ×2 | 4 de 4 em erro |
| `colab-rl/models/benchmarking_models/` | 6 CSVs de 10.000 épocas + `.h5` | **Este é o tesouro:** as curvas históricas do DQN |
| `D:\GitHub\SnakeAI` (local) | `SnakeAI_PPO_Keras3.ipynb`, `snakeai_dqn_(...).py` | O PPO é a nova referência; o `.py` é o export do notebook DQN monolítico (3.204 linhas) |

**13 notebooks, 11 com erro na última execução.** Nenhum roda hoje sem intervenção.

### 0.2 O problema central da comparabilidade

Os números do DQN e do PPO **não medem a mesma coisa**:

| | DQN legado | PPO novo |
|---|---|---|
| Métrica registrada | `snake.length`, começando em **3** | `score` = comida comida, começando em **0** |
| Ações | 5 absolutas (com IDLE e movimentos proibidos) | 3 relativas com máscara de morte |
| Estado | canvas 10×10 **ordinal** (0..4), 4 frames empilhados | 5 canais egocêntricos, um passo |
| Recompensa | `-0,005` passo, `-1` morte, **`+length`** ao comer | `+1` / `-1` + shaping potencial |
| Fome | `steps > 50 × length` (cumulativo) | `starve_base = board²` desde a última comida |
| Unidade de tempo | épocas (10.000) | passos de ambiente (30 M) |
| Avaliação | `test()` no fim, ε=0,01 | `evaluate()` periódico, 1.000 episódios, greedy, seed fixa |

Ou seja: um "16" do CSV do DQN é um **score 13**, medido num jogo diferente, com recompensa diferente, contado em unidade diferente. Colocar as duas curvas no mesmo eixo hoje seria desonesto.

### 0.3 Bugs encontrados (a lista de ataque)

**No jogo (`snake.py`)** — estes explicam parte do platô do DQN:

1. **A cabeça some do estado.** Em `state()`, `canvas[body[0]] = HEAD` é seguido de `for part in body: canvas[part] = BODY`, que sobrescreve a própria cabeça. A rede nunca soube onde a cabeça estava — só via um blob de corpo.
2. **Recompensa não estacionária.** `get_reward()` devolve `self.snake.length` ao comer (4, 5, 6, … 30) contra `-1` de morte. O alvo de Q cresce ao longo do episódio; a escala do erro muda com o progresso. É a receita para valor divergente.
3. **O hack `DANGEROUS` destrói o tabuleiro.** `eval_local_safety` escreve o código 4 na **última linha** do canvas — apagando o que existia ali — para codificar 3 booleanos numa posição espacialmente sem sentido para uma CNN.
4. **Codificação ordinal.** Um canal com valores 0..4 faz a convolução tratar `FOOD(1) < BODY(2) < HEAD(3)` como magnitude. Deveria ser one-hot / multicanal.
5. `generate_food()` é chamado a cada passo dentro de `play()`.
6. `nb_actions = 5` inclui `IDLE`, e movimentos proibidos são tratados por tabela em vez de máscara.

**No `colab-rl`:**

7. `networks.py`: `CNN1` e `CNN2` fazem `return model` — nome indefinido. **As funções nunca funcionaram.**
8. `K.set_image_dim_ordering('th')` foi removido do Keras (é o `AttributeError` do notebook "SnakeAI - Keras"). Todo o código assume `channels_first`.
9. `memory.py` PER: `self.memory[self.pos] = experience` numa lista não pré-alocada → `IndexError` (é o erro do notebook PER-Dueling-CNN4).
10. `print(loss.params)` num float → `AttributeError` (notebook CNN2-KL).
11. Três notebooks Working 2.x morrem com `NameError: name 'game'/'nb_frames' is not defined` — células de teste órfãs, coladas de outro notebook.
12. **ACER:** `KerasTensor` passado para API TF que não aceita dispatch (TF2 grafo funcional) e, no 1.x, `expected shape=(None, 256, 100), found (None, 100)` — dimensão de tempo/lote perdida. São dois bugs distintos.
13. **KFAC:** dependia de `tensorflow.contrib.kfac`, que não existe desde o TF2. Não há como "consertar" sem reimplementar.
14. `run_dqn.py`: `tf.Session`, `tf.global_variables_initializer`, `self.args.load` dentro de função (fora de classe), `NB_EPOCH = 50` contra 10.000 dos notebooks.
15. `plot_test.py`: lê `Training DQN.xlsx`, que não está no repo; e `data2` é atribuído duas vezes (o *Moving Max* é sobrescrito pelo *Moving Min* antes de ser usado).
16. Notebooks salvos **sem extensão `.ipynb`** — GitHub não renderiza, badge do Colab não funciona.
17. `game/snake` está no `.gitignore` **e** é submódulo no `.gitmodules`. O submódulo é ignorado.
18. Os CSVs de benchmark têm colunas sem nome (`,0,1,2,3`).

---

## 1. O contrato de comparabilidade

Antes de qualquer código, isto vira um documento no repo (`COMPARABILITY.md`) e um teste automatizado. **Nenhum resultado entra no gráfico se não obedecer:**

| Item | Valor fixado |
|---|---|
| Ambiente | `VecSnake`, tabuleiro 10×10, `starve_base = 100` |
| Observação | 5 canais egocêntricos `(B, B, 5)` |
| Ações | 3 relativas, com máscara de morte imediata |
| Recompensa | `+1` comer, `-1` morrer, `0` passo, shaping potencial com coeficiente decaindo a 0 em 25% do treino |
| **Métrica** | `score` = comida comida (0 no início). Nunca `length` |
| Piso de referência | política aleatória **com máscara** = **1,08** (medido, 1.000 episódios) |
| Teto | 97 (score perfeito num 10×10) |
| Orçamento | **mesmo número de passos de ambiente** para todos. Padrão: 5 M. Episódios reportados em paralelo, nunca como orçamento |
| Avaliação | `evaluate()`, 1.000 episódios, greedy, `seed=123`, **sem** filtro de segurança |
| Filtro de segurança | coluna separada da tabela, nunca na curva principal |
| Sementes | 3 por algoritmo (0, 1, 2). Curva = mediana; banda = IQR |
| Registro | `runs/<algo>/<variante>/seed<N>/history.json`, esquema único |

**Sobre as curvas legadas:** os 6 CSVs de 10.000 épocas são convertidos (`score = length − 3`) e entram no gráfico como **linhas tracejadas em cinza**, com nota explícita de que foram medidos no ambiente antigo. São contexto histórico, não competidores. Quem quiser o número honesto do DQN roda o DQN portado.

---

## 2. Estrutura alvo do repositório

```
snake-arena/
├── snakeai/                      # o pacote — fonte única de verdade
│   ├── env/
│   │   ├── vec_snake.py          # VecSnake (extraído do notebook de PPO)
│   │   ├── single.py             # wrapper de 1 env, API gym, para debug/render
│   │   └── render.py             # GIF sem pygame + ponte para o snake.py
│   ├── nets/
│   │   ├── resnet.py             # tiny/small/base (do PPO)
│   │   ├── classic.py            # CNN2/CNN3/CNN4 portadas e corrigidas
│   │   ├── heads.py              # dueling, noisy dense, C51
│   │   └── registry.py           # "resnet_small", "cnn3", ... por string
│   ├── agents/
│   │   ├── base.py               # laço comum: rollout, log, checkpoint, eval
│   │   ├── ppo.py
│   │   ├── dqn.py                # família: ER/PER, double, dueling, n-step, noisy
│   │   ├── rainbow.py
│   │   ├── a2c.py
│   │   └── acer.py
│   ├── memory/                   # replay uniforme + PER com sum-tree (reescrito)
│   ├── eval.py                   # evaluate, verdict, random_baseline, safety filter
│   ├── record.py                 # esquema de history.json + Recorder
│   ├── plot.py                   # o gráfico comparativo
│   └── export.py                 # .keras + TFLite fp16/int8 + latência
├── notebooks/
│   ├── 00_arena.ipynb            # roda/agrega tudo e gera o gráfico final
│   ├── 01_ppo.ipynb
│   ├── 02_dqn.ipynb
│   ├── 03_rainbow.ipynb
│   ├── 04_a2c.ipynb
│   ├── 05_acer.ipynb
│   └── 99_ablation_redes.ipynb   # CNN2 vs CNN3 vs CNN4 vs ResNet, algoritmo fixo
├── runs/                         # history.json de cada execução (versionado)
├── models/                       # .keras + .tflite do melhor de cada algoritmo
├── legacy/                       # os 13 notebooks antigos, com .ipynb, congelados
├── results/legacy/               # os 6 CSVs normalizados e convertidos para score
├── tests/
├── COMPARABILITY.md
├── MODELS.md
└── README.md
```

**Sobre "vários .ipynb":** cada notebook é próprio e roda sozinho no Colab, mas a **primeira célula clona o repo e importa `snakeai`** em vez de redefinir o ambiente. Assim você tem os N notebooks que quer e uma correção de bug no `VecSnake` vale para todos — em vez de precisar editar 6 cópias. As células de configuração, treino, avaliação e gráfico continuam visíveis e editáveis no notebook.

**Requisitos de todo notebook (o Colab é o ambiente de execução de primeira classe):** roda em GPU T4
do nível gratuito; primeira célula faz clone + `pip install` + checagem de TF/Keras/GPU; sem `pygame`
(visualização sai como GIF); treino retomável por checkpoint, com `USE_DRIVE` para persistir no Google
Drive quando a sessão cair; parâmetros expostos via `# @param`; uma célula final "▶ rodar tudo" que
treina, avalia, exporta e grava o GIF; badge "Open in Colab" no README apontando para o notebook.

---

## 3. Fases de execução

### Fase 0 — Criar o repositório e congelar o passado ✅ *em andamento*
*Baixo risco, faz o resto ficar limpo.*

- **Criar o `snake-arena`**, com a estrutura de pastas, `.gitignore`, `LICENSE`, `requirements.txt`
  e o `README.md` que contempla os dois repositórios antigos. ✅
- Tag `legacy-v1` no `colab-rl`; READMEs dos dois repos antigos apontam para o `snake-arena`.
- Copiar os 13 notebooks para `legacy/`, **adicionando `.ipynb`** ao nome, organizados por estado (`legacy/dqn/`, `legacy/acer/`, `legacy/kfac/`).
- Cada um ganha uma célula de cabeçalho: o que era, por que quebrou (o erro exato da §0.3), e o link para o substituto moderno.
- Normalizar os 6 CSVs → `results/legacy/<variante>.csv` com colunas `episode,length,steps,loss,reward,score` (`score = length − 3`).
- Consertar `.gitmodules` / `.gitignore` do `colab-rl` (o submódulo `game/snake` está sendo ignorado).
- No `snake-on-pygame`: nota no README apontando para cá, e correção dos três bugs de estado/recompensa
  da §0.3 — ele continua sendo o front-end humano e o renderizador dos agentes.

### Fase 1 — O núcleo `snakeai/` (e o porte para Keras 3)
*A fase que decide tudo. Nada depois dela é confiável se ela estiver errada.*

- **Porte para Keras 3 + TensorFlow é requisito de entrada do pacote.** Nada é copiado do legado sem
  passar por: fim de `tf.Session` / `global_variables_initializer`; `channels_last` no lugar de
  `set_image_dim_ordering('th')`; `keras.ops` e `keras.losses.Huber` no lugar de `keras.backend` e
  `tf.losses`; otimizadores customizados reescritos sobre `keras.optimizers.Optimizer`; `NoisyDense`
  reescrita com `add_weight` + `keras.random`; `.keras` no lugar de `.h5`. Backend fixado em código
  com `os.environ["KERAS_BACKEND"] = "tensorflow"`.
- Extrair `VecSnake`, `evaluate`, `verdict`, `random_baseline`, `apply_safety_filter`, `export_model` e `render_episode` do notebook de PPO para módulos.
- Escrever `tests/test_env.py` com as invariantes que hoje só existem no smoke test: `occ >= 0`, `(occ > 0).sum() == length`, comida nunca dentro do corpo, máscara nunca deixa passar morte imediata, episódio termina por fome exatamente em `starve_base`, determinismo por seed.
- `record.py`: um único esquema JSON. Toda curva do projeto sai daqui.
- `plot.py`: a função que produz o gráfico comparativo (mediana + banda IQR + piso + teto + legados tracejados).
- **Portfólio de redes (`nets/`)** — você pediu que a arquitetura entre na comparação:
  - `cnn2`, `cnn3`, `cnn4` portadas de `networks.py`, **corrigidas** (o `return model` indefinido), convertidas para `channels_last` e para entrada de 5 canais;
  - `resnet_{tiny,small,base}` do PPO;
  - cabeças `dueling`, `noisy`, `c51` plugáveis em qualquer tronco;
  - `registry.py` para que qualquer agente aceite `net="cnn3"` ou `net="resnet_small"`.
  - Isso transforma "qual rede é melhor" numa **ablação medida** (notebook 99) em vez de folclore.

### Fase 2 — PPO como referência
- Refatorar o notebook atual para importar do pacote; o notebook cai de ~27 células para ~10.
- Rodar as 3 sementes no orçamento oficial de 5 M passos, com o `evaluate` periódico.
- **Este é o primeiro `history.json` válido.** Ele define o formato e é o número contra o qual todos os outros são lidos.

### Fase 3 — DQN unificado
*Substitui os 6 notebooks Working 2.x por um.*

- Um agente configurável: `replay ∈ {uniform, per}`, `double`, `dueling`, `n_steps`, `noisy`, `net`, `optimizer ∈ {rmsprop, adam, adamw}`, `loss ∈ {huber, mse}`.
- Replay reescrito para coletar de N ambientes em paralelo (o gargalo do DQN antigo era o ambiente pygame de um jogo só; com `VecSnake` isso desaparece).
- Sum-tree do PER pré-alocada — corrige o `IndexError` da §0.3-9.
- Conferência de fidelidade: rodar a variante `epsgreedy` e comparar com o CSV legado convertido. Se o DQN portado ficar **muito acima** do legado no mesmo orçamento, a diferença é atribuída ao ambiente/estado corrigidos — e isso vira um parágrafo do README, não um mistério.
- Variantes oficiais na arena: `dqn`, `dqn+per`, `dqn+double+dueling+per+3steps` (o "quase-Rainbow" do repo antigo).

### Fase 4 — A2C
- Novo (o README do `colab-rl` prometia A2C e nunca teve código). Reaproveita ~80% do rollout do PPO: mesmo GAE, sem clipping, sem múltiplas épocas.
- Barato de escrever depois do PPO, e é o controle experimental que mostra **quanto o clipping do PPO realmente vale**.

### Fase 5 — Rainbow
- C51 (distribucional) + double + dueling + PER + n-step + noisy, sobre a família DQN da Fase 3.
- É o topo natural da linhagem DQN e fecha o arco do repositório original, que já tinha 5 dos 6 componentes espalhados.

### Fase 6 — ACER
*O mais difícil. Deixado por último de propósito.*

- Reescrever do zero em Keras 3: Retrace(λ), importance sampling truncado com correção de viés, replay ratio, e a rede de política com *average policy network* (a restrição de confiança).
- Os dois bugs legados morrem por construção: o `KerasTensor` passado a API TF vira uma `keras.Layer` própria; a incompatibilidade `(None, 256, 100)` vs `(None, 100)` era a dimensão de tempo — o rollout do pacote já é `(T, N, …)` explicitamente.
- **Critério de desistência honesto:** se depois de um esforço delimitado o ACER não superar o piso aleatório de forma consistente, ele entra no README como "implementado, não convergiu neste domínio, curva incluída" em vez de ser escondido. Um resultado negativo medido vale mais que uma pasta chamada "Not Working".

### Fase 7 — K-FAC: aposentadoria explícita
- `tensorflow.contrib.kfac` não existe mais; reimplementar K-FAC do zero é um projeto próprio, não uma correção.
- **Recomendação:** aposentar, documentar o porquê, e substituir a intenção original (*"o otimizador importa?"*) por um **eixo de otimizador medido** dentro do DQN e do PPO: RMSprop vs Adam vs AdamW vs Lion, mesma rede, mesmo orçamento. Responde à mesma pergunta, com código que roda.

### Fase 8 — A arena
- `notebooks/00_arena.ipynb` e `snakeai/arena.py`: lê todos os `runs/**/history.json` e produz:
  - **Gráfico principal:** score de avaliação × passos de ambiente (x em log), mediana de 3 sementes com banda IQR, linha do piso aleatório (1,08), linha do perfeito (97), curvas legadas tracejadas em cinza.
  - **Painel 2:** score × tempo de parede — quem é eficiente por GPU-hora, não só por amostra.
  - **Painel 3:** distribuição final de score (histograma sobreposto, 1.000 episódios).
  - **Tabela final:** algoritmo, rede, parâmetros, passos, score médio/mediana/p95/máx, taxa de tabuleiro cheio, score com filtro de segurança, ms/inferência, tamanho do `.tflite`.
- Um comando: `python -m snakeai.arena --all` regenera tudo.

### Fase 9 — Modelos, documentação e higiene
- `models/`: melhor checkpoint de cada algoritmo em `.keras` + TFLite fp16/int8, com `MODELS.md` (score, orçamento, seed, hash do commit). É o aviso que você quer dar: **os últimos modelos estão neste repositório.**
- README novo: gráfico da arena no topo, tabela de resultados, o contrato de comparabilidade resumido, seção "por que os números antigos eram diferentes", badges do Colab por notebook.
- Integrar com o `resources/scores.json`: o leaderboard humano e o de agentes passam a usar **a mesma métrica** — hoje o humano registra `curr_len − 3` (que já é score) e o DQN registrava `length`. Uma linha de correção resolve, e aí "quem é melhor, você ou a IA?" finalmente tem resposta.
- Corrigir os bugs de `snake.py` da §0.3 (cabeça sumida, recompensa `+length`, `DANGEROUS` na última linha) — o jogo continua sendo o front-end humano e o renderizador dos agentes.
- CI no GitHub Actions: `pytest` + treino de fumaça de 50 k passos que precisa bater score > 3. Pega regressão silenciosa.

### Fase 10 — Verificação
Antes de declarar pronto:

- [ ] Todos os `history.json` passam no validador de esquema
- [ ] `evaluate` com a mesma seed e o mesmo modelo dá o mesmo número em duas execuções
- [ ] Todo algoritmo bate o piso **1,21** com folga (ou está documentado como não-convergido) —
      o número mudou de 1,08 para 1,21 quando o viés de amostra do `evaluate` foi corrigido;
      ver o README, "Por que o piso subiu"
- [ ] Um agente treinado joga no `snake.py` com pygame e o score da tela bate com o do `evaluate`
- [ ] O `.tflite` int8 e o `.keras` dão a mesma ação em 1.000 estados aleatórios — **exceto**
      para políticas com memória (DreamerV3, SOAP), onde um `.tflite` sem estado não consegue
      reproduzir a política e a paridade não é afirmada; ver `COMPARABILITY.md`
- [ ] Os 13 notebooks legados abrem e renderizam no GitHub
- [ ] Um notebook de arena executado do zero num Colab limpo reproduz o gráfico do README

---

## Fase 11 — O que veio depois do plano ✅

Este documento foi escrito para nove algoritmos. Chegaram doze, e a diferença não é uma
extensão da lista: os três últimos entraram para atacar hipóteses que as **medições** deste
repositório levantaram, e não porque estavam num plano.

| algoritmo | a hipótese que ele testa | de onde ela veio |
|---|---|---|
| **LBC** (`10`) | a exploração aqui é agendada por uma reta que nunca olhou o resultado | os agendamentos lineares de `AgentBase`, nunca medidos |
| **SOAP** (`11`) | a observação do contrato não é markoviana, e o sexto canal foi a resposta errada | `CANAL_DE_FOME.md`, que fechou com resultado negativo |
| **ACEKTR** (`12`) | a Fisher aproximada do K-FAC subestima a curvatura | `REVISAO_ALGORITMOS.md` §2.7 e a medição de KL do ACKTR |

O que a Fase 1 comprou aparece aqui: os três couberam **sem tocar** no ambiente, no contrato,
no protocolo de avaliação nem no registro. O LBC e o SOAP precisaram de um construtor novo em
`nets/registry.py` e de um laço de coleta próprio; o ACEKTR precisou de **um método**
extraído do `__init__` do ACKTR. Nenhum precisou de um `if` no `AgentBase`.

Duas peças de infraestrutura nasceram junto e são reutilizáveis:

* `snakeai/bandit.py` — UCB não-estacionário com janela deslizante, testado isolado do treino;
* `EKFac` em `snakeai/kfac.py` — subclasse do `KFac`, reaproveitando captura, patches e fatores.

E uma dívida da Fase 1 apareceu só agora: `variancia_explicada` mora em `agents/ppo.py` e é
importada por três agentes que não são PPO, o que faz cada notebook deles embarcar o módulo
inteiro. O lugar certo é `snakeai/eval.py`. Ver `REVISAO_ALGORITMOS.md` §5.2.

---

## 4. Ordem e esforço

| Fase | Depende de | Esforço | Pode paralelizar? |
|---|---|---|---|
| 0 Congelar | — | Baixo | sim |
| 1 Núcleo `snakeai/` | 0 | **Alto** | não — é o gargalo |
| 2 PPO referência | 1 | Baixo | não |
| 3 DQN unificado | 2 | Alto | sim (com 4) |
| 4 A2C | 2 | Baixo | sim (com 3) |
| 5 Rainbow | 3 | Médio | — |
| 6 ACER | 2 | **Alto, risco alto** | sim |
| 7 KFAC aposentado | — | Baixo | sim |
| 8 Arena | 2–6 | Médio | não |
| 9 Docs/modelos | 8 | Médio | — |
| 10 Verificação | 9 | Baixo | não |

**Caminho crítico:** 1 → 2 → 3 → 8 → 9. As fases 4, 6 e 7 penduram fora dele — se o ACER travar, a arena sai sem ele e ele entra depois.

**Custo de GPU:** 5 M passos × 3 sementes × ~7 configurações ≈ 105 M passos. Numa T4 com o preset `small` (~6 mil passos/s) são **~5 horas** de treino no total. Cabe em uma sessão longa de Colab por algoritmo, e todos os agentes herdam o checkpoint retomável do PPO.

---

## 5. Decisões que ainda preciso de você

1. **Orçamento oficial.** Sugiro 5 M passos (~15–25 mil episódios, ~50 min/execução na T4). Alternativas: 2 M (rápido, pode não separar os algoritmos) ou 20 M (definitivo, ~20 h de GPU no total).
2. **`colab-rl`: arquivar ou manter?** Ele tem 30+ notebooks que não são de RL (trading, BERT, DeepDream, GAN…). Sugiro: arquivar a parte de RL apontando para cá, e deixá-lo como o repositório de "notebooks diversos" que na prática ele é.
3. **Piso do ACER.** Confirma o critério de desistência da Fase 6, ou você quer que ele funcione custe o que custar?
4. **Filtro de segurança.** Fica como coluna separada da tabela (minha recomendação, é pós-processamento não aprendido) ou entra na comparação principal?

---

*Nenhuma linha de código foi alterada. Este documento é só o plano.*
