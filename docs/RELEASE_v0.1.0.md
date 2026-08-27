# v0.1.0 — a plataforma completa, a arena pela metade

Doze algoritmos implementados, testados e gerados a partir do mesmo pacote. **Seis já
treinados no orçamento oficial** — os outros seis esperam GPU, não código. Este release
existe para dar um nome e uma data ao que já está de pé, e para tirar os pesos de dentro do
git sem que ninguém perca o acesso a eles.

![arena](https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0/assets/arena_light.png)

---

## O placar

Score médio do modelo do **último passo** — o número oficial, mediana entre sementes.
Tabuleiro 10×10, teto perfeito **97**, 5 M passos de ambiente para todos.

| algoritmo | sementes | | score |
|---|---:|---|---:|
| **PPO** | 3 | `█████████████████████████░░░░░` | **81,50** |
| **ACKTR** | 3 | `████████████████████████░░░░░░` | **78,13** |
| **ACER** | 1 | `████████████████████████░░░░░░` | **77,84** |
| **A2C** | 3 | `█████████████████████░░░░░░░░░` | **69,61** |
| **Rainbow** | 2 | `█████████████████░░░░░░░░░░░░░` | **54,46** |
| **DQN** | 3 | `███████████████░░░░░░░░░░░░░░░` | **47,11** |
| _piso aleatório_ | — | `▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` | **1,21** |

**25 execuções oficiais**, 6 curvas históricas do repositório antigo e **5 execuções
listadas fora da arena** com o motivo escrito — protocolo antigo, truncamento por fome
anterior à correção, observação de 6 canais. Excluir em silêncio é pior que incluir; a lista
está no topo da saída do `arena --all`.

O melhor checkpoint conta outra história, e ela também está publicada: o ACKTR passou por
**85,84**, o ACER por **85,77** e uma semente do Rainbow por **86,13** antes de terminar em
43,50. **RL profundo não melhora monotonicamente** — por isso `last` e `best` são duas
colunas, e não uma escolha.

### As ablações, no mesmo eixo

| ablação | contra | resultado | o que isso mede |
|---|---|---|---|
| `ppo · esparso` | PPO | 64,56 vs **81,50** | ~2.400 atualizações de gradiente contra ~38.300, mesmo orçamento de ambiente |
| `a2c · esparso` | A2C | 53,60 vs **69,61** | o mesmo eixo, numa família sem épocas nem minilotes |
| `acktr · kl_nominal` | ACKTR | 64,53 vs **78,13** | a região de confiança sem calibrar erra a KL por ~7× |
| `rainbow · n3` | Rainbow | **0,57** vs 54,46 | a janela canônica de 3 passos não alcança a maçã: 100% dos episódios terminam por fome |
| `rainbow · n3+sem_noisy+eps_greedy` | `rainbow · n3` | **49,17** vs 0,57 | trocar as noisy nets pela escada de ε tira o mesmo agente do chão |

A última linha é o resultado mais barato do lote: **mesma janela de 3 passos, mesma rede,
uma semente de cada lado** — o que muda é só quem explora. Uma semente não fecha a questão;
a diferença qualitativa fecha.

---

## Os doze

| # | algoritmo | estado | o que ele existe para responder |
|---|---|---|---|
| 01 | PPO | **treinado** · 3 sementes | a referência do benchmark |
| 02 | DQN | **treinado** · 3 sementes | a família inteira como flags independentes |
| 03 | Rainbow | **treinado** · 2 sementes | os seis componentes ligados juntos |
| 04 | A2C | **treinado** · 3 sementes | o PPO sem clipping e sem reaproveitar rollout |
| 05 | ACER | **treinado** · 1 semente | Retrace(λ), IS truncado, região de confiança |
| 06 | AlphaZero | implementado | busca sobre o simulador **real** |
| 07 | MuZero | implementado | a mesma busca, sobre um modelo aprendido |
| 08 | ACKTR | **treinado** · 3 sementes | gradiente natural com K-FAC — a dívida de 2019 paga |
| 09 | DreamerV3 | implementado | treinar dentro de um modelo do mundo |
| 10 | LBC | implementado | exploração **selecionada** em vez de agendada |
| 11 | SOAP | implementado | opções discretas para uma observação não markoviana |
| 12 | ACEKTR | implementado | EK-FAC: os autovalores medidos, não fatorados |

---

## Capacidade: igualada em seis, e declarada nos doze

A arena iguala o orçamento de **passos de ambiente**. Ela **não** iguala o tamanho da rede,
e a variação é de 21×. Isso não é defeito — obrigar o DreamerV3 a caber no orçamento de
parâmetros do PPO seria mutilar o que ele é. O defeito seria não dizer.

| notebook | `model` | extras | total | × PPO |
|---|---:|---:|---:|---:|
| `01_ppo` · `04_a2c` · `06_alphazero` · `08_acktr` · `12_acektr` | 180.464 | — | **180.464** | 1,00× |
| `07_muzero` | 154.608 | 118.485 | 273.093 | 1,51× |
| `10_lbc` | 286.896 | — | 286.896 | 1,59× |
| `11_soap` | 300.036 | — | 300.036 | 1,66× |
| `02_dqn` | 333.475 | — | 333.475 | 1,85× |
| `05_acer` | 334.878 | — | 334.878 | 1,86× |
| `03_rainbow` | 1.196.648 | — | 1.196.648 | 6,63× |
| `09_dreamerv3` | 198.403 | 3.530.887 | **3.729.290** | 20,67× |

**Seis dos doze são a mesma rede** — as comparações dentro desse grupo têm capacidade
igualada por construção, e são as únicas que têm. A tabela sai dos próprios construtores
(`python tools/tabela_parametros.py`) e um teste falha se ela envelhecer. A regra de leitura
está em [`docs/COMPARABILITY.md`](https://github.com/voaneves/snake-arena/blob/v0.1.0/docs/COMPARABILITY.md): **quando duas curvas diferem e a
capacidade também, o efeito não está isolado**.

---

## O que esta versão corrigiu

**A paridade do `.tflite` comparava eixos diferentes** (§2.26). A conferência reduzia a saída
da rede a "um escore por ação" só do lado Keras. No Rainbow isso quebrava com `ValueError`
depois de **19.288 s de GPU**, na penúltima célula do notebook; no LBC e no ACER **não
quebrava** — publicava `acoes_iguais` de 0,315 e 0,210, que é o acaso com três ações. Agora a
mesma redução vale nos dois lados, a saída do `.tflite` é casada pela forma do tensor do
Keras, e uma falha de conferência vira `{"erro": ...}` no relatório em vez de levar o treino
junto. Um relatório de exportação anterior a esta correção **não sustenta** a linha
`acoes_iguais`.

**A identidade da execução não podia depender de quem lembra** (§2.25). `Rainbow._variante`
já marcava desvio em `n_steps` e nos componentes; faltava a exploração. Uma execução que
troca as noisy nets pela escada de ε saía como `completo` e dividia `(algo, variant, seed)`
com o padrão — as duas viravam **uma** curva na arena. A marca `eps_greedy` fecha isso, e só
aparece quando o ε está **de fato** agindo: sob `noisy=True` sem `eps_mesmo_com_noisy` ele é
ignorado, e marcar um parâmetro morto seria pior que não marcar.

**A capacidade virou número antes da execução.** `tools/tabela_parametros.py` responde "o
ACER tem quase o dobro do A2C?" sem gastar 5 M de passos para descobrir. Resposta: tem —
1,86×, e a diferença inteira está numa `Conv2D(8, 1)` que não tem justificativa escrita.

**O gráfico principal virou um braço por algoritmo.** Ablação sai da figura e continua na
tabela: o `ppo · esparso` desenhado ao lado do PPO, na mesma cor, se lê de longe como "o PPO é
instável" em vez de "este é o controle de orçamento". Quantas ficaram de fora vai no rodapé da
figura — sumir em silêncio seria a mesma falha que a arena já não comete com as execuções fora
do contrato.

**Os pesos saíram do git.** `runs/**/*.keras` e `*.npz` ficam no disco, onde `retomar()` e a
exportação precisam deles, e são publicados aqui. Os **GIFs entram**: ~240 KB cada, e são o
único artefato que mostra *como* o agente perde — morrer preso no próprio corpo e morrer de
fome dão a mesma linha na curva.

---

## O que ainda não está de pé

Isto não é roadmap, é a lista do que **não** se pode afirmar com esta versão na mão:

- **seis dos doze nunca rodaram** no orçamento oficial. AlphaZero, MuZero, DreamerV3, LBC,
  SOAP e ACEKTR estão implementados e testados — não medidos;
- **ACER e Rainbow têm menos de três sementes.** Com uma semente não existe amplitude, e a
  ordem entre 78,13 e 77,84 não está estabelecida;
- **a assimetria de capacidade do ACER** (1,86×) é um confundidor declarado e ainda não
  medido: alinhar a cabeça do crítico ao actor-critic é ablação, não conserto;
- **o posterior do DreamerV3 é 47% do agente** porque recebe um `emb` de 6.400 — o encoder
  não reduz resolução. Uma projeção 1×1 antes do achatamento levaria o total de 3,73 M para
  ~2,30 M sem colapsar espaço nenhum. Também é ablação;
- **a exploração do Rainbow** (§2.16) tem uma semente de cada lado e um mecanismo plausível.
  Falta o par de 5 M com três sementes.

---

## Arquivos deste release

| arquivo | conteúdo | tamanho |
|---|---|---|
| `modelos-v0.1.0.zip` | os 58 `.keras` de `runs/**/modelos/`, com a estrutura de pastas preservada | ~83 MB |
| `runs-v0.1.0.zip` | as 30 execuções inteiras — `history.json`, `curva.png`, GIFs e modelos | ~110 MB |
| _código-fonte_ | gerado pelo GitHub | — |

O `history.json`, a `curva.png` e os GIFs continuam **versionados no repositório** — o
registro é o que sustenta qualquer número acima, e ele não depende de baixar anexo nenhum.
Os `.zip` existem para quem quer os pesos.

```bash
# reproduzir a arena a partir do que já está no clone
python -m snakeai.arena --all

# treinar do zero: abra um notebook no Colab ou no Kaggle
notebooks/01_ppo.ipynb        # ~0,9 h de T4 por semente
```

---

## Commits

**Fundação**
- `e261065` Fase 1: nova snake-arena
- `e44646b` feat: finalizar notebooks para uso
- `92e0279` fix: corrigindo notebooks
- `edac33d` Atualizações de erros nos notebooks

**Plataforma, protocolo e contrato**
- `88ca4d8` fixing .keras files after runs
- `09133f6` Add plataforma module; support Kaggle & fix moving average
- `793ab4f` Add GPU-hours arena and improve platform handling
- `da04243` Prefer Kaggle in plataforma.detecta; add tests
- `3fa55d6` Enforce eval protocol and record metadata
- `5710cbc` Add variant-suffix support and update notebooks
- `856141c` Add arena images and update RESULTS.md
- `b41d915` Add pre-submission checklist for paper

**Ambiente, observação e exportação**
- `35bd9a8` Acrescentar o sexto canal para fome, além de reformular o log
- `3662c44` Add canal_fome evaluation flag & update README
- `1b02cf2` Pequenos ajustes nos notebooks, com a justificativa de não adicionar o canal de fome
- `7257f75` Detect model input channels for export

**Desempenho**
- `6f1a27a` Make DreamerV3 training run in graph mode
- `2f95802` Make policy step a tf.function for collection

**Execuções e ablações**
- `ddd0b3f` Add PPO resnet_small_denso training runs
- `a33320b` Resultados do ACKTR com duas ablações e mudanças na quantidade de atualizações
- `187285e` Mudança de nomenclaturas e seed1 acktr_calibrado
- `03c1771` Add ACKTR run artifacts and update docs
- `5f2acc3` Normalize ACKTR resnet_small runs
- `7cdfe2c` Refine arena stats and A2C controls
- `6302c72` Add A2C resnet_small_esparso runs and test
- `31786e1` Atualiza resultados DQN base e registra seed1

**A caça aos defeitos do Rainbow**
- `ba10543` Rainbow debug and fixing
- `33dd15c` Fix n-step truncation, C51 support, and eps
- `9ee6196` Fix checkpoint policy for C51/Rainbow
- `3b43ed9` Fix update counting and PER priority
- `f52fa40` Decay PER max_priority; set Rainbow lr=3e-4
- `d711eec` Rainbow fixed and LBC added to the row of models

**Três algoritmos novos**
- `6910151` Add 3 new algorithms: LBC, SOAP and EK-FAC (ACEKTR — novel)

**Esta rodada**
- `b54dbe4` Corrigindo os notebooks, acrescentando ablação do rainbow e arrumando documentação
- `bed4cfb` Corrigindo erro na exportação no modelo, que surgiu no final do rainbow
- `f493bcb` Fix TFLite/Keras parity docs; add Rainbow artifacts
- `6c5b924` Add parameter-count tool and fix Rainbow variant
- `550bfa4` Adicionando os GIFs

---

**954 testes**, 3 pulados, nenhuma falha. TensorFlow ≥ 2.20, Keras 3, licença MIT. As referências — um identificador arXiv conferido um a um contra título e
autores — estão em [`docs/REFERENCIAS.md`](https://github.com/voaneves/snake-arena/blob/v0.1.0/docs/REFERENCIAS.md).
