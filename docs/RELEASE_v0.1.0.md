# v0.1.0-alpha — a plataforma completa, a arena pela metade

Doze algoritmos implementados, testados e gerados a partir do mesmo pacote. **Seis já
treinados no orçamento oficial** — os outros seis esperam GPU, não código.

É um **alpha** por uma razão só, e ela está escrita no fim desta página: metade da arena
não foi medida, e duas das seis configurações treinadas têm menos de três sementes. O
código está de pé; a comparação ainda não. Marcar como estável um benchmark pela metade
seria o tipo exato de coisa que este repositório existe para não fazer.

![arena](https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/assets/arena_light.png)

---

## O placar

Score médio do modelo do **último passo** — o número oficial, mediana entre sementes.
Tabuleiro 10×10, teto perfeito **97**, 5 M passos de ambiente para todos.

| algoritmo | sementes | | score | tabuleiro cheio |
|---|---:|---|---:|---:|
| **ACER** | 2 | `██████████████████████████░░░░` | **83,96** | 47,3% |
| **PPO** | 3 | `█████████████████████████░░░░░` | **81,50** | 61,4% |
| **ACKTR** | 3 | `████████████████████████░░░░░░` | **78,13** | 60,7% |
| **A2C** | 3 | `██████████████████████░░░░░░░░` | **69,61** | 2,2% |
| **Rainbow** | 2 | `█████████████████░░░░░░░░░░░░░` | **54,46** | 19,9% |
| **DQN** | 3 | `███████████████░░░░░░░░░░░░░░░` | **47,11** | 0,0% |
| _piso aleatório_ | — | `█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` | **1,21** | 0,0% |

**26 execuções oficiais**, 6 curvas históricas do repositório antigo e **5 execuções
listadas fora da arena** com o motivo escrito — protocolo antigo, truncamento por fome
anterior à correção, observação de 6 canais. Excluir em silêncio é pior que incluir; a lista
sai no topo do `arena --all`.

**O ACER lidera, e a liderança não está estabelecida.** As duas sementes dele são 77,84 e
**90,08** — amplitude de 12,24 contra 3,45 do PPO em três sementes. Some a isso a
capacidade: ele é o único do topo com 1,86× a rede dos outros (ver abaixo), e essa
diferença não tem justificativa escrita no código. Duas coisas para desconfiar, as duas
declaradas.

O melhor checkpoint conta outra história, e ela também está publicada: o ACER passou por
**88,25**, o ACKTR por **85,84** e uma semente do Rainbow por **86,13** antes de terminar em
43,50. **RL profundo não melhora monotonicamente** — por isso `last` e `best` são duas
colunas, e não uma escolha.

---

## Média não é vitória, e aqui elas discordam

![quem fecha o tabuleiro](https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/assets/arena_vitorias_light.png)

São dois funcionais da **mesma** distribuição — `E[X]` e `P(X = 97)` — e a ordem muda entre
eles. O **ACER lidera a média e é o terceiro em vitórias**; o Rainbow é o penúltimo em média
e fecha o tabuleiro **nove vezes mais** que o A2C, que tem 15 pontos a mais de score
(19,9% contra 2,2%).

O que cada régua joga fora explica a discordância. A taxa de vitória é um limiar no extremo:
um episódio de 96 conta igual a um de 3. A média usa o episódio inteiro, mas não distingue
"sempre 78" de "metade perfeito, metade zero". Por isso a barra mostra a **repartição
inteira** das causas de fim — e ela diz coisas que nenhum dos dois números diz: o Rainbow
perde 31,6% dos episódios **por fome**, o A2C perde 97,5% **por colisão**. São dois modos de
falhar completamente diferentes.

---

## O agente jogando

Um GIF responde o que a curva não responde: *como* ele perde.

| ACER · 97, tabuleiro cheio | Rainbow · 97, tabuleiro cheio | Rainbow · 0, morreu de fome |
|---|---|---|
| <img src="https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/runs/acer/resnet_small/seed1/episodio_s7.gif" width="240"> | <img src="https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/runs/rainbow/completo/seed1/episodio_s42.gif" width="240"> | <img src="https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/runs/rainbow/completo/seed1/episodio_s7.gif" width="240"> |

Os dois GIFs do Rainbow são **o mesmo modelo, a mesma semente, a mesma avaliação** — só muda
a semente do episódio. Um jogo perfeito e um zero por inanição, lado a lado. É isso que a
média de 54,46 está resumindo, e é por isso que ela não descreve execução nenhuma.

Os GIFs de cada execução ficam versionados em `runs/<algo>/<variante>/seed<N>/` — três por
execução, sementes 7, 21 e 42, sempre as mesmas.

---

## Os doze

| # | algoritmo | estado | o que ele existe para responder |
|---|---|---|---|
| 01 | PPO | **treinado** · 3 sementes | a referência do benchmark |
| 02 | DQN | **treinado** · 3 sementes | a família inteira como flags independentes |
| 03 | Rainbow | **treinado** · 2 sementes | os seis componentes ligados juntos |
| 04 | A2C | **treinado** · 3 sementes | o PPO sem clipping e sem reaproveitar rollout |
| 05 | ACER | **treinado** · 2 sementes | Retrace(λ), IS truncado, região de confiança |
| 06 | AlphaZero | implementado | busca sobre o simulador **real** |
| 07 | MuZero | implementado | a mesma busca, sobre um modelo aprendido |
| 08 | ACKTR | **treinado** · 3 sementes | gradiente natural com K-FAC — a dívida de 2019 paga |
| 09 | DreamerV3 | implementado | treinar dentro de um modelo do mundo |
| 10 | LBC | implementado | exploração **selecionada** em vez de agendada |
| 11 | SOAP | implementado | opções discretas para uma observação não markoviana |
| 12 | ACEKTR | implementado | EK-FAC: os autovalores medidos, não fatorados |

### As ablações, cada uma contra o seu controle

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

## Capacidade: igualada em seis, declarada nos doze

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
está no [`COMPARABILITY.md`](https://github.com/voaneves/snake-arena/blob/v0.1.0-alpha/docs/COMPARABILITY.md):
**quando duas curvas diferem e a capacidade também, o efeito não está isolado**.

### E no eixo do custo

O eixo oficial iguala os *dados vistos*; ele não iguala o *esforço*. Um passo de AlphaZero é
uma busca em árvore inteira, um de DQN é uma passada de rede.

![custo](https://raw.githubusercontent.com/voaneves/snake-arena/v0.1.0-alpha/assets/arena_tempo_light.png)

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

**Três painéis, três perguntas.** O gráfico principal passou a mostrar **um braço por
algoritmo** — ablação sai da figura e continua na tabela, porque desenhada ao lado do
controle, na mesma cor, ela se lê de longe como instabilidade do braço principal. Quantas
ficaram de fora vai no rodapé da própria figura. E entrou o painel de vitórias, pelo motivo
da seção acima. Os três saem de `python -m snakeai.arena --all`, junto com a tabela.

**Os pesos saem do git.** `runs/**/*.keras` e `*.npz` ficam no disco, onde `retomar()` e a
exportação precisam deles, e são publicados **aqui**, como anexo. Os **GIFs entram**: ~240 KB
cada, 26 MB para a arena inteira, e são o único artefato que mostra *como* o agente perde.

---

## O que ainda não está de pé

Isto não é roadmap. É a lista do que **não** se pode afirmar com esta versão na mão — e é
por ela que este release é um alpha:

- **seis dos doze nunca rodaram** no orçamento oficial. AlphaZero, MuZero, DreamerV3, LBC,
  SOAP e ACEKTR estão implementados e testados — não medidos;
- **ACER e Rainbow têm duas sementes.** A amplitude do ACER entre elas é de 12,24 pontos, e
  a do ACKTR em três sementes é de 19,11: **maior que quase toda diferença entre algoritmos
  que a tabela mostra**. A ordem no topo não está estabelecida;
- **a assimetria de capacidade do ACER** (1,86×) é um confundidor declarado e não medido —
  e ele é justamente quem lidera. Alinhar a cabeça do crítico ao actor-critic é ablação, não
  conserto;
- **o posterior do DreamerV3 é 47% do agente** porque recebe um `emb` de 6.400: o encoder
  não reduz resolução. Uma projeção 1×1 antes do achatamento levaria o total de 3,73 M para
  ~2,30 M sem colapsar espaço nenhum. Também é ablação;
- **a exploração do Rainbow** (§2.16) tem uma semente de cada lado e um mecanismo plausível.
  Falta o par de 5 M com três sementes.

---

## Arquivos deste release

| arquivo | conteúdo | tamanho |
|---|---|---|
| `modelos-v0.1.0-alpha.zip` | os 58 `.keras` e `.npz` de `runs/**/modelos/`, com a estrutura de pastas preservada | ≈83 MB |
| `runs-v0.1.0-alpha.zip` | as 31 execuções inteiras — registro, curva, GIFs e modelos | ≈105 MB |
| _código-fonte_ | gerado pelo GitHub | — |

O `history.json`, a `curva.png` e os GIFs continuam **versionados no repositório**: o
registro é o que sustenta qualquer número acima, e ele não depende de baixar anexo nenhum.
O primeiro `.zip` existe porque os pesos **não** estão no git; o segundo é o instantâneo
autossuficiente desta tag, para quem quer estes números depois que o `main` já andou.

Os dois saem de um comando, e a estrutura de pastas vai preservada dentro deles — o que
uma linha de `Compress-Archive` alimentada por pipeline **não** faz: ela achata tudo e
entrega vinte e nove arquivos chamados `last.keras`.

```bash
python tools/empacotar_release.py v0.1.0-alpha     # → os dois .zip

# reproduzir a arena inteira a partir do que já está no clone
python -m snakeai.arena --all                      # → 3 figuras + docs/RESULTADOS.md

# treinar do zero: abra um notebook no Colab ou no Kaggle
notebooks/01_ppo.ipynb                             # ~0,9 h de T4 por semente
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
- `e3b5346` acer | resnet_small | seed1 | training_data

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
- `b080f4a` Plot: show only main arm per algorithm

---

**962 testes**, 2 pulados, nenhuma falha. TensorFlow ≥ 2.20, Keras 3, licença MIT. As
referências — um identificador arXiv conferido um a um contra título e autores — estão em
[`docs/REFERENCIAS.md`](https://github.com/voaneves/snake-arena/blob/v0.1.0-alpha/docs/REFERENCIAS.md).
