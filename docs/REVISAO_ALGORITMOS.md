# Revisão dos algoritmos — pontos de melhoria

Levantamento primeiro, correções depois. A lista inteira continua aqui; os sete itens da
§4 já foram implementados, e cada um tem um teste em `tests/test_revisao.py` que falha no
código anterior — um `git revert` de qualquer correção acende exatamente uma luz.

| item | estado |
|---|---|
| §1.1 truncamento por fome | **corrigido** no `AgentBase` e no DQN/Rainbow; pendente em ACER, DreamerV3, AlphaZero e MuZero (ver a nota no fim da seção) |
| §1.3 `validate()` conferir a curva | **corrigido** |
| §1.4 `avaliar_melhor` e o checkpoint do Dreamer | **corrigido** |
| §2.1 orçamento de gradiente | **preset `PPOConfig.denso()`**, mais `meta["atualizacoes"]` no registro — falta a execução que decide |
| §2.2 variância explicada | **corrigido** (logada por iteração no PPO e no A2C) |
| §2.3 ruído das noisy nets na coleta | **corrigido** |
| §2.5 alvo de valor sem bootstrap | **corrigido** no AlphaZero e no MuZero |
| todo o resto | levantado, não tocado |

Cada achado traz o grau de confiança:

| marca | significado |
|---|---|
| **✔** | conferido linha a linha no código durante esta revisão |
| **○** | achado da revisão, plausível e citado com arquivo:linha, mas não reconferido |
| **?** | hipótese — exige experimento para virar fato |

A pergunta de fundo é a que o `CANAL_DE_FOME.md` deixou em aberto: **por que o melhor
agente para em ~62 de um teto de 97**. A resposta mais provável não é o algoritmo, é que o
orçamento de otimização quase não é gasto — ver §2.1.

---

## 1. Os números do benchmark

Esta seção é a mais séria, porque o valor do repositório é o contrato. Um erro aqui não
piora um agente: invalida a comparação.

### 1.1 ✔ Só o PPO tratava truncamento por fome — **corrigido em parte**
`vec_snake.py:311-330` · `ppo.py:205-210` · `dqn.py:288-292` · `acer.py:185-187` ·
`dreamerv3.py:354` · `alphazero.py:250` · `muzero.py:246` · `mcts.py:184,215`

O ambiente sabe que fome é truncamento: marca `done` mas exporta `trunc_idx`, `final_obs` e
`final_mask` exatamente para o bootstrap. Um `grep trunc_idx snakeai/` devolve **duas**
ocorrências: `ppo.py` e `eval.py`. Todos os demais gravam `done=1` e perdem o `γ·V(s_final)`.

No DQN é pior que perder o bootstrap: o `next_obs` gravado é a observação **do episódio já
resetado** (`vec_snake.py:332` reseta sozinho), então o estado certo nem existe no buffer
para uma correção posterior.

**O que mudou:** `AgentBase.desfaz_truncamento` devolve o estado final verdadeiro e
`done=0` para os truncados, e o DQN/Rainbow passou a usá-lo — os dois defeitos somem
juntos. **O que falta:** ACER, DreamerV3, AlphaZero e MuZero continuam tratando fome como
terminação. Nos três primeiros a correção não é a mesma: eles guardam **sequências**, e o
estado final não cabe no lugar do próximo passo da sequência — precisa de um sinalizador de
truncamento por passo (o `is_last` do Dreamer canônico, o estado absorvente do MuZero).
Está em aberto de propósito, e não escondido.

Por que importa: os algoritmos estão otimizando MDPs diferentes. Quem não faz bootstrap
aprende que sobreviver muito termina em −0,5, o que empurra para episódios curtos — que é
justamente o que a métrica mede. A curva da arena compara "algoritmo + tratamento de
truncamento", não algoritmo. É a maior ameaça à comparabilidade que encontrei, e o
consertável mais direto: o laço de coleta está triplicado (`ppo.py:191-214`,
`dqn.py:285-296`, `acer.py:176-192`), e a correção existe numa cópia só.

### 1.2 ✔ A recompensa real não é a do contrato, e o PPO ainda treina com shaping exclusivo
`vec_snake.py:298-309` · `record.py:54-55` · `ppo.py:168-170,200`

O contrato promete `+1` comer · `−1` morrer · `0` passo. O código também dá **`+2` por
encher o tabuleiro** e **`−0,5` por fome**, nenhum dos dois no `CONTRATO`. E o PPO passa
`shaping_coef` decaindo até 25% do orçamento — 1,25 M passos com uma função de recompensa
que **nenhum outro agente recebe** (nenhum outro chama `step` com shaping).

As chaves `reward_food`/`reward_death` do contrato nunca são lidas por ninguém: `validate()`
compara `env_spec["reward_food"]` com `CONTRATO["reward_food"]`, ou seja, uma constante
consigo mesma.

○ O shaping também não é estritamente potencial — `vec_snake.py:308` zera o delta quando
`dead | won | ate`, quebrando o telescópio do PBRS. A invariância da política ótima passa a
depender do decaimento, não da forma.

### 1.3 ○ `validate()` conferia declarações, não a execução — **corrigido**
`record.py:116,305-331` · `arena.py:53-58`

A revisão construiu um `RunRecord` com **1.000 passos** de curva e `config.total_steps =
5.000.000`: `validate()` devolveu lista vazia. Três buracos somados: o orçamento é lido do
`config` auto-declarado e não de `curve[-1]["global_step"]`; o `env_spec` nasce como cópia
do `CONTRATO` (só `n_channels` passou a ser real, ontem); e a segunda linha de defesa da
arena relê o mesmo campo declarado.

Um `VecSnake(starve_base=300)` grava `starve_base: 100` e valida.

### 1.4 ✔ `avaliar_melhor` media o modelo **final** no DreamerV3 e no MuZero — **corrigido**
`base.py:406-424` · `muzero.py:132-138` · `dreamerv3.py:245,611`

`avaliar_melhor` faz `self.model = m` e chama `avaliar()`. Funciona para quem tem
`politica() = keras_policy(self.model)`. Mas o MuZero declara `model` como *property* com
**setter vazio** (`def model(self, _): pass`) e o Dreamer devolve uma `PoliticaRecorrente`
que lê `self.ator`, não `self.model`. Nos dois casos a atribuição não tem efeito: a coluna
`melhor` do registro é uma segunda medição do modelo final, gravada com o `global_step` do
checkpoint `best`. Nada denuncia — o passo reportado é o certo e o score é do outro modelo.

Corolário: `salvar()` grava `self.model`, e no Dreamer `self.model = self.ator`
(`dreamerv3.py:245`). O `.keras` da pasta da execução **não reproduz o número da curva** —
falta o modelo do mundo inteiro. `retomar()` restaura só o ator e segue contando o
orçamento.

### 1.5 ✔ `avaliar_com_busca` reimplementa o protocolo e reintroduz dois bugs já corrigidos
`alphazero.py:168-196` vs `eval.py:194-230`

Duas linhas contradizem comentários explícitos do `eval.py`, que citam os testes que os
travam:

* `antes = env.score.copy()` antes do `step`, e `coletados[i].append(int(antes[i]))` — é
  exatamente o "perde um ponto nos episódios que terminam comendo, que são precisamente as
  vitórias" de `eval.py:194-197`;
* `vitorias += info["wins"]` acumulado no laço e dividido por `scores.size` — o
  `eval.py:217-219` explica por que isso não corresponde aos episódios medidos; a taxa pode
  passar de 1.

Ainda: `"completo": True` é literal (o estado real é `faltam == 0`), não há guarda de
`max_steps`, e faltam `score_std` e `motivos`. ○ Além disso o método **não é chamado de
lugar nenhum** e o MuZero nem o tem, embora o `COMPARABILITY.md` prometa a coluna "com
busca" — e `muzero.py:80` tenha um `sims_avaliacao` morto.

### 1.6 ✔ Constantes mortas e checagens desligadas
* `eval.py:49` — `PISO_ALEATORIO_10X10 = 1.08`, contra `record.py:69` `PISO_ALEATORIO =
  1.21`. O 1,08 é o número **anterior** à correção de viés que o próprio contrato descreve;
  a constante sobreviveu à correção que documentava. Quem ler `eval.py` erra o ganho sobre
  o piso em 12%.
* `record.py:262-263` — `load()` faz `d.pop("schema_version")` e reinjeta o valor atual. A
  violação `schema_version X != Y` **nunca pode disparar** para um registro lido do disco,
  que é o único caso em que ela importaria.
* `base.py:369-370` — `CONTRATO.get("obs_channels", 5)`: a chave é `"n_channels"`, então o
  fallback é sempre 5. Só morde se `self.env` sumir, mas é um valor fixo escondido dentro
  do metadado que existe para ser conferível por máquina.
* `vec_snake.py:399-401` — `assert not occupied_food.any() or (self.length >=
  self.cells).any()`: o `.any()` da direita é global. Um único ambiente com tabuleiro cheio
  desliga a checagem "comida dentro do corpo" para todos os outros 249.

### 1.7 ✔ A grade de avaliação não é a mesma entre algoritmos
`base.py:109,332-333`

`_proximo_eval = self.global_step + eval_every_steps` reancora no passo **atingido**, não
numa grade absoluta. Como cada agente avança em blocos diferentes (PPO 49.152; A2C/ACKTR
8.192; DQN 256), as avaliações caem em passos diferentes, e a primeira acontece logo após a
primeira iteração em vez de em 250 k. A coluna "passos até 40" do contrato é definida sem
interpolação, com resolução igual à cadência — então herda um viés de até um bloco.

### 1.8 ○ "A mesma sequência de comidas para todos os algoritmos" não se sustenta
`eval.py:140-141` · `vec_snake.py:117-149`

A revisão executou: dois agentes sobre `VecSnake(250, rng=default_rng(123))` têm `env.food`
idêntico no passo 0 e **divergente no passo 50**. A causa é um `Generator` único
compartilhado — quantos números são sorteados depende de quantos ambientes resetaram
naquele passo, que depende da política. Só os 250 tabuleiros iniciais são comuns.

Isso não invalida nada (a amostra continua sendo 1.000 episódios do mesmo protocolo), mas a
redução de variância que o docstring promete não existe. Um `SeedSequence(123).spawn(n)`
por ambiente entregaria de fato variáveis aleatórias comuns — e apertaria a comparação
entre algoritmos, que hoje tem ruído entre sementes maior que a maioria das diferenças.

### 1.9 ○ Execuções `comparable=False` somem da arena em silêncio
`arena.py:30-41,117-127`

`oficiais` filtra por `r.oficial`, `fora` por `not oficial and r.comparable`. Uma execução
com `comparable=False` — a ablação do canal de fome, por exemplo — não cai em nenhuma das
três listas: não entra no gráfico, nem na tabela, nem na seção "execuções que não
entraram". O `COMPARABILITY.md` diz que excluir em silêncio é pior que incluir.

---

## 2. O que provavelmente limita o score

### 2.1 ✔ O orçamento de **passos de gradiente** é minúsculo — é o suspeito número um
| agente | amostras/iteração | atualizações em 5 M passos |
|---|---:|---:|
| PPO | 49.152 | ~2.440 (menos, com early-stop por KL) |
| A2C / ACKTR | 8.192 | ~610 |
| DQN / Rainbow | 512 | ~19.500 |
| AlphaZero | 512 | ~4.900 |
| MuZero | 256 | ~4.900 |

Um PPO de referência em Atari faz ~10⁵ passos de gradiente em 10 M frames. Aqui são duas
ordens de grandeza menos, e o `lr` ainda decai linearmente até 5e-5 ao longo desses ~2.400
passos. O ponto crucial: **redistribuir isso não custa passos de ambiente** — `rollout`
menor com mais `minibatches` e mais `epochs` mantém os 5 M do contrato intactos. É a
mudança de maior razão benefício/risco de toda a lista.

### 2.2 ✔ `vf_clip = 0.2` em unidades absolutas trava o crítico do PPO — **medição no ar**
`ppo.py:54,267-270`

`v_clip = old_val + clip(valor − old_val, −0,2, +0,2)` com `max(...)` das duas perdas: fora
da faixa, o ramo clipado vence e seu gradiente é **zero**. Como `old_val` é fixo dentro de
uma `update()`, o valor de cada amostra se move no máximo ~0,2 por iteração — e são ~102
iterações no orçamento inteiro. Os retornos alvo chegam a dezenas.

Se isso estiver certo, o crítico nunca alcança a escala do retorno, a vantagem GAE vira o
retorno cru menos uma baseline enviesada, e o `λ=0,95` deixa de ajudar. ? Confirmar é
barato: registrar a *explained variance* de `val_buf` contra `ret` por iteração. Se for
~0, está explicado. Note que o A2C **não** tem esse clipping — o que também significa que a
comparação PPO×A2C mede mais coisas do que anuncia (§3.1).

### 2.3 ✔ O Rainbow treinava 5 M passos sem exploração nenhuma
`dqn.py:161,174-177` · `heads.py:79-88` · `rainbow.py:46,52-53` — **corrigido**

`_q_valores(..., training=False)` é o padrão e `_escolher` não passa nada; `NoisyDense` só
usa `w_mu + w_sigma·ε` quando `training=True`. Ao mesmo tempo, `epsilon()` devolve `0.0`
sempre que `noisy=True`, e o `RainbowConfig` tem `eps_start = eps_end = 0`. Resultado: a
política de comportamento é **argmax determinístico** o treino inteiro. O σ é treinado, mas
só adiciona variância ao gradiente em `_passo_treino`.

O código chega a se proteger do caso `noisy=False` sem ε (`rainbow.py:65-69`) — a variante
canônica é a que falha em silêncio. Isto é, sozinho, uma explicação suficiente para o DQN
ser muito pior que o PPO.

### 2.4 ✔ `learn_every` conta iterações vetorizadas: 1 gradiente a cada 256 passos
`dqn.py:68,283-285,299-300`

O comentário diz "passos de ambiente por atualização de gradiente" e o valor é 4. Mas o
laço de `learn_every` é sobre `env.step` **vetorizado em 64 ambientes**, e `_aprender()`
roda uma vez por `iterate()`. O valor efetivo é `4 × 64 = 256` — 64× mais esparso que o
documentado.

E a defasagem da rede alvo é medida na mesma moeda errada (`dqn.py:302-305`):
`target_update = 2.000` passos de ambiente ÷ 256 = **7,8 atualizações de gradiente** de
defasagem (31 no Rainbow), contra a ordem de 2.000 das implementações de referência. Com o
alvo praticamente igual à rede online, o `double` perde o efeito e o alvo deixa de ser
ponto fixo. Os dois botões estão acoplados sem que nada denuncie.

○ A causa provável de ambos: `replay.py:115-118,202,208-212` faz laço Python por transição
e ~9.200 iterações de interpretador por atualização. Baixar `learn_every` hoje tornaria o
treino inviável — o gargalo é a memória, não a GPU.

**A correção teve um segundo tempo, e ele quase passou batido.** Trocada a moeda de
`target_update` para atualizações de gradiente, o DQN teve o valor recalculado junto
(2.000 → 250, ~1,3% do orçamento de ~19.500 atualizações). O Rainbow não: ficou com os
8.000 canônicos do paper, que na moeda nova são **41% do orçamento** — duas sincronizações
no treino inteiro. É o mesmo defeito pelo outro extremo: um alvo congelado por 40% do
treino anula o `double` tão bem quanto um alvo colado na rede online. Corrigido para 1.000
(mantendo a razão de 4× que o paper usa contra o DQN base: ~19 sincronizações, ~5% do
orçamento), com um teste que exige pelo menos dez sincronizações para qualquer agente de
valor — para que o próximo que chegar não repita a conta.

A lição geral, que vale para o artigo: **mudar a unidade de um hiperparâmetro é mudar todos
os valores absolutos que dependem dela**, e o valor "do paper" deixa de ser o valor certo
no instante em que a unidade muda.

### 2.5 ✔ AlphaZero e MuZero: 62% das amostras tinham alvo de valor sem bootstrap
`alphazero.py:242-254` · `muzero.py:238-250` — **corrigido**

Com `rollout=16` e `n_step=10`, a condição `if t + k + 1 < T` é falsa para todo `t ≥ 6`:
**10 dos 16 passos** de cada rollout viravam retorno truncado puro, como se o fim da janela
de coleta fosse fim de episódio. Em Snake, onde a recompensa só existe ao comer, um retorno
truncado de 3 passos é quase sempre exatamente 0 — e esse valor é o bootstrap do rollout
seguinte e o `v_raiz` da busca.

Uma correção à revisão original, que atribuía o problema ao `k` vazando do laço: **o índice
estava certo** — não existe estado `t + n_step` dentro do buffer para `t ≥ 6`, e
bootstrapar ali seria ler fora da janela. O erro era de política, não de índice: em vez de
desistir do bootstrap, o horizonte deve **encolher** e fazer bootstrap no último estado
disponível. É o que o código faz agora (`n = min(n_step, T - 1 - t)`), e sobra um único
passo sem bootstrap por janela — o último, que de fato não tem sucessor. O código é cópia
literal nos dois agentes, e a correção também.

### 2.6 ✔ `clipnorm` no Keras 3 é por variável, não global
`otimizadores.py:58-69`

Todos os agentes passam `max_grad_norm` para `clipnorm=`, que clipa **cada tensor de
gradiente** separadamente; o equivalente canônico é `global_clipnorm=`. Com ~20 tensores, a
norma global efetiva chega a ~2,2 com `max_grad_norm=0,5`, e quando morde distorce a
proporção entre camadas.

○ No ACKTR isso é destrutivo, não cosmético: o `apply_gradients` recebe a direção natural
`F⁻¹∇` já dimensionada por `escala_kl`, e reescalar camada a camada muda exatamente a
relação entre camadas que o K-FAC existe para acertar. O ACKTR herda
`max_grad_norm=0,5` do `PPOConfig` sem sobrescrever.

### 2.7 ○ ACKTR: momento 0,9 sob um passo dimensionado por KL de segunda ordem
`otimizadores.py:81` · `acktr.py:216-221` · `kfac.py:343-356`

`ACKTRConfig.optimizer = "sgd"` e `cria_otimizador` devolve `SGD(momentum=0.9,
nesterov=True)`. O `η` de `escala_kl` supõe que o deslocamento aplicado é `η·Δ`; com
momento, o deslocamento acumulado chega a `η·Δ/(1−0,9)` quando as direções se correlacionam
— e a KL é quadrática no passo, então um fator `k` no deslocamento vira `k²` na KL.

O docstring do módulo registra KL medida 11,8× / 12,4× / 7,5× / 5,2× / 4,4× acima do alvo e
atribui isso à qualidade de `F̃`. `k ≈ 3,4` explica 11,8× sem invocar a Fisher, e a queda ao
longo do treino também (as direções decorrelacionam). Se for isso, o `kl_calibrado` está
compensando um artefato do otimizador com uma malha de realimentação. ? Uma execução curta
com `momentum=0` decide.

### 2.8 ✔ C51 com suporte mal calibrado — **corrigido**, e era pior do que esta seção dizia
`rainbow.py` · `dqn.py:_alvo_c51`

O diagnóstico original: `v_min=−2`, `v_max=60`, 51 átomos → `Δz = 1,24`, e a recompensa de
comer é **+1, menor que o espaçamento entre átomos**. Correto, e confirmado.

**O que faltava é a consequência maior.** Na inicialização os logits são ~0, então a softmax
do C51 é uniforme sobre o suporte e o `Q` inicial é o **ponto médio** dele — não zero. Com
`[−2, 60]` todo estado nascia valendo **+29**. Medido diretamente:

| suporte | ponto médio | `Q` inicial medido |
|---|---:|---:|
| `[−2, 60]` × 51 | 29,0 | **+28,90** |
| `[−2, 60]` × 201 | 29,0 | +29,02 |
| `[−10, 10]` × 51 | 0,0 | −0,03 |
| `[−24, 24]` × 121 | 0,0 | +0,01 |

E +29 é um **ponto fixo do bootstrap**: o alvo de uma transição não terminal é
`r + γ³·29 ≈ 28,6`, que é o que a rede já prevê. A única correção vinha das transições
terminais (`−1`), e como a morte por fome é truncamento (`done=0`, §1.1) mais de 90% dos
fins de episódio não corrigiam nada. Numa execução de controle de 200 mil passos o `Q` médio
ficou preso entre +28,5 e +28,6 do início ao fim, sem se mover — enquanto uma maçã valia +1
sobre essa linha de base, 3% do sinal. O agente aprendia a evitar colisão e nada mais, e o
score **caía** ao longo do treino (0,93 → 0,65) com a fome subindo para 96%.

Corrigido para `[−24, 24]` com 121 átomos: simétrico (`Q` inicial zero), largo o bastante
para o retorno de um jogo perfeito — 97 maçãs a ~10 passos cada com γ=0,995 rendem **20,3**,
e não os ~14 estimados aqui — e com `Δz = 0,4`, exatamente a resolução do C51 canônico do
Atari. Uma maçã passa a valer 2,5 átomos.

Três testes prendem isso: a simetria do suporte, o `Q` inicial perto de zero, e o `Δz`
contra a recompensa.

### 2.8b ✔ A projeção do C51 indexava fora do suporte — **corrigido**
`dqn.py:_alvo_c51`

A frase "a projeção em si está correta" acima estava errada, e o defeito era **latente**.
`tz` é preso a `[v_min, v_max]`, mas `delta_z` sai de uma subtração em float32 e a divisão
devolve `50,0000476` para o átomo do topo — `ceil` dá 51 e o `np.add.at` estoura o eixo:

| suporte | `b` no topo | `ceil` | |
|---|---:|---:|---|
| `[−2, 60]` × 51 | 49,9999996 | 50 | ok — arredondava para baixo, e escondia |
| `[−20, 20]` × 51 | 50,0000477 | **51** | `IndexError` |
| `[−10, 10]` × 51 | 50,0000477 | **51** | `IndexError` |

A configuração antiga só não quebrava por sorte da aritmética. Trocar o suporte por qualquer
canônico — inclusive o `[−10, 10]` do Atari que a docstring de `suporte_c51` cita como
referência — derrubava o treino. Foi descoberto ao tentar a correção do §2.8. `b` agora é
preso ao índice válido, que é o que a implementação canônica faz.

### 2.9 ○ MuZero: o desenrolar atravessa o fim de episódio
`muzero.py:252-264,290-323`

> **O mesmo defeito estava no buffer de replay, e lá foi corrigido** — ver §2.13. Vale ler
> as duas juntas: a revisão encontrou a classe do bug no MuZero e não procurou o irmão em
> `memory/replay.py`, onde ele custou o Rainbow inteiro. Aqui continua aberto.

`done_b` é usado só no alvo de valor; não há corte no desenrolar de `K=5`. Como o ambiente
reseta sozinho, quando a cobra morre em `t+2` os passos `k=3,4` são de **outra partida** — a
rede `g` aprende a prever recompensa e política de um episódio sem relação causal com aquele
estado oculto. O tratamento canônico (estado absorvente com ação uniforme, recompensa 0,
valor 0) não está lá nem está declarado como omitido. Falta também o `scale_gradient(loss_k,
1/K)`, então mudar `unroll` muda a taxa de aprendizado efetiva — e `unroll` é o nome da
variante.

### 2.10 ○ DreamerV3: quatro desvios do canônico
`dreamerv3.py:331,422-425,371-373,498` · `nets/dreamer.py:43,341-346`

* **`unimix` nunca chega ao ator.** O docstring diz "toda categórica recebe 1% de uniforme";
  no código só as latentes do RSSM recebem. A política de coleta e a do sonho usam logits
  crus, e com `ent_coef=3e-4` e 3 ações o ator pode saturar.
* **Free bits aplicado depois da média** (`maximum(kl_free, reduce_mean(kl))` em vez de
  elemento a elemento). Vira bang-bang, e com as KLs medidas de 5–7 nats o termo está, na
  prática, desligado.
* **Latente zerado a cada janela** amostrada do buffer, mesmo no meio de um episódio, sem
  carry nem burn-in. O modelo aprende que "às vezes o passado some", e há descasamento com a
  coleta e a avaliação, onde `(h,z)` atravessa centenas de passos.
* **λ-returns calculados com o crítico EMA** em vez do crítico rápido (? confirmar contra a
  referência da v3 — o DreamerV2 usava o lento).

### 2.11 ? Capacidade e representação — as hipóteses que exigem ablação
* **Cabeça de política linear.** `registry.py:106-124`: a política é `Conv2D(4,1×1) → GN →
  ReLU → Flatten(400) → Dense(3)`, um mapa **linear** sobre 4 canais comprimidos; o crítico
  ganha um `Dense(256, relu)` no caminho. A assimetria não está justificada em lugar nenhum,
  e Snake tardio é planejamento de caminho.
* **"Egocêntrica" é só rotação.** `vec_snake.py:189-197` aplica `rot90` e nada mais: a
  cabeça fica em posição arbitrária do plano 10×10, e o tronco precisa relocalizá-la a cada
  passada antes de avaliar três ações locais. Centrar por `np.roll` com um canal de parede
  tornaria a leitura local.
* **A fome é invisível por contrato** — e o `CANAL_DE_FOME.md` testou **uma** codificação
  (plano constante), não a hipótese. Um escalar concatenado depois do tronco, ou o relógio
  como distância-ao-limite espacial, continuam não testados.

### 2.12 ○ Onde o custo não compra qualidade
* **AlphaZero/MuZero:** ~60–70% dos rótulos produzidos pelo MCTS **nunca entram em nenhum
  gradiente** (buffer grande, uma atualização por iteração). E não há reanálise — os alvos
  no buffer vêm de uma rede até ~100 gerações mais velha. Subir `epochs_por_iter` custa GPU
  e não custa orçamento.
* **MuZero:** `_busca` roda `f` na raiz e joga o resultado fora, porque o `mcts.run` a
  recalcula.
* **DreamerV3:** `PoliticaRecorrente` roda o ator duas vezes por passo, uma delas em eager,
  descartando a ação já amostrada dentro do grafo. E `num_envs=64` paga ~8× mais despacho
  de kernel por passo de ambiente que o PPO; subir para 256 mantém `train_ratio` idêntico.
* **ACKTR:** `extract_patches` sobre o lote inteiro gera ~1,4 GB por camada convolucional.
  A Fisher é uma média — um subconjunto do lote bastaria.

---

### 2.13 ✔ A janela de n passos atravessava o fim do episódio — **corrigido**
`memory/replay.py:_add_um` · `dqn.py:iterate`

É o §2.9 outra vez, em outro arquivo — e este custou o Rainbow inteiro.

O `desfaz_truncamento` (§1.1) grava `done=0` na morte por fome, **corretamente**: é
truncamento, e o alvo precisa bootstrapar do estado final verdadeiro. Mas o buffer de n
passos usa `done` como a **única** marca de fim de episódio. Com `done=0` a fila não é
esvaziada, e as janelas seguintes somam recompensas do episódio **seguinte**, com um
`next_obs` de outra trajetória. Medido com γ=0,9, `n_steps=3`, dois episódios de três passos
com recompensas `[1, 2, 4]` e `[100, 200, 400]`:

| janela | guardado | correto |
|---:|---:|---:|
| 0 | 6,04 | 6,04 |
| 1 | **86,60** | 5,60 |
| 2 | **256,00** | 4,00 |

Duas de cada três janelas de cada fronteira saíam contaminadas. **O DQN base é imune porque
usa `n_steps=1`** — cada janela é um passo — e é exatamente por isso que o DQN aprendia
(47,67) e o Rainbow, com `n_steps=3`, não saía do piso. Neste ambiente mais de 90% dos
episódios acabam por fome, então a contaminação é a regra, não a exceção.

A correção separa os dois conceitos que estavam colapsados num campo só: `done` é
**terminação**, e vai para o alvo de TD; `fim` é a **fronteira do episódio**, e é ela que
corta a janela. `add_batch(..., fim=None)` mantém `fim = done` por padrão, então nada muda
para quem não trunca. O `AgentBase.aplica_truncamento_no_rollout` já dizia a regra em
palavras — *"o `done` continua 1: a fronteira do episódio é real dentro do buffer, e é ela
que impede o retorno de atravessar para o episódio seguinte"* — mas só o caminho do PPO a
seguia.

Junto veio um erro de segunda ordem: `_alvo` descontava por `γ**n_steps` mesmo nas janelas
esvaziadas na fronteira, que são mais curtas. Com `done=1` isso era anulado pelo
`(1 − done)`; com truncamento, `done=0`, o bootstrap acontece de verdade e o desconto errado
desloca o alvo. O buffer agora guarda `n_real` e o alvo usa `γ**n_real`.

### 2.14 ✔ O checkpoint não voltava do disco — **corrigido**
`nets/heads.py` · `base.py:modelo_melhor`

As cabeças `dueling` e C51 usavam `layers.Lambda(lambda t: t − mean(t))`. O Keras 3 recusa
desserializar um lambda Python (`ValueError: Requested the deserialization of a Lambda
layer...`) sem `safe_mode=False`. Como `avaliar_melhor()` recarrega o checkpoint `best`
**no fim do treino**, o erro chegava depois do orçamento inteiro gasto — **8.931 s de GPU
perdidos** numa execução do Rainbow, na última linha.

| configuração | antes | depois |
|---|---|---|
| Rainbow (dueling + C51 + noisy) | `ValueError` | carrega, saída idêntica |
| DQN + `dueling=True` | `ValueError` | carrega, saída idêntica |
| DQN base | ok | ok |
| C51 sem dueling | ok | ok |

Duas coisas escondiam o defeito: o DQN base não liga `dueling`, e a cabeça C51 só usa
`Lambda` no ramo dueling. O Rainbow é o primeiro agente com os dois ligados. **Qualquer
ablação de DQN com `dueling=True` teria batido também** — era uma mina em todo o eixo.

A correção é `CentraNaMedia`, camada registrada com `@keras.saving.register_keras_serializable`
— a mesma solução que `nets/muzero.py` já usava, com o comentário certo (*"Camada em vez de
`Lambda`: `Lambda` não sobrevive a `save`/`load` sem gambiarra"*), no arquivo errado. Dois
testes prendem: o round-trip por disco sem `safe_mode=False`, e a ausência de `Lambda` nas
cabeças.

### 2.15 ✔ `eps_start` era ignorado em silêncio sob `noisy=True` — **corrigido**
`dqn.py:epsilon`

`epsilon()` devolvia `0.0` incondicionalmente sempre que `noisy=True`. O padrão está certo —
é o §2.3, e é o que o paper manda —, mas o efeito colateral é que `RainbowConfig(eps_start=0.1)`
não dava erro **e** não tinha efeito. Silêncio é o pior dos dois mundos, e era o que impedia
medir a hipótese que sobrou.

Agora existe `eps_mesmo_com_noisy: bool = False`. O padrão não mudou — a composição canônica
do Rainbow segue sem ε, e as ablações de DQN com `noisy` também — mas o botão existe e é
declarado.

### 2.17 ✔ A política do checkpoint não colapsava a distribuição do C51 — **corrigido**
`agents/dqn.py:politica_do_modelo` · `base.py:keras_policy`

O §2.14 um passo adiante. Com o `Lambda` corrigido o checkpoint volta do disco — e aí quebra
na primeira jogada:

```
ValueError: Dimensions must be equal, but are 250 and 3 ...
input shapes: [250,3], [250,3,121], [250,3,121]
```

`keras_policy` assume saída `(lote, ações)` e faz `tf.where(mask, logits, ...)`. Com
`n_atoms > 0` a rede devolve `(lote, ações, átomos)`. O `DQN` sobrescreve `politica()` — que
colapsa pelo `_q_valores` — mas **não** sobrescrevia `politica_do_modelo()`, e o único
caminho que passa por ali é `avaliar_melhor()`. De novo: erro no fim do treino, orçamento
inteiro gasto.

O teste `test_the_best_checkpoint_can_actually_play` faz o caminho completo — salvar,
recarregar, montar a política, jogar um lote — e prende os dois defeitos de uma vez.

### 2.18 ✔ `meta["atualizacoes"]` gravava o dobro no DQN e no Rainbow — **corrigido**
`agents/dqn.py:153` · `agents/base.py:132,456`

O `DQN` criava `self._atualizacoes` — **o mesmo nome** do contador do `AgentBase` — para
medir `target_update` desde a construção. Aí `DQN.iterate` incrementava (linha 391) e
`AgentBase.train` incrementava **de novo** (linha 456) a partir do `stats` que `iterate()`
devolve. Medido:

| | passos de gradiente reais | `meta["atualizacoes"]` | razão |
|---|---:|---:|---:|
| DQN / Rainbow | 250 | 500 | **2,00×** |
| PPO | 252 | 252 | 1,00× |

O viés valia para **uma família só**, que é o pior caso num repositório cujo eixo declarado
é justamente essa coluna. Consequências já corrigidas em `ORCAMENTO_DE_GRADIENTE.md`: o DQN
sai de 38.908 para ~19.450 atualizações, os pontos por mil atualizações vão de 1,2 para 2,5,
e a afirmação de que PPO × DQN era "o único par com orçamento casado" **foi retirada** — o
PPO tem o dobro. O par casado de verdade é ACKTR × A2C esparso, ~610 contra 611.

A sincronia do alvo nunca esteve errada: ela usa `_desde_alvo`, que é outro contador. O
atributo do DQN passou a se chamar `_passos_gradiente`.

### 2.19 ✔ A prioridade da PER era a entropia cruzada — e ficava **anticorrelacionada** com o erro — **corrigido**
`agents/dqn.py:_passo_treino,_aprender`

No ramo C51 a perda é a entropia cruzada, e `CE = KL(alvo‖pred) + H(alvo)`. O `H(alvo)` não
mede erro: mede quão difusa a rede alvo está no estado sucessor, e com 121 átomos fica preso
perto de `ln 121 = 4,796`. Medido num lote real de 512:

| | média | desvio | correlação com a KL |
|---|---:|---:|---:|
| CE (usada) | 4,7922 | 0,0178 | **−0,9066** |
| KL (correta) | 0,0363 | 0,2382 | +1,0 |

Não era só ruído. A amostra de **maior** erro do lote (KL = 3,88) recebia prioridade 2,4903
e a de **menor** erro (KL = 0,01) recebia 2,5581: a mais surpreendente era amostrada
*menos*. A massa dos 10% maiores dava **0,100** — exatamente uniforme. Um dos seis
componentes do Rainbow não fazia nada, e o pouco que fazia era ao contrário.

Depois da correção (`prioridade = CE − H(alvo)`, com o `H` calculado fora do grafo): a
correlação com a KL vira **+1,0000**, a massa dos 10% maiores vai a **0,434**, e a razão
entre a maior e a menor prioridade do lote sai de 1,02 para **34,1**.

Junto veio o ramo escalar: a prioridade era a perda de Huber, e como `(δ²/2)**α ∝ |δ|**2α`
o expoente efetivo da PER dobrava na região quadrática — `per_alpha=0,6` virava 1,2, e a
ablação "quanto a PER vale" media um `α` que não era o do `config`. Agora é `|δ|`.

*Nota de honestidade:* a fórmula anterior bate com o Dopamine, que também usa a CE como
prioridade. O que quebra aqui é a combinação com 121 átomos e alvos que continuam quase
uniformes por falta de atualizações (§2.18, §2.20). A degradação está medida, não inferida.

### 2.20 ✔ O alvo sincronizava 18,6 vezes no treino inteiro — **corrigido**
`agents/rainbow.py:target_update`

Com a contagem certa, 5 M passos compram ~18.500 atualizações (`num_envs × learn_every =
256` passos por atualização). `target_update = 1.000` dava **18,6 sincronizações** — a
informação de valor se propagava dezenove vezes em cinco milhões de passos. O DQN da Nature
faz ~1.250; o Rainbow do paper, ~6.250; o DQN base deste repositório, 74.

O comentário que defendia os 1.000 argumentava com "~5% do orçamento" — conta feita sobre a
contagem dobrada. Corrigida a contagem, o argumento cai. `target_update = 250`, o mesmo do
DQN base, dá 74 sincronizações.

Continua aberto se **18.500 atualizações bastam** para uma cabeça categórica de 3×121
saídas: uma das revisões mediu que o C51 precisa de ~2.000 atualizações para chegar onde o
Q escalar chega com ~350, uma razão de 5,7× que casa com os 6,1× observados em passos de
ambiente. Se for isso, o botão é `learn_every`, e aí é uma mudança no eixo declarado de
orçamento — não uma correção de bug. Ver §2.16.

### 2.21 ✔ `lr = 1e-4` era a taxa de um orçamento quarenta vezes maior — **corrigido**
`agents/rainbow.py:lr`

Mesmo formato de erro do §2.20: o comentário defendia `1e-4` com "o paper usa LR menor que
o DQN base". Usa — 6,25e-5 — para **200 M de frames**. O orçamento deste repositório é 5 M,
quarenta vezes menor, e a taxa foi herdada sem reescalar.

Medido, com todo o resto igual:

| `lr` | decolagem | score final | estado no fim do orçamento |
|---|---:|---:|---|
| 1e-4 | ~4,6 M | 0,69 | mal começou a subir |
| **3e-4** | **~1,85 M** | **26,99** | inclinação máxima, fome caindo de 49% para 28% |

`3e-4` é o valor do DQN base, que decola aos 750 k neste mesmo ambiente. A execução que
produziu os 26,99 ainda rodou com a PER anticorrelacionada (§2.19) e metade das
sincronizações do alvo (§2.20).

### 2.22 ✔ O máximo de prioridade catracava para sempre — **corrigido**
`memory/replay.py:max_prioridade`

Consequência direta do §2.19, e a revisão tinha avisado para corrigir os dois juntos — não
corrigi, e o efeito apareceu na primeira medição depois.

`max_prioridade` é a prioridade de entrada de uma transição nova, e era o máximo
**histórico**: só subia. Isso é o que a implementação de referência faz, e é inofensivo no
Atari, onde a recompensa é cortada em ±1 e o erro de TD tem teto. Aqui a prioridade passou a
ser a KL do C51, que não tem teto. Medido em 250 iterações logo depois de §2.19:

| iteração | `max_prioridade` | mediana da árvore |
|---:|---:|---:|
| 50 | 4,21 | 0,112 |
| 250 | **4,90** | 0,086 |

Subindo de um lado, caindo do outro, sem nada que faça o máximo voltar. Um pico isolado
fixaria o piso de toda transição nova para sempre. Com o decaimento (0,99 por atualização,
meia-vida de ~70), o máximo estabiliza em ~3,76 em vez de catracar, e um regime de erro
genuinamente alto continua sustentando o valor.

*Correção de leitura:* a razão entre a prioridade de entrada e a mediana da árvore fica em
~26× mesmo depois do decaimento, e isso **não** é patologia — é o comportamento projetado da
PER, em que a transição nova entra no topo e cai para o valor real assim que é amostrada uma
vez. O defeito era só a catraca.

### 2.25 ✔ `n_steps=3` não alcançava a recompensa — **corrigido, e é o achado que destravou tudo**
`agents/rainbow.py:n_steps`

O agente gasta ~12 passos por maçã. Com uma janela de 3, a decisão que o levou até a comida
**sai do retorno antes de a recompensa entrar**: a atribuição de crédito passa a depender só
do bootstrap, e o bootstrap depende das sincronias do alvo — dezenas num treino inteiro. É
por isso que o agente passava mais de um milhão de passos parado em 100% de morte por fome:
ele encontrava a comida e não conseguia ligar o encontro à decisão.

| `n_steps` | decolagem | fome aos 850 k | `γ**n` |
|---:|---:|---:|---:|
| 3 | ~1,85 M | 100% | 0,985 |
| **20** | **~700 k** | **69,8%** | **0,905** |

Com 20, o score de treino vai de 2,24 a 8,45 em 150 mil passos e os episódios crescem de 158
para 330 passos. O `γ**n` caindo para 0,905 também reduz o peso do bootstrap, que era
exatamente o mecanismo que estava faltando.

O 20 vem do **Data-Efficient Rainbow** (van Hasselt et al., 2019), que usa `multi-step 20`
no regime de poucos dados. É um desvio declarado do Rainbow canônico, que usa 3.

**Isto derruba o §2.16.** A hipótese de que o poço era falta de exploração foi testada
diretamente: com `eps_mesmo_com_noisy=True` e ε=1,0 — 66% de desvio do greedy e 64 ambientes
independentes — o agente **continuou** caindo em 100% de fome. Ele explorava de sobra; não
aprendia com o que encontrava. Registrar hipóteses erradas junto com as certas é o ponto de
ter um documento assim.

### 2.23 ○ O regime de reamostragem é um quarto da referência — **aberto por decisão**
`agents/rainbow.py:learn_every,target_update`

O `Kaixhin/Rainbow` treina com lote 32 uma vez a cada 4 passos: **8 amostras sorteadas por
passo de ambiente**, cada transição revisitada ~8 vezes. Com `learn_every=4` e lote 512 nós
sorteamos `512/256 = 2,0`. `learn_every=1` daria exatamente 8,0 — o número não é chute, é o
que a referência implica — ao custo de 4× o trabalho de gradiente.

O mesmo vale para `target_update`: a referência sincroniza a cada **2.000 atualizações**
(8.000 passos ÷ 4); nós estamos em 250, ou seja 8× mais frequente.

Os dois chegaram a ser alterados para os valores da referência e **foram revertidos**. A
execução que funcionou — decolagem aos 700 k — rodou com `learn_every=4` e
`target_update=250`, e mudar os dois junto com o `n_steps=20` mediria a soma em vez do
efeito. Ficam como a próxima ablação, de uma variável.

### 2.24 ✔ O ruído das noisy nets era um sorteio para os 64 ambientes — **corrigido, desligado por padrão**
`nets/heads.py:NoisyDense.por_amostra`

Um `ε` por passada é fiel ao paper porque o paper tem **um** ambiente. Com `num_envs=64` os
64 seguem a mesma política perturbada. Medido, com 60 passadas:

| | P(desvia do greedy) | ambientes independentes |
|---|---:|---:|
| ruído compartilhado (padrão) | 0,299 | **10,9 / 64** |
| ruído por ambiente | 0,293 | **64 / 64** |
| ε = 1,0 | 0,665 | 64 / 64 |

Mesma taxa marginal de exploração, seis vezes a diversidade efetiva. As implementações
distribuídas do mesmo lineage (Ape-X, R2D2) dão a cada ator o seu ruído, então
`ruido_por_ambiente=True` é a **vetorização correta** do paper, não um desvio. Fica desligado
por padrão porque muda a política de comportamento e a execução medida não o usava.

### 2.16 ✗ A exploração do Rainbow neste ambiente — **hipótese REFUTADA**

> **Refutada em 24/08.** Ligado o ε com `eps_mesmo_com_noisy=True` (66% de desvio do greedy,
> 64 ambientes independentes), o agente **continuou** convergindo para 100% de morte por
> fome. Exploração não era o gargalo: o gargalo era atribuição de crédito, e está no §2.25.
> O que está abaixo continua factualmente correto e é a razão de o §2.24 existir, mas a
> conclusão que esta seção tirava estava errada.

Depois de §2.8, §2.8b, §2.13 e §2.14, o Rainbow ainda não decola. O que está medido:

* a entropia das ações de um Rainbow **não treinado** é 0,949 contra 1,099 do aleatório
  uniforme, e o score dele é 0,69 contra 1,04 do piso — ele explora **pior que o acaso**
  desde a inicialização;
* o consenso entre os 64 ambientes é 0,65 contra 0,40 do aleatório, porque `NoisyDense`
  sorteia **um** ruído por passada e o compartilha com o lote inteiro. Isso é fiel ao paper,
  que tem um ambiente; aqui são 64 cópias correlacionadas em vez de 64 exploradores
  independentes;
* o `sigma` **não** colapsa (0,02446 → 0,02417 em 120 mil passos), então a exploração não
  morre — ela encolhe em termos relativos, porque os gaps de `Q` crescem em volta dela.

E o incentivo empurra na mesma direção — mas **não** pelo motivo que esta seção afirmava
antes. A versão de 22/08 dizia que "a fome é truncamento sem penalidade". É falso:
`vec_snake.py:298-302` cobra **−0,5** por inanição, e ainda dá **+2 extra** por vitória —
dois termos que a docstring do módulo (linha 23: "+1 comer, −1 morrer, 0 passo") não
menciona. O que é estranho é a combinação: a fome é penalizada **e** bootstrapada ao mesmo
tempo (`done=0`), ou seja o alvo diz "o episódio continua" e a recompensa diz "você
fracassou", em mais de 90% dos fins de episódio. Não sei dizer se é intencional; sei que a
docstring não descreve o código e que o argumento anterior foi construído sobre a versão
errada. O DQN escapa disso
porque passa ~1 M de passos com ε alto enchendo o buffer de comida acidental; o Rainbow, com
ε = 0, enche o buffer de trajetórias circulando com recompensa zero — e a PER piora, porque
prioriza erro de TD alto, que no começo são as colisões, não as maçãs.

**O experimento que fecha isto** são dois braços de 5 M: o Rainbow como está, e o mesmo com
`eps_mesmo_com_noisy=True, eps_start=1.0, eps_end=0.02, eps_frac=0.2` — a escada do DQN. Se o
segundo decola e o primeiro não, a exploração está confirmada, e vira um desvio declarado do
canônico, do mesmo tipo do eixo de orçamento.

Duas hipóteses já **descartadas** pelo caminho, e vale registrá-las: o `sigma` não colapsa, e
o retorno de n passos em si (fora da fronteira) está aritmeticamente correto.

## 3. Estrutura e manutenção

### 3.1 ○ O A2C não é o controle experimental que o docstring afirma
O módulo diz "o PPO sem as duas coisas que definem o PPO... tudo o mais é igual". Diferem
também em `rollout` (16 vs 96), `lr_start` (7e-4 vs 3e-4), **value clipping** (ausente no
A2C, presente no PPO) e, por consequência, tamanho de lote e número de atualizações. A
diferença entre as curvas é lida como "quanto valem o clipping e o reaproveitamento" e mede
pelo menos cinco coisas. Dado §2.2, é possível que o A2C pareça melhor **porque** não tem o
value clipping — a leitura sairia invertida.

### 3.2 ✔ O laço de coleta está triplicado, e é onde o §1.1 diverge
`ppo.py:191-214` · `dqn.py:285-296` · `acer.py:176-192` — mesmo corpo (step → registra_fim →
acumula scores/wins → `global_step += N`), com o tratamento de truncamento presente em uma
cópia só. O `AgentBase` é o lugar declarado para "o que precisa ser idêntico entre
algoritmos".

### 3.3 ○ `N_CHANNELS` constante em vez do estado do ambiente
`dqn.py:95-98,115` · `acer.py:124,168` — buffers e redes construídos a partir da constante.
Hoje é latente (nem `DQNConfig` nem `ACERConfig` têm `canal_fome`), mas o andaime comum já
propaga o canal: acrescentar o campo daria sufixo de variante e avaliação em 6 canais com
rede de 5. É exatamente o bug que quebrou o notebook 97, esperando em outra porta.

### 3.4 ○ `tf.function` de método presa às variáveis antigas depois de `retomar()`
`alphazero.py:128-135` · `muzero.py:125-128,149-164` — as funções compiladas leem
`self.model` / `self.h` / `self.g` / `self.f` no corpo; `retomar()` substitui os objetos e
`on_model_reloaded` só recria o otimizador. No MuZero o sintoma seria gradiente `None` e
treino parado em silêncio. ? Confirmar salvando, alterando pesos, retomando e conferindo se
algum gradiente vem `None`.

### 3.5 ○ Memória
`trajectory.py:61-70` — o buffer do ACER reserva ~2 GB em float32 (500 segmentos × 2.048
transições × 500 floats). A observação é 3 canais binários + 1 em (0,1] + 1 plano constante:
`uint8`/`float16` ou reconstrução a partir de `CAMPOS_ESTADO` custaria uma fração. No DQN o
mesmo padrão dá ~800 MB, com `next_obs` duplicando `obs` integralmente.

---

## 4. Se fosse para escolher

Ordenado por (impacto esperado) ÷ (custo de implementar e risco de quebrar o contrato):

1. **§2.1 redistribuir o orçamento de gradiente** — não toca no contrato, não muda o
   ambiente, e é a hipótese mais provável para o teto de ~62. Uma semente decide.
   *Disponível como `PPOConfig.denso()`; o padrão continua o antigo até haver medição.*
2. **§2.2 medir a *explained variance* do crítico do PPO** — dez linhas de log; se for ~0,
   o `vf_clip` está confirmado como trava e a correção é trivial.
3. **§1.1 tratamento de truncamento no `AgentBase`** — corrige a comparabilidade **e**
   provavelmente melhora todos os agentes off-policy de uma vez.
4. **§2.3 ligar o ruído das noisy nets na coleta** — uma linha; hoje o Rainbow não explora.
5. **§2.5 o `if t + k + 1 < T`** — uma linha em dois arquivos, 62% das amostras afetadas.
6. **§1.3 `validate()` conferir a curva** em vez do `config` declarado — barato, e é o que
   impede que os itens acima passem despercebidos numa próxima execução.
7. **§1.4 `avaliar_melhor` e o `salvar()` do Dreamer** — a coluna `melhor` de dois
   algoritmos está errada hoje.

O resto entra depois, e boa parte (§2.11) só faz sentido como ablação de uma variável, três
sementes, no mesmo orçamento — do jeito que o `CANAL_DE_FOME.md` fez.

---

## Método

Cinco revisões independentes, por área (on-policy · off-policy e memória · busca e modelo ·
DreamerV3 · infraestrutura), cada uma com instrução de citar arquivo:linha e de marcar o que
não conseguisse verificar. Os achados marcados **✔** foram reconferidos linha a linha
depois. Nenhum arquivo foi modificado.
