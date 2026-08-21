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

### 2.8 ○ C51 com suporte mal calibrado
`dqn.py:83-84`

`v_min=−2`, `v_max=60`, 51 átomos → `Δz = 1,24`. A recompensa de comer é **+1, menor que o
espaçamento entre átomos**. E `v_max=60` não é alcançável: com γ=0,995 e uma comida a cada
~15 passos, o retorno descontado satura perto de 14, então mais de dois terços do suporte
nunca recebe massa. O efeito líquido é um C51 com ~12 átomos úteis e resolução grosseira
justamente no evento mais informativo do jogo. A projeção em si está correta.

### 2.9 ○ MuZero: o desenrolar atravessa o fim de episódio
`muzero.py:252-264,290-323`

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
