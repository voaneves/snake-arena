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
| §2.36 a região de confiança do ACKTR: o estouro era aquecimento, e o que sobra é um **piso** | **medido, com três conclusões minhas retiradas** — com 300 atualizações o controle cai de 7,4× para **1,2×** (os 4,4×–12,4× da §2 eram o regime frio do K-FAC, sem `cold_iter`). Mas pedir 0,0150 e pedir 0,0020 entregam a **mesma** KL (0,0187 e 0,0185): a KL não responde ao alvo. Suspeito: `escala_kl` usa o gradiente combinado e a KL mede só a política — o tronco compartilhado é movido pelo valor e pela entropia. Braços `so_politica` e `sem_entropia` |
| §2.35 o sorteio do replay do MuZero é uniforme e o Apêndice G prioriza | **implementado, desligado** — `P(i) ∝ |ν − z|^α` com `α = β = 1`; `ν` e `z` são fixos na coleta, então a prioridade nunca é atualizada, ao contrário do PER do DQN. Gravada mesmo desligada, porque mede o quanto busca e jogo discordaram. Braço `priorizado` |
| §2.34 o agendamento de temperatura é o de jogo de tabuleiro, não o de Atari | **implementado, desligado** — com episódios de ~1.200 lances, `temp_passos=30` deixa 97,5% do episódio a τ=0,25 desde a primeira iteração; o Apêndice D, em Atari, amostra o episódio inteiro com τ por passo de treino. Braço `temp_de_treino` |
| §2.33 o valor e a recompensa do MuZero são regressão escalar, e o Apêndice F usa suporte categórico | **implementado, desligado** — two-hot + entropia cruzada, com o suporte dimensionado pelo domínio (teto 60 real) em vez de copiado do `[-300,300]` de Atari, que daria resolução de ~3 pontos perto de zero. Braços `categorico`, `transformacao_h`, `categorico_h` |
| §2.32 o reúso de amostra do MuZero está no regime do **Reanalyse**, sem o Reanalyse | **implementado, desligado por padrão** — 2,0 amostras por estado é o número do Apêndice H, onde ele vem acompanhado de refazer a busca com a rede atual e de rede alvo; `reanalise` traz o primeiro (só a política, só o passo 0). Custo medido: 1,32×–1,57× em CPU, sublinear porque as buscas são em lote |
| §2.31 a perda do MuZero soma os `K+1` passos do desenrolar **sem peso**, e o reúso de amostra está no regime do Reanalyse sem o Reanalyse | **levantado, com conserto opcional pronto** — o passo 0, o único que a métrica oficial mede, vale 14,5% da perda de política com `unroll=5`; `normaliza_unroll` traz o `scale_gradient(loss, 1/K)` do paper e o põe em ~46%. Padrão **desligado** até a medição; braços no `92_muzero_ablacoes` |
| §2.30 o desenrolar de K passos do MuZero atravessava a morte da cobra | **corrigido** — 25% das amostras treinavam a dinâmica contra a recompensa de outra partida; a máscara também recuperou os 31% de passos que eram descartados |
| §2.29 a temperatura do AlphaZero transforma o alvo de política em rótulo duro | **corrigido e é o padrão** — `temp_alvo=1,0` separa o alvo da exploração e `temp_passos=30` traz o agendamento do paper; braço `sem_conserto_do_alvo` |
| §2.28 alvo de valor não normalizado domina o tronco compartilhado do AlphaZero | **corrigido e é o padrão** — `valor_symlog` + `vf_coef=0,5` levam `‖∇v‖/‖∇π‖` de 71× para 7× no `|z|` real; braço `sem_conserto_do_tronco`. **Aplicado também no MuZero** |
| §2.27 PUCT com `Q = 0` para filho não visitado, num jogo de valor positivo | **corrigido e é o padrão** — `fpu="pai"` + `q_normalizado`; o braço `sem_conserto_da_busca` do `93` mede quanto valeu |
| §2.25 janela de n passos do Rainbow | **medido e corrigido** — o padrão passou a 20 |
| §2.26 paridade `.keras` × TFLite | **corrigido** — quebrava no C51 e publicava número de acaso no LBC e no ACER |
| §2.23 razão de reaproveitamento e rede alvo | levantado, **não** tocado — é a próxima ablação |
| §3.6 GIF gravado com o estado interno congelado | **corrigido** |
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

### 2.23 ? A razão de reaproveitamento da memória, e a tensão com a rede alvo
`rainbow.py` — `learn_every = 4` e `target_update = 250`. Os dois estão **declarados como o
valor da execução que funcionou**, não como o valor certo, e é isso que os torna um item
aberto em vez de uma decisão fechada.

**A razão de reaproveitamento.** O Rainbow do `Kaixhin` treina com lote 32 uma vez a cada 4
passos: **8 amostras sorteadas por passo de ambiente**, cada transição revisitada ~8 vezes.
Com `learn_every=4` e lote 512 aqui são `512/256 = 2,0` — um quarto disso. `learn_every=1`
daria exatamente 8,0, e custa 4× o trabalho de gradiente.

**A rede alvo.** Pela unidade que importa — quantas atualizações a rede alvo fica parada —
`target_update=250` sincroniza **8× mais** que os 2.000 da referência. A tensão é real e não
tem valor que a resolva: com poucas atualizações no total, ou o alvo é fiel e propaga pouco,
ou propaga e é infiel.

Nenhum dos dois foi mexido porque a execução que decolou aos 700 k rodou com esses valores, e
trocar duas coisas junto com o `n_steps` mediria a soma. `learn_every=1` é a hipótese mais
forte para a próxima ablação do Rainbow — e ela **substitui** a §2.16 ("a exploração neste
ambiente") como a pergunta em aberto, porque a §2.25 mostrou que o que parecia falta de
exploração era falta de alcance do sinal.

### 2.25 ✔ A janela de n passos do paper deixava o Rainbow no chão — **medido e corrigido**
`rainbow.py` — `n_steps = 3`, o canônico de Hessel et al., era o padrão. Ele produziu uma
execução de 5 M passos parada em **0,57**, abaixo do piso aleatório de 1,21. Com
`n_steps = 20` a mesma configuração faz **65,43**.

| `n_steps` | score final | fim por fome | fim por colisão | decolagem |
|---:|---:|---:|---:|---:|
| 3 | **0,57** | **100,0%** | 0,0% | ~1,85 M (e no braço isolado, nunca) |
| 20 | **65,43** | 12,2% | 87,8% | **~700 k** |

**O diagnóstico não veio do score, veio da coluna do meio.** As duas curvas são igualmente
planas nos primeiros passos, e "não aprendeu" seria a leitura natural para as duas. É falsa
para a de cima: com 100% de fome e **zero** colisões, o agente não falhou em aprender a
sobreviver — ele aprendeu a andar em círculo, que num tabuleiro com máscara de ação é o ponto
fixo mais barato que existe. É o cenário que a nota do `snakeai/eval.py` sobre 100% de fome
descreve, aparecendo pela primeira vez numa execução de orçamento completo.

Curvas de avaliação, em milhões de passos:

```
n=20   0,8 · 0,6 · 0,9 · 5,2 · 26,6 · 42,0 · 39,6 · 37,2 · 64,1 · 61,3 · 65,4
n=3    0,8 · 0,5 · 0,6 · 0,6 · 0,6  · 0,6  · 0,6  · 0,6  · 0,0  · 0,6  · 0,6
```

**Mecanismo.** O agente gasta ~12 passos por maçã. Com uma janela de 3, a decisão que o levou
até a comida sai do retorno antes de a recompensa entrar, e a atribuição de crédito passa a
depender **inteiramente do bootstrap** — que depende das sincronias do alvo, dezenas num
treino inteiro. Com 20 a maçã entra na mesma janela da decisão, e há um segundo efeito que
alivia a §2.23: `γ**n` cai de 0,985 para 0,905, o que **reduz o peso do bootstrap** — que era
exatamente a peça frágil.

**Não é um desvio inventado.** 20 é o `multi-step` do **Data-Efficient Rainbow** (van Hasselt
et al., 2019, [arXiv:1906.05243](https://arxiv.org/abs/1906.05243)), a configuração do
Rainbow para o regime de poucos dados. O contrato daqui dá 5 M passos contra os 200 M do
Rainbow canônico — é o regime de poucos dados, e o valor certo é o de lá.

**O padrão que isto fecha.** É o **terceiro** hiperparâmetro deste arquivo herdado de um
regime quarenta vezes mais longo: o `lr` (§2.21), o `target_update` (§2.20) e agora o
`n_steps`. Nos três a forma do argumento é a mesma. A diferença é que este tem uma referência
que já resolveu o problema no mesmo regime.

**O que a medição não estabelece.** Uma semente de cada lado, e as duas execuções **não têm a
mesma assinatura de pacote** — `ruido_por_ambiente` entrou entre elas, desligado
(`PROCEDENCIA.md`, caso 4). O tamanho do efeito não está estabelecido; a diferença
qualitativa entre 100% de fome e um agente que joga, sim.

**Efeito colateral que vale saber.** Com uma janela maior, a memória só recebe a primeira
transição depois que a janela fecha: 20 passos por ambiente. No orçamento real isso some
dentro dos 20.000 de `warmup_steps`; num teste de fumaça é a diferença entre `loss = None` e
um treino de verdade — e o `None` ali não é bug, é "ainda não havia o que aprender".

Correções que acompanharam:

* o padrão passou a ser `n_steps = 20`, com a medição no docstring do campo;
* `Rainbow._variante` passou a **marcar por construção** qualquer desvio da composição
  canônica. Antes, a execução de `n_steps=3` só se distinguia se quem a rodou lembrasse de
  passar o nome à mão — e esquecer faria as duas curvas virarem uma só na arena. Os nomes
  que a marcação automática produz são os mesmos que as execuções de agosto receberam;
* o braço de controle virou notebook: `94_rainbow_nstep3`.

**A nomenclatura, escrita para não depender de quem lembra.** O ponto de referência de
`_variante` é o `RainbowConfig` **vigente**, não o paper: `completo` nomeia a configuração
que de fato decolou aqui (`n_steps=20`, `lr=3e-4`) e o valor canônico de Hessel et al.
aparece como `completo+n3`, porque é ele o desvio em relação ao que está rodando. Lido de
fora o rótulo parece invertido, e a escolha é deliberada — renomear o padrão moveria os
nomes das execuções de agosto, e o histórico vale mais que a coincidência com a literatura.
Os desvios do canônico estão declarados campo a campo em `agents/rainbow.py`.

Faltava uma marca, e ela custou uma colisão de identidade real. `_variante` cobria
`n_steps`, os quatro componentes booleanos e `n_atoms`, mas **não** a exploração: trocar as
noisy nets pela escada de ε — que é o braço do §2.16, outra política de comportamento
inteira — saía como `completo`. A execução do Kaggle com `noisy=False, eps_start=1.0,
n_steps=3` (score 49,17) gravou `variant: "completo"` e passou a dividir
`(rainbow, completo, 0)` com a execução vigente da mesma semente; como `load_all` agrupa
pela tripla e ignora o caminho, renomear a pasta à mão não resolvia — as duas curvas viravam
uma só. Agora `_variante` marca `eps_greedy` quando a escada está **de fato** agindo, com a
mesma condição de `DQN.eps()`: `eps_start > 0` e (`noisy=False` ou `eps_mesmo_com_noisy`).
Marcar por `eps_start > 0` sozinho seria pior que não marcar — sob `noisy=True` sem
`eps_mesmo_com_noisy` o ε é ignorado (§2.15), e o rótulo afirmaria uma exploração que a
execução não teve. Aquela execução foi renomeada para
`completo+n3+sem_noisy+eps_greedy`, com o motivo gravado em `meta["variante_corrigida"]`;
o teste é `test_the_epsilon_ladder_marks_the_variant_and_a_dead_epsilon_does_not`.

### 2.36 ✔ O passo da região de confiança do ACKTR não desconta o momento
`acktr.py:272-273` · `otimizadores.py:81` · `kfac.py:342-356`

**A §2 deste documento e o docstring do `acktr.py` concluem que a KL medida sai 4,4× a
12,4× acima do alvo porque a Fisher aproximada subestima a curvatura.** Essa conclusão é a
premissa do ACEKTR — o EK-FAC existe aqui para corrigir exatamente esse erro. Há uma
explicação concorrente que não foi considerada, e ela é aritmética.

`escala_kl` devolve `η = √(2·kl_max / Δᵀ∇)`: o passo tal que **uma** atualização `ηΔ`
induz `kl_max`. Ele é atribuído como `learning_rate` de um `SGD(momentum=0.9,
nesterov=True)` (`optimizer = "sgd"`, `acktr.py:161` → `otimizadores.py:81`). Com momento,
o deslocamento em regime não é `ηΔ`: é até `ηΔ/(1−μ) = 10·ηΔ`. Na aproximação quadrática a
KL vai com o **quadrado** do passo, então o estouro fica entre 1× (gradientes
descorrelacionados) e 100× (perfeitamente correlacionados). Os 4,4×–12,4× medidos
correspondem a uma amplificação de passo de 2,1×–3,5×, que é exatamente o que um momento de
0,9 sobre gradientes parcialmente correlacionados produz.

O `baselines` original faz `MomentumOptimizer(lr·(1−momentum), momentum)` — o fator
`(1−μ)` está lá justamente para cancelar isto.

**E o padrão temporal também casa.** O docstring nota que o estouro é *maior no começo e
diminui ao longo do treino*, e usa isso como evidência a favor da Fisher ("o erro encolhe
conforme a média móvel dos fatores amadurece"). Mas gradientes sucessivos são muito mais
correlacionados no começo do treino, quando a política se move consistentemente numa
direção, e vão descorrelacionando depois — o que prevê a mesma curva. Os dois mecanismos
explicam os mesmos dados.

**Um segundo suspeito, no mesmo lugar:** o `clipnorm = 0.5` herdado do PPO
(`ppo.py:185-186`, `max_grad_norm` do `PPOConfig`) é aplicado pelo Keras **por variável,
dentro do `apply_gradients`** — ou seja, sobre a direção **já pré-condicionada**. No
`baselines`, o `max_grad_norm` só existe no caminho SGD "cold", nunca sobre a direção
natural. Direções naturais com `damping = 1e-2` têm norma bem maior que as cruas, então o
clip provavelmente age quase sempre: distorce a razão entre camadas que é a razão de ser do
K-FAC e invalida o `η` que a fórmula da KL calculou.

**O estouro era aquecimento, e o que sobra é um piso.** Medindo com 300 atualizações em
vez de 60 (forma do contrato, `resnet_small`, GPU), o `controle` cai de **7,4× para 1,2×**.
A região de confiança entrega o que pede. Os 4,4×–12,4× da §2 e os 7,4× da primeira rodada
eram o **regime frio do K-FAC** — o `baselines` usa `cold_iter = 100` antes de confiar nos
fatores e este repositório não tem cold start, então as primeiras dezenas de atualizações
usam uma média móvel imatura.

Mas a segunda coluna diz algo que a razão esconde:

| braço | `kl_max` pedido | KL entregue (300 it) | razão |
|---|---|---|---|
| `controle` | 0,0150 | **0,01866** | 1,2× |
| `kl_do_paper` | 0,0020 | **0,01848** | 9,2× |

**Pedidos que diferem 7,5× entregam KL que difere 1%.** A KL entregue não responde ao alvo:
o que existe não é um ganho multiplicativo, é um **piso** de ~0,0185 por atualização. O
`kl_max = 1,5e-2` do repositório está *acima* desse piso, e é só por isso que a razão do
controle parece boa — não porque a região de confiança esteja funcionando, mas porque ela
está pedindo mais do que o piso já entrega de graça.

**A causa provável, e o braço que a testa.** `escala_kl` calcula `Δᵀ∇` sobre o gradiente
**combinado** — `perda = pg + vf_coef·vl − ent_coef·ent` (`acktr.py:258`) — mas a KL é
medida **só na política**. O tronco é compartilhado: o valor e a entropia o movem por conta
própria, e essa parte do deslocamento nenhuma fórmula de KL controla. Um piso independente
do alvo é exatamente o que isso produziria. Os braços `so_politica` (`vf_coef = 0`, entropia
zerada) e `sem_entropia` isolam as duas contribuições.

**As três conclusões anteriores minhas, e por que caíram.** (1) "O momento explica 93% do
excesso" — veio de uma medição em CPU que não replicou. (2) "O resíduo de 12,9× prova que é
a Fisher" — o braço estava com η **saturado no teto** (`lr_start`), medindo o teto e não a
curvatura; o instrumento novo mostra `sem_clip` com η preso em 83% das atualizações e
`η` mediano exatamente 5,00e-01. (3) "O `clipnorm` é um freio acidental" — ele não é freio:
muda a direção por variável, o que muda `Δᵀ∇` e tira η do teto; removê-lo **desliga** a
região de confiança em vez de soltá-la. As três vieram de medições curtas demais ou de
braços cujo passo não estava sob controle da fórmula.

**O que o diagnóstico ganhou por causa disso:** a coluna `no teto` (fração de atualizações
em que η bateu em `lr_start`), o `η` mediano, os braços `sem_teto` e `so_a_fisher` (sem
momento, sem clip e sem teto — o único que isola a aproximação), e os braços `so_politica` e
`sem_entropia`. E o aviso de "sem estouro" passou a dizer que isso é **um resultado**, não
uma falha de medição.

**O que não muda ainda.** Três execuções oficiais do ACKTR estão gravadas e o padrão continua
`momento=0,9`, `descontar_momento=False`, `max_grad_norm=0,5`, `kl_calibrado=True`. A
calibração já absorve o fator empiricamente — o que estava errado era a **atribuição**, e é
ela que este achado corrige.

### 2.34 ✔ O agendamento de temperatura é o de jogo de tabuleiro, não o de Atari
`muzero.py:temperatura` · Apêndice D

O Apêndice D descreve **dois** agendamentos, e o repositório implementou o primeiro:

> *Using a variation of this scheme, in the Atari domain actions are sampled from the visit
> count distribution **throughout the duration of each game, instead of just the first k
> moves**. (…) `T` is decayed as a function of the number of training steps of the network.
> Specifically, for the first 500k training steps a temperature of 1 is used, for the next
> 250k steps a temperature of 0.5 and for the remaining 250k a temperature of 0.25.*

`temp_passos = 30` é o esquema de jogo de tabuleiro: τ = 1 nos 30 primeiros lances, τ = 0,25
depois. Num jogo de tabuleiro, 30 lances é uma fração grande da partida. Aqui, o agente bom
faz episódios de **~1.200 a 1.500 lances** — então 30 lances são **2,5% do episódio**, e os
outros 97,5% são jogados a τ = 0,25 **desde a primeira iteração**, quando o paper estaria a
τ = 1,0 no episódio inteiro. Num jogo de recompensa esparsa, é muito menos exploração do que
o paper prescreve para o domínio parecido.

`temp_esquema = "treino"` traz o segundo: escalar, episódio inteiro, degraus por fração do
orçamento. Fica desligado por padrão, com braço `temp_de_treino` no `92`. **O mesmo vale
para o AlphaZero** (`alphazero.py:temp_passos`), onde a §2.29 introduziu o agendamento por
lance — lá o argumento é mais forte, porque o AlphaZero *é* o algoritmo de jogo de
tabuleiro; mas os episódios são igualmente longos, e o número merece ser medido.

### 2.33 ✔ O valor e a recompensa são regressão escalar, e o paper usa suporte categórico
`nets/muzero.py:build_predicao,build_dinamica` · `muzero.py:_dois_quentes,_perda_escalar` ·
Apêndice F

O Apêndice F não faz regressão escalar. Ele aplica a transformação invertível
`h(x) = sign(x)(√(|x|+1) − 1 + εx)` ao alvo, projeta o resultado num **suporte discreto**
com two-hot — *"a target of 3.7 would be represented as a weight of 0.3 on the support for
3 and a weight of 0.7 on the support for 4"* — e treina cabeças **softmax** com entropia
cruzada, lendo o número de volta pela esperança e invertendo a escala. Aqui as duas cabeças
eram `Dense(1)` com erro quadrático, e a transformação era o `symlog` do DreamerV3.

Por que pode importar, e não é só fidelidade: §2.31 mediu `perda_v ≈ 0,19` em `symlog`, o
que vira uma banda de `[6,7; 17,5]` na escala real — e é esse valor que a árvore soma no
backup. Um MSE tem gradiente proporcional ao erro e escala dependente do alvo; uma entropia
cruzada sobre suporte fixo tem gradiente limitado e calibra uma **distribuição** em vez de
um ponto. É a mesma razão pela qual o C51 existe no lado off-policy deste repositório.

**Dimensionar, não copiar.** O paper usa 601 átomos em `[-300, 300]`, porque um retorno de
Atari é grande. Transplantar esses números daria espaçamento de 1,0 em espaço `h`, isto é
**~3 pontos de resolução perto de zero**, num jogo cujo valor medido vive entre 0 e ~11 — a
cabeça categórica seria mais grosseira que a escalar. O suporte aqui é definido por um teto
na **escala real** (`teto_suporte = 60`) e transformado, o que dá a mesma faixa para as duas
transformações e resolução de ~0,07 perto de zero e ~0,78 perto de 10.

**Uma armadilha que o teste agora protege:** `LIMITE_SYMLOG = 6,0` vale ~402 na escala real
(`e⁶ − 1`), mas `h` cresce como `√x`, então o mesmo 6,0 cortaria o valor em **47** — um teto
que este jogo encosta. `h` ganhou o seu próprio `LIMITE_H = 19,5`, e o teste confere que os
dois descrevem a mesma fronteira real. Um limite errado aqui não levanta exceção nenhuma:
só devolve um valor sistematicamente baixo para a árvore somar.

Braços `categorico`, `transformacao_h`, `categorico_h` no `92_muzero_ablacoes`. Tudo
desligado por padrão — há uma execução de controle a preservar, e o §2.31 tem um conserto de
graça que ataca o mesmo sintoma e vem primeiro.

### 2.32 ✔ O reúso de amostra está no regime do Reanalyse, sem o Reanalyse
`muzero.py:_aprender,_reanalisar` · `tools/diag_reanalise.py` · Apêndice H

**A aritmética que fecha.** Este repositório faz `epochs_por_iter=8 × batch_size=256 =
2048` amostras de gradiente por iteração contra `num_envs=64 × rollout=16 = 1024` passos
novos: **2,0 amostras por estado**. O Apêndice H diz o número dos dois lados:

> *several other hyperparameters were adjusted — primarily to increase sample reuse and
> avoid overfitting of the value function. Specifically, **2.0 samples were drawn per
> state, instead of 0.1**; the value target was weighted down to 0.25 (…); and the n-step
> return was reduced to n = 5 steps instead of n = 10.*

0,1 é o MuZero puro. 2,0 é o **Reanalyse**. Estamos no número do Reanalyse — 20× o reúso
do MuZero puro — e não temos o Reanalyse. Que não é um número de reúso, é maquinário:

> *MuZero Reanalyze **revisits its past time-steps and re-executes its search using the
> latest model parameters**, potentially resulting in a better quality policy than the
> original search. This fresh policy is used as the policy target for 80% of updates (…)
> Furthermore, a **target network** (…) is used to provide a fresher, stable n-step
> bootstrapped target for the value function.*

O Reanalyse existe **porque** reúso alto precisa de alvo fresco. Sem ele, o alvo de visitas
de uma amostra veio de uma rede `g` que já não existe — com buffer de 50 mil e 1.024 passos
novos por iteração, até 49 iterações atrás — e é reamostrado duas vezes contra um modelo que
se moveu. É a descrição do modo de falha medido no §2.31: professor estável em 58–60, aluno
oscilando entre 31,7 e 66,0, `perda_pi` subindo enquanto o `lr` desce.

**O que foi implementado, e o que não foi.** `reanalise` refaz a busca com a rede atual sobre
uma fração de cada minilote e reescreve o alvo de política do **passo 0**, gravando de volta
no buffer para o refresco compor em vez de se perder. Fora do escopo, e dito com todas as
letras:

* os passos `1..K` do desenrolar, porque o buffer guarda só a observação do passo 0. Isto
  cobre exatamente o termo que a métrica oficial mede (§2.31) e deixa os imaginados de fora;
* o alvo de **valor**, porque `z` é um retorno de n passos com bootstrap e refazê-lo exigiria
  a rede alvo do Apêndice H mais o estado em `t+n`, que o buffer não guarda.

É, portanto, o Reanalyse **da política**. Chamá-lo de "Reanalyse" sem esta lista seria
afirmar o Apêndice H inteiro.

**Uma escolha que muda o alvo:** sem ruído de Dirichlet. O ruído da raiz existe para explorar
durante a geração de dados; aqui o que se produz é um alvo, e um alvo não deve depender de um
sorteio. A consequência é que um buffer meio refrescado carrega **duas** distribuições de
alvo, uma sorteada e uma determinística. Qual delas é mais aguda depende do estado do treino —
com a rede treinada o ruído espalha e o refeito sai mais afiado; com a rede recém-iniciada um
sorteio de `Dir(1,1,1)` tem máximo esperado ~0,61 contra o prior quase uniforme, e a direção
se inverte. O teste protege a **reprodutibilidade**, que vale nos dois casos.

**O custo, medido antes de gastar sete horas** (`tools/diag_reanalise.py`, forma do contrato,
2 núcleos de CPU):

| `reanalise` | raízes / coleta | lotes / coleta | s/coleta | s/treino | s/iter | × base |
|---|---|---|---|---|---|---|
| 0,00 | 0,00× | 0,00× | 8,9 | 8,5 | 17,4 | 1,00× |
| 0,25 | 0,50× | 0,50× | 9,3 | 13,7 | 23,0 | **1,32×** |
| 0,50 | 1,00× | 0,50× | 8,8 | 16,5 | 25,3 | **1,46×** |
| 0,80 | 1,60× | 0,50× | 8,8 | 18,4 | 27,3 | **1,57×** |

O interessante é a **sublinearidade**, e ela não é um acidente de medição: as buscas são
feitas em lote. A coleta roda `rollout` buscas batelada de largura `num_envs` (16 × 64); o
Reanalyse roda `epochs_por_iter` buscas batelada de largura `reanalise × batch_size` (8 ×
205). **O número de laços de árvore em Python é 8, qualquer que seja a fração** — só a
largura do lote cresce. Em trabalho de rede 0,80 é 1,6× a coleta; em iterações do laço é
sempre 0,5× dela.

Isso inverte a recomendação conforme o hardware, e vale escrever antes que alguém escolha
errado: **numa GPU o laço em Python domina e a largura do lote é quase de graça, então
0,80 custa quase o mesmo que 0,25** — não há razão para não ir direto ao número do paper.
Numa CPU manda a coluna das raízes, e 0,25 é o ponto razoável. O botão de custo para GPU é
`reanalise_sims`, que encurta o laço; ele é desvio do paper, porque produz alvo de qualidade
menor que o da coleta.

Extrapolando para a execução real (6,8 h, ~4.900 iterações em GPU, onde a busca da coleta é
o termo dominante), o Reanalyse deve custar cerca de meio termo de coleta a mais — algo como
**9 a 10 h**. É estimativa, não medição: o número honesto sai do primeiro braço que rodar.

**Fica desligado por padrão.** Há uma execução de controle a preservar, e — mais importante —
o §2.31 tem um conserto de **graça** que ataca o mesmo sintoma. A ordem certa é
`normaliza_unroll` primeiro; se ele resolver, este maquinário fica registrado e não gasta
GPU nenhuma. Os braços `reanalise_25`, `reanalise_80`, `reanalise_80_sims12` e
`normaliza_e_reanalise` estão no `92_muzero_ablacoes`.

**O que está certo e não precisa mexer:** `n_step = 10` é o valor do MuZero puro (Apêndice G,
Atari); o `n = 5` do Apêndice H vem no pacote do Reanalyse, então trocá-lo sozinho não é
seguir o paper. `coef_valor = 0,25` **é** o número do Apêndice H. E `gamma = 0,997` é
literalmente o do paper, herdado do R2D2.

### 2.31 ✔ A perda do MuZero soma os `K+1` passos do desenrolar sem peso
`muzero.py:_passo` · `tools/diag_unroll.py` · `runs/muzero/unroll5/seed0`

**O sintoma.** A primeira execução de 5 M passos sob o contrato terminou em **49,26**, com o
melhor ponto em **66,05** (3,75 M) — 16,8 pontos acima do final. A leitura fácil é "mínimo
local". A curva desmente:

| | 1,5 M | 2,5 M | 3,0 M | 3,25 M | 3,75 M | 4,0 M | 5,0 M |
|---|---|---|---|---|---|---|---|
| **eval** (rede pura, greedy) | 58,12 | 60,44 | 33,25 | 31,74 | **66,05** | 48,05 | 49,26 |
| **train** (a busca) | 46,79 | 58,30 | 60,34 | 60,16 | 59,79 | 59,34 | 58,02 |
| `perda_pi` | 2,67 | 2,54 | 2,62 | 2,42 | 2,67 | 2,82 | **3,09** |
| `perda_v` | 0,19 | 0,18 | 0,18 | 0,19 | 0,16 | 0,12 | 0,19 |

Três coisas de uma vez. **(a)** O professor está estável: o `train_score`, que é o da busca,
fica em 58–60 de 2,5 M até o fim. Quem oscila é o aluno, e a oscilação é real — 31,7 contra
66,0 num protocolo de 1000 episódios cujo erro padrão é **0,9**, ou seja ~37σ. **(b)**
`perda_pi` **sobe** no último terço **enquanto o `lr` desce** pela reta de decaimento. Isso
descarta passo grande demais: não é o otimizador passando do ponto, é o alvo se afastando.
**(c)** O modo de falha tem assinatura: `fim_fome` é **25,6%** no checkpoint final contra
**5,8%** no melhor, enquanto no treino, com busca, `frac_fome` fica em ~0%. A rede pura
perde o impulso de ir atrás da maçã no fim de jogo e a busca resgata. É falha de
**destilação**, não de busca.

**A previsão pré-registrada do §2.28 foi falsificada.** Escrevi lá que o alvo de valor não
normalizado dominaria o tronco compartilhado do MuZero como dominava o do AlphaZero. Com
`valor_symlog` ligado desde o início, o que se mede é o oposto: `perda_v ≈ 0,19` contra
`perda_pi ≈ 3,09` — a perda de política é **16×** a de valor, e é ela que não converge. O
conserto do §2.28 funcionou; o problema que sobrou é outro, e não é o que eu tinha
apostado.

**A aritmética.** `perda_pi` é uma **soma crua** sobre `K+1` termos:

* o passo 0, que sai de `f(h(o))` — a observação **real**, e o único caminho que
  `politica()` percorre na avaliação oficial;
* `K` passos imaginados, que saem de `f(g^k(...))` — um caminho que a métrica do contrato
  nunca usa.

Nenhum peso separa os dois. Medindo a fatia do passo 0 sobre lotes reais
(`tools/diag_unroll.py`):

| `unroll` | soma crua | com `normaliza_unroll` | `1/(K+1)` |
|---|---|---|---|
| 1 | 45,8% | 45,8% | 50,0% |
| 2 | 29,7% | 46,0% | 33,3% |
| 3 | 22,3% | 46,1% | 25,0% |
| **5** (o padrão) | **14,5%** | 46,0% | 16,7% |
| **10** | **11,0%** | 55,2% | 9,1% |

Com o padrão, **85% do gradiente de política treina um caminho que a métrica oficial nunca
percorre**. E a consequência é contraintuitiva o bastante para valer o destaque: **aumentar
`unroll` sem peso dilui ainda mais o único termo que produz o número do contrato** — a
reação instintiva a uma curva que oscila vai para o lado errado.

**Desvio do paper, e onde ele entrou.** O Apêndice G do MuZero (Schrittwieser et al., 2020,
arXiv:1911.08265v2) é explícito em ter **duas** escalas de gradiente, e lista as duas em
sequência:

> *To maintain roughly similar magnitude of gradient across different unroll steps, we scale
> the gradient in two separate locations:*
> * *We scale the loss of each head by `1/K`, where `K` is the number of unroll steps.*
> * *We also scale the gradient at the start of the dynamics function by `1/2`.*

Este repositório tinha **a segunda** — o `s = s*0.5 + stop_gradient(s)*0.5` no estado oculto,
que controla o gradiente que chega em `h` — e não a primeira. As duas têm nome parecido e
propósito diferente; ter uma delas é fácil de confundir com ter as duas.

Vale registrar uma ambiguidade honesta: lida ao pé da letra, "the loss of each head" incluiria
o passo 0. Só que essa leitura **não muda nada aqui** — dividir a perda inteira por uma
constante, sob Adam, é quase um no-op, porque o segundo momento normaliza a escala do
gradiente; sobraria só o `clipnorm=5` mordendo menos. O que muda a *fatia* do passo 0 é
deixá-lo fora da escala, que é o que o pseudocódigo publicado faz (`gradient_scale = 1.0` na
inferência inicial e `1/len(actions)` nos passos seguintes) e o que `normaliza_unroll`
implementa.

**Conserto:** `normaliza_unroll`, que escala só os `K` termos imaginados por `1/K`. E
`_passo` passou a devolver `perda_pi_0` separada, com `frac_pi_0` no registro — sem
instrumentar o passo 0 não dá para distinguir "a destilação falha no estado real" de "a
destilação falha nos estados imaginados", e a soma esconde os dois casos igualmente bem.

**O padrão continua desligado, de propósito.** Ligar por argumento de paper repetiria o erro
que o §2.27 documenta na direção contrária: lá a convenção do paper estava errada *para este
domínio*. Aqui o argumento é bom mas não é medição, e existe uma execução de controle a
preservar (`unroll5/seed0`, 49,26). O `92_muzero_ablacoes` mede — e leva junto duas previsões
pré-registradas de que **não** vão ajudar, para o registro ser falsificável nos dois sentidos:

* `unroll10` sozinho fica **igual ou pior** que o controle (leva o passo 0 de 14,5% para 11,0%);
* `sims32` **não ganha nada**, porque o professor não é o gargalo — melhorar o professor
  alarga o vão que já não está sendo atravessado.

**Uma hipótese minha que o paper derrubou.** Eu havia proposto `coef_valor: 0,25 → 1,0` como
segunda aposta, pelo argumento de que `perda_v ≈ 0,19` em `symlog` vira uma banda de
`[6,7; 17,5]` na escala real e que valor ruidoso produz contagem de visitas ruidosa. O
Apêndice H diz o oposto, e diz por quê: no MuZero Reanalyse *"the value target was weighted
down to **0.25** compared to weights of 1.0 for policy and reward targets"*, entre os ajustes
feitos "primarily to increase sample reuse and **avoid overfitting of the value function**".
Ou seja, 0,25 já **é** o número do paper, e subir para 1,0 é andar contra ele. O braço
continua no `92` porque a hipótese é testável; deixou de ser a segunda coisa a rodar.

**O que o Apêndice H revelou no lugar, e é mais sério.** Este repositório faz
`epochs_por_iter=8 × batch_size=256 = 2048` amostras de gradiente por iteração contra
`num_envs=64 × rollout=16 = 1024` passos novos — **2,0 amostras por estado**. O paper usa
**0,1** no MuZero puro e sobe para **exatamente 2,0** no MuZero Reanalyse. E o Reanalyse não
é só um número de reúso: ele **refaz a busca** com os parâmetros atuais sobre estados antigos,
usando essa política fresca como alvo em 80% das atualizações, e acrescenta uma **rede alvo**
`f_{θ⁻}` para o bootstrap de valor. Nenhum dos dois existe aqui.

Estamos, portanto, no regime de reúso do Reanalyse **sem** o Reanalyse: alvos de visitas
congelados de uma rede `g` de ~49 iterações atrás, reamostrados duas vezes cada, contra um
modelo que se move. É a descrição exata do modo de falha medido — professor estável, aluno
oscilando, `perda_pi` subindo com o `lr` caindo — e o Reanalyse foi introduzido no paper
precisamente para esse regime. O braço `reuso_do_paper` (`epochs_por_iter=1`) volta ao reúso
do MuZero puro, que é a única forma de sair do regime **sem** implementar o Reanalyse; o custo
é orçamento de gradiente (§2.1), então é uma troca e não um conserto. Implementar Reanalyse de
verdade é a entrada nova da fila em `docs/ANTES_DO_ARTIGO.md`.

Vale dizer o que está **certo**: `n_step = 10` é o valor do MuZero puro (Apêndice G, Atari), e
o `n = 5` do Apêndice H vem no pacote do Reanalyse — trocar só ele não seria seguir o paper.
E `gamma = 0,997` é literalmente o do paper, herdado do R2D2.

### 2.30 ✔ O desenrolar do MuZero atravessava a fronteira do episódio
`muzero.py:collect,_passo`

O `VecSnake` reseta sozinho ao terminar. `_guardar` empilhava `act_b[t..t+K-1]`,
`rew_b[t..t+K-1]`, `pi_b[t..t+K]` e `z[t..t+K]` **sem consultar `done_b`** — então uma
janela que atravessa a morte continua em índices que pertencem a uma partida nova,
sorteada, com a cobra em outro lugar. Simulado no cenário do contrato (`T=16`, `K=5`),
**25% das amostras guardadas atravessam pelo menos uma morte**.

O alvo de valor `z` estava protegido (a máscara `vivo` do laço de n passos), o desenrolar
não. E o dano cai justamente na `perda_r`, que o docstring do módulo chama de "a única
âncora que liga o estado oculto ao mundo": treiná-la contra a recompensa de um jogo que o
latente não tem como conhecer é pior do que não treiná-la.

**Conserto:** uma máscara `vivo` de forma `(T, N, K+1)`, guardada junto com a amostra, que
zera todo passo do desenrolar posterior a uma terminação. A média das perdas passa a ser
sobre os passos reais (`_media_mascarada`), com o denominador sendo a contagem e não o lote
— senão a perda encolheria só porque a janela atravessou uma morte, e o gradiente junto.

**De quebra, o §2.1 do MuZero:** com a máscara, as `K` últimas linhas da janela deixam de
ser descartadas. Antes, `validos = T - K` jogava fora **31% dos passos coletados** — que
continuavam contados no orçamento de 5 M. Sob "os mesmos 5 M passos", o MuZero treinava
sobre ~3,4 M.

### 2.29 ✔ A temperatura serve a dois papéis, e estraga o segundo
`alphazero.py:temperatura,collect` · `mcts.py:politica_das_visitas`

`pi_b[t] = pi`: a mesma distribuição temperada que amostra a ação vira o alvo de treino. No
AlphaZero os dois papéis são separados — a temperatura é exploração na coleta, o alvo é a
contagem de visitas crua.

Com `temp_fim = 0,25` as contagens são elevadas à quarta potência. Nas contagens medidas
com 32 simulações (`[24, 5, 3]`, entropia normalizada 0,662) o alvo vira
`[0,998 · 0,002 · 0,0002]`, entropia **0,015**. Da metade do treino em diante — `temp_frac
= 0,5` — a rede é treinada para confiança máxima no argmax de uma busca de 32 simulações, e
`ent_coef = 0,0` não segura nada.

Dois detalhes que só aparecem lendo o código: `temperatura()` usa `self.frac()`, que é
fração do **treino**, não do episódio — não existe "metade do episódio estocástica"; e o
agendamento canônico do paper (τ alto nos primeiros lances de cada episódio) simplesmente
não existia aqui.

**Estado:** `temp_alvo` separa os dois papéis, `temp_passos` traz o agendamento do paper.
Os dois são o padrão no AlphaZero e no MuZero; braços `sem_alvo_cru` e
`sem_temp_por_lance` no `93_alphazero_ablacoes`.

### 2.28 ? O alvo de valor não normalizado domina o tronco compartilhado (AlphaZero e MuZero)
`alphazero.py:_passo` · `ppo.py:303` · `nets/registry.py:127`

`perda = perda_pi + vf_coef * perda_v` com `vf_coef = 1,0`, `perda_v` em MSE sobre um
retorno descontado **não normalizado** e `perda_pi` em entropia cruzada sobre 3 ações. O
AlphaZero original treina o valor contra o resultado da partida em `[-1, 1]`, onde os dois
termos nascem comparáveis; aqui o alvo vale ~9 em 1 M de passos e cresce com o agente.

Medido em `tools/diag_balanco_perdas.py`, a razão entre as normas dos gradientes **no
tronco** cresce **linearmente com a escala do valor**, que por sua vez cresce quando o
agente melhora. Com `valor_symlog` ela cresce só logaritmicamente.

E agora há o dado da execução de 5 M passos, que é melhor que qualquer proxy:
`perda_v/perda_pi` = **57,6×** depois de 4 M — porque `perda_pi` desabou para 0,016
(a rede reproduz o alvo quase perfeitamente) enquanto `perda_v` **subiu** de 0,34 para 1,0
e nunca convergiu, sobre um `valor_raiz` de 3,2. No `|z|` dessa execução:

| `valor_symlog` | `vf_coef` | `perda_v/perda_pi` | `‖∇v‖/‖∇π‖` no tronco |
|---|---:|---:|---:|
| não *(a execução de 5 M)* | 1,0 | 20,4× | **71,4×** |
| não | 0,5 | 10,2× | 35,7× |
| sim | 1,0 | 1,5× | 14,1× |
| **sim** | **0,5** | 0,8× | **7,0×** |
| sim | 0,25 | 0,4× | 3,5× |

Referência de saudável: 4,8× num agente que quase não come, onde nada está quebrado. O
`symlog` faz o grosso do reequilíbrio e `vf_coef = 0,5` — o valor do PPO — chega na faixa.
**`0,25` passaria do ponto:** a `perda_v` não convergiu, e enfraquecer mais o valor piora a
busca, que depende dele para avaliar folhas. Foi por isso que o braço curado mudou de 0,25
para 0,5 depois que a execução de controle terminou.

O PPO escapa por normalizar a vantagem por minilote, o que torna o gradiente de política
invariante à escala do valor, e por usar `vf_coef = 0,5`. O A2C e o ACKTR herdam a mesma
normalização. AlphaZero e MuZero não normalizam nada — e o MuZero tem o mesmo `_passo` com
o mesmo problema.

**No MuZero é pior, e agora está medido.** As três perdas são **somas** sobre o desenrolar,
e a de política é uma entropia cruzada presa perto de `ln 3` em cada passo. Medindo o
gradiente que chega na representação `h` — o tronco que as três dividem — no mesmo `|z|`:

| `valor_symlog` | `coef_valor` | `‖∇v‖/‖∇π‖` | `‖∇r‖/‖∇π‖` |
|---|---:|---:|---:|
| não *(era o estado do código)* | 0,25 | **84,4×** | 20,7× |
| sim | **0,25** *(o padrão hoje)* | **28,7×** | 18,9× |
| sim | 0,5 | 57,5× | 18,9× |
| sim | 1,0 | 115,0× | 18,9× |

Duas leituras. A primeira: o `symlog` corta a razão por 3, e **subir** o `coef_valor` para
o 0,5 do AlphaZero pioraria — a intuição de que "0,25 era um freio calibrado para o alvo
cru e agora subponderaria o valor" está errada na direção. A segunda, que é a que importa:
mesmo em 0,25 a política recebe cerca de **1/48** do gradiente do tronco (1 : 28,7 : 18,9).

**Estado:** `valor_symlog` e `vf_coef=0,5` são o padrão do AlphaZero (braços `sem_symlog` e
`vf_1` no `93`). No MuZero o `symlog` entrou e o `coef_valor` **ficou em 0,25**, que é o
melhor dos três medidos. Escolher um valor menor exigiria dado de resultado, e o MuZero
ainda não rodou — fica como previsão pré-registrada: se a política pura dele empacar
enquanto a busca vai bem, o primeiro suspeito é este desbalanço, e o botão é
`coef_valor`/`coef_recompensa`, não o algoritmo.

### 2.27 ? A busca do AlphaZero degenera assim que o valor aprendido fica positivo
`mcts.py:_selecionar` · `nets/registry.py:127` · `agents/alphazero.py:76`

`q = (filho.recompensa + gamma * filho.valor) if filho.visitas else 0.0`. O `0.0` é a
convenção do AlphaZero e está certa onde o valor é uma `tanh` em `[-1, 1]` centrada em
zero. Aqui a cabeça de valor é `Dense(1)` **linear**, a recompensa é `+1` por maçã e o
agente come a cada ~12 passos: com `γ = 0,997` o ponto fixo do valor é `1/(1 − γ¹²) ≈ 28`.
O bônus de exploração vale no máximo `c_puct · P · √N ≈ 2,8` com 32 simulações — o filho
virgem nunca é escolhido, a busca colapsa no primeiro filho que tocou, e como esse é a
primeira ação da máscara (`np.nonzero` crescente = virar à esquerda), o agente gira até
morrer de fome.

Medido em `tools/diag_busca.py`: a mesma heurística de folha, somada de uma constante que
não muda o ranking de estado nenhum, leva o score de 21,70 (100% colisão) a **0,00** (100%
fome). Com `q_normalizado=True` (min-max do MuZero, Apêndice B) volta a 19,71; com
`fpu="pai"`, a 22,12. Tabelas completas em `docs/BUSCA_DEGENERADA.md`.

**Tamanho do efeito, corrigido pela execução de 5 M.** A primeira versão desta seção
estimava `V ≈ 28` supondo uma maçã a cada 12 passos; o `valor_raiz` medido vai de 0,26 a
3,50 (o agente come a cada ~40 passos). Com `V ≈ 3,5` a busca não colapsa — ela fica
**incapaz de discordar da rede**: o bônus `c_puct·P·√N` cobre a diferença na raiz quando
`P = 0,7` (6,03 contra 3,5) e não cobre quando `P = 0,15` (1,29), nem em nenhum nó interno.
Uma busca que só confirma o prior não é operador de melhoria de política. O resultado da
execução — política pura 10,62, **86,9% de fim por fome**, `perda_pi` 0,016 — é consistente
com isso.

Por que passou: a heurística com que a busca foi medida — no docstring do `mcts.py` e em
`test_search_beats_random_with_an_informative_value` — é **negativa**, e nessa escala o
`0` é otimista. `tests/test_search.py` ganhou
`test_search_collapses_when_the_value_is_positive_and_q_is_unnormalized` (a
caracterização) e `test_search_is_invariant_to_a_constant_shift_in_the_value` (o conserto,
parametrizado nos dois).

**Estado:** os dois são o padrão desde que a medição os validou, no AlphaZero e no
MuZero — é o mesmo `_selecionar`. O braço `sem_conserto_da_busca` do `93_alphazero_ablacoes`
mede quanto valeram, contra `06` na mesma semente.

### 2.26 ✔ A conferência de paridade do TFLite comparava eixos diferentes — **corrigido**
`export.py:conferir_paridade`

A terceira vez que a mesma suposição sobre a forma da saída cobra o preço, e a mais cara. O
`.tflite` do Rainbow foi convertido e escrito em disco; foi a **validação** que quebrou:

```
ValueError: operands could not be broadcast together with shapes (200,) (200,121)
```

19.288 s de execução, na penúltima célula do notebook. O §2.14 foi o `Lambda` do dueling, o
§2.17 foi a política do checkpoint, este é o exportador — as três são a mesma frase, "a
saída da rede é `(lote, ações)`", escrita em três lugares diferentes.

O mecanismo: com `n_atoms > 0` a rede devolve `(lote, ações, átomos)`. A função colapsava
os átomos **só no lado Keras** e comparava o `argmax(1)` dos dois lados — `(200, 3)` contra
`(200, 3, 121)`. E a busca pela saída de política (`c.shape[-1] == N_ACTIONS`) nunca acha o
tensor do C51, cuja última dimensão são os 121 átomos: caía no `cand[0]` sem dizer nada.

O defeito não é do Rainbow. O eixo das ações muda de lugar em três dos cinco formatos que
`nets/registry.py` produz:

| construtor | saída de política | quem usa |
|---|---|---|
| `build_actor_critic` | `(lote, ações)` | PPO, A2C, ACKTR, ACEKTR, AlphaZero, MuZero |
| `build_q_network` sem C51 | `(lote, ações)` | DQN |
| `build_q_network` com `n_atoms > 0` | `(lote, ações, átomos)` | Rainbow |
| `build_actor_critic_populacao` | `(lote, políticas, ações)` | LBC |
| `build_policy_q` | `(lote, ações)` **duas vezes** | ACER |

E onde ele não quebrava, mentia. Medido, com conversão de verdade e pesos aleatórios:

| modelo | `acoes_iguais` antes | depois (fp32) |
|---|---|---|
| C51, `n_atoms=121` | `ValueError` — leva o notebook junto | 1,000 |
| LBC, 3 políticas | **0,315** | 1,000 |
| ACER | **0,210** | 1,000 |

0,315 e 0,210 são o acaso — 1/3, com três ações. No LBC a busca por `N_ACTIONS` colunas
casava o **crítico** `(lote, 3)`, porque a população padrão tem 3 políticas e o jogo tem 3
ações; no ACER as duas saídas têm a mesma forma, e qual delas o `Interpreter` lista primeiro
decidia se a comparação era política × política ou política × crítico — a ordem das saídas
do interpretador não é a do `keras.Model`. Esses dois números iam para o relatório de
exportação como se fossem medição. É o pior dos três desfechos, porque não levanta nada.

Correções:

* `_escores_por_acao` — **uma** redução, aplicada nos dois lados, que cobre os três
  formatos. O caso ambíguo `(lote, 3, 3)` do LBC é desempatado a favor da população: um C51
  de três átomos não existe, o C51 existe para ter resolução;
* `_q_de_logits_c51` — a esperança do índice do átomo sob a softmax, no lugar da média dos
  logits. Como o suporte é afim e crescente (`z_i = v_min + i·Δz`), vale
  `argmax_a Σ p(a,i)·z_i = argmax_a Σ p(a,i)·i`: a **ação escolhida** não depende de
  `v_min`/`v_max`, que moram no agente e não chegam ao exportador. A média dos logits, que
  estava ali, ignora a softmax e troca a ação escolhida — e o teste mostra que troca;
* `_indice_da_politica` — casa a saída do `.tflite` pela **forma** do tensor do Keras e, no
  empate, pelo valor. Usar a semelhança para escolher não enfraquece a afirmação: se
  nenhuma candidata se parecer com a referência, todas reprovam igual;
* uma falha da conferência não derruba mais a exportação. Ela vira `{"erro": ...}` no
  relatório — que é impresso, então continua visível — em vez de matar a célula e levar
  junto o `export/best`, o `int8` e o `.zip` da execução. A validação roda **depois** de os
  arquivos estarem em disco; deixá-la apagar o resto foi exatamente o que aconteceu aqui.

`tests/test_export.py`, 13 testes, três deles convertendo TFLite de verdade nos três
formatos. Com `int8` a paridade fica entre 0,985 e 0,995 sobre pesos aleatórios — isso é a
quantização, não o defeito.

O que **não** mudou: DreamerV3 e SOAP continuam sem afirmação de paridade, pelo motivo do
`COMPARABILITY.md` — a política tem memória e um `.tflite` sem estado não a reproduz. E
nenhuma curva da arena se move: a conferência mede o arquivo exportado, não o treino.

### 2.16 ? A exploração do Rainbow neste ambiente — **hipótese aberta**

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

### 3.6 ✔ O GIF era gravado com o estado interno congelado — **corrigido**
`env/render.py` — `quadros_do_episodio` chamava `politica(obs, mask)` e nunca `apos_passo`.

`snakeai/eval.py` respeita esse contrato desde o DreamerV3: uma política com memória precisa
saber **qual ação de fato saiu** — que pode não ser o argmax, se o filtro de segurança agiu —
e onde o episódio terminou, para zerar o estado interno ali. O renderizador não chamava, e o
resultado era um GIF gravado com o latente parado no valor inicial: **o agente do vídeo não
era o agente da curva**.

Custo real: o GIF é o único artefato que responde *como* o agente joga, e é justamente para
um agente com memória que essa pergunta é mais interessante. Todos os GIFs de DreamerV3 já
gerados mostram uma política que nunca existiu. Achado ao implementar o SOAP, que tem o mesmo
contrato; corrigido em uma linha, com teste
(`test_soap.py::test_the_gif_advances_a_policy_with_memory`).

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

## 5. Os três algoritmos acrescentados depois desta revisão

LBC, SOAP e ACEKTR entraram depois que esta lista foi escrita. Eles **não** foram revisados
pelo mesmo processo (cinco revisões independentes por área); o que segue é o registro de como
cada um se posiciona em relação aos achados de cima, e o que fica em aberto neles.

### 5.1 O que eles herdam de graça

Os três nascem sobre o `AgentBase` corrigido, e por isso não repetem os erros da §1:

| achado | como os três se comportam |
|---|---|
| §1.1 truncamento por fome | tratado nos três. No LBC e no SOAP o bootstrap é **por política / por opção**, porque o crítico é vetorial — um bootstrap único ensinaria a todas as cabeças o valor terminal de uma delas |
| §1.4 `avaliar_melhor` medindo o modelo final | o SOAP e o LBC sobrescrevem `politica_do_modelo`, então o checkpoint `best` é medido de verdade. O ACEKTR herda o caminho do ACKTR, que já estava certo |
| §1.7 grade de avaliação | herdada do `proximo_multiplo` |
| §3.2 laço de coleta triplicado | o LBC e o SOAP têm laço próprio (a coleta **é** diferente: crença, mistura de comportamento), mas usam `registra_fim` e `bootstrap_truncados` do andaime. O ACEKTR herda o `collect` do A2C sem tocar |

### 5.2 ○ O que fica em aberto neles

* **LBC — o estado do bandit não sobrevive a `retomar()`.** `salvar()` grava `self.model` e
  o dicionário de estado fixo; a janela do `BanditUCB` e o `ψ` de cada ambiente ficam de
  fora. Retomar um treino recomeça a seleção do uniforme. É o mesmo padrão do `_fator_kl` do
  ACKTR, que também não é salvo — e a consequência é da mesma ordem: a janela reenche em
  algumas dezenas de episódios. Vale registrar porque a curva depois de um `retomar()` tem um
  degrau que não é do algoritmo.
* **SOAP — `ζ` também não sobrevive a `retomar()`.** Menos grave: a crença é zerada a cada
  episódio de qualquer forma, então o custo é um episódio por ambiente.
* **SOAP — o sinal de opção morre no fim de cada rollout.** `U_{T} = 0` é a mesma truncatura
  que o GAE já tem, mas ela cai sobre o eixo que é a razão de existir do algoritmo. Com
  `rollout = 32`, um passo em 32 não recebe gradiente de troca de opção. ? Medir se subir o
  rollout muda a persistência das opções.
* **ACEKTR — `inv_every` está no valor do ACKTR, e isso handicapa o EK-FAC.** Deliberado, para
  que a comparação isole uma variável; ver `docs/EKFAC.md` §3.2. Não é um defeito, é uma
  escolha que precisa ser lida junto com a curva.
* **ACEKTR — "autovalores exatos" numa convolução é exato só sob a hipótese de homogeneidade
  espacial** que o KFC já faz. O EK-FAC corrige os autovalores *dentro* da hipótese, não a
  hipótese. Ver `docs/EKFAC.md` §6.
* **Os três dependem de `variancia_explicada` importado do `ppo.py`.** É a função certa e uma
  definição só, mas amarra três agentes ao módulo do PPO — e faz o notebook de cada um
  embarcar o PPO inteiro. ○ Candidata a subir para `snakeai/eval.py` numa próxima limpeza; o
  custo hoje é ~400 linhas mortas por notebook.

### 5.3 O que eles acrescentam à pergunta de fundo

A pergunta desta revisão era **por que o melhor agente para em ~62 de um teto de 97**, e a
resposta mais provável foi o orçamento de gradiente (§2.1), medida e confirmada em
`docs/ORCAMENTO_DE_GRADIENTE.md`. Os três novos atacam três suspeitos diferentes do que
sobrou:

| algoritmo | o que ele propõe como causa |
|---|---|
| LBC | a exploração é **agendada** por uma reta que nunca olhou para o resultado |
| SOAP | a observação do contrato **não é markoviana**, e o sexto canal foi a resposta errada para isso |
| ACEKTR | a curvatura aproximada do K-FAC **subestima** a Fisher — e é isso que faz a região de confiança do ACKTR entregar KL 7× maior que a pedida |

Nenhum dos três está medido no orçamento oficial. As previsões estão escritas nos documentos
respectivos, **antes** da medição, de propósito.

---

## Método

Cinco revisões independentes, por área (on-policy · off-policy e memória · busca e modelo ·
DreamerV3 · infraestrutura), cada uma com instrução de citar arquivo:linha e de marcar o que
não conseguisse verificar. Os achados marcados **✔** foram reconferidos linha a linha
depois. Nenhum arquivo foi modificado.
