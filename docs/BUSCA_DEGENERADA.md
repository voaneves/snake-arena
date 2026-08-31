# A busca do AlphaZero degenera quando o valor aprendido é positivo

**Resposta curta:** o PUCT deste repositório dá `Q = 0` a um filho ainda não visitado. Essa
é a convenção do AlphaZero e ela está certa em Xadrez e Go, onde o valor é uma `tanh` em
`[-1, 1]` centrada em zero. Aqui a cabeça de valor é **linear** e a recompensa é `+1` por
maçã: o valor aprendido é positivo, e cresce conforme o agente melhora.

> **Correção, com o dado real na mão.** A primeira versão deste documento estimava o valor
> em `1/(1 − γ¹²) ≈ 28`, supondo uma maçã a cada 12 passos. A execução de 5 M passos
> (`runs/alphazero/resnet_small_sem_correcoes_sims32/history.json`) mede `valor_raiz` indo
> de **0,26 a 3,50**, terminando em 3,18: o agente come uma maçã a cada ~40 passos, não a
> cada 12. O mecanismo é o mesmo; o tamanho dele é **8× menor** do que estava escrito aqui.

Com `V ≈ 3,5`, o bônus de exploração vale `c_puct · P · √N`, ou seja `8,6 · P` na raiz com
32 simulações. A conclusão deixa de ser "a busca colapsa" e vira algo mais específico — e,
para o que se quer aqui, igualmente ruim:

| onde | `√N` | prior `P` | bônus do filho virgem | contra `Q ≈ 3,5` do irmão |
|---|---:|---:|---:|---|
| raiz, ação de que a rede gosta | 5,7 | 0,70 | 6,03 | ganha |
| raiz, ação de que a rede **não** gosta | 5,7 | 0,15 | 1,29 | **perde** |
| nó interno com 4 visitas | 2,0 | 0,15 | 0,45 | **perde** |

Não é uma busca parada: é uma busca que **só consegue confirmar o que a rede já achava**.
Uma ação de prior baixo nunca é experimentada fundo o bastante para a busca discordar da
rede — e uma busca que não discorda não é operador de melhoria de política, é um
amplificador de confiança. O efeito endurece com a profundidade, porque `√N` encolhe
conforme se desce, e endurece com o tempo, porque `V` cresce enquanto o bônus não.

É o que a execução mostra: `perda_pi` cai a **0,016** — a rede reproduz o alvo da busca
quase perfeitamente — enquanto a política pura empaca em ~11 pontos, com **86,9% dos
episódios terminando por fome** e os três GIFs do fim do treino terminando por fome (scores
0, 1 e 22). A curva sobe até 3,0 M (13,03) e depois oscila entre 9,6 e 12,5 sem tendência.

Reproduzir: `python tools/diag_busca.py`. Os números abaixo estão em `docs/diag_busca.json`.

## O experimento que isola a causa

Mesma busca, mesmas 8 simulações, mesma heurística de folha (distância de Manhattan até a
comida). A única diferença entre os dois blocos é **uma constante somada ao valor** — o
ranking de todos os estados é idêntico, e nenhuma decisão deveria mudar.

| valor da folha | `fpu` | `q_normalizado` | score | como termina |
|---|---|---|---:|---|
| negativo, ∈ [−0,9 , 0] | `zero` | não | 21,70 | 100% colisão |
| negativo | `pai` | não | 21,70 | 100% colisão |
| negativo | `zero` | sim | 24,12 | 100% colisão |
| negativo | `pai` | sim | 22,75 | 100% colisão |
| **positivo, ∈ [0,1 , 1,0]** | **`zero`** | **não** *(o padrão)* | **0,00** | **100% fome** |
| positivo | `pai` | não | 22,12 | 100% colisão |
| positivo | `zero` | sim | 19,71 | 100% colisão |
| positivo | `pai` | sim | 23,00 | 100% colisão |

Uma linha em oito faz score zero, e é a linha que o repositório roda hoje assim que a rede
aprende alguma coisa.

**Por que isso passou despercebido por tanto tempo.** O docstring do `mcts.py` afirma que
"8 simulações por jogada já dão score 24", e `tests/test_search.py` tem um teste desenhado
exatamente para pegar bug silencioso na busca
(`test_search_beats_random_with_an_informative_value`). Os dois usam a mesma heurística —
e ela é **negativa**. Nessa escala o `Q = 0` do filho virgem é *otimista*, e força a busca a
experimentar todo mundo. A medição estava certa; o que ela media era um regime que o treino
abandona. A primeira linha da tabela é aquele número, e ele continua lá.

## As duas correções, e por que são duas

Ambas entram como flags **desligadas por padrão**, para que `06_alphazero` continue sendo o
braço de controle das ablações do `93_alphazero_ablacoes`.

**`q_normalizado=True`** — a normalização min-max do MuZero (Schrittwieser et al., 2020,
Apêndice B): cada `Q` é mapeado para `[0, 1]` pela faixa observada *naquela árvore*. Isso
devolve `c_puct` à escala em que ele foi calibrado. O filho virgem continua valendo `0`,
mas `0` passa a significar "o pior Q já medido aqui" em vez de "28 abaixo de tudo" — e aí
o bônus de exploração alcança.

**`fpu="pai"`** — o *first play urgency*: o filho virgem herda o valor do próprio nó, que é
o palpite honesto quando não se mediu nada. Ataca o mesmo problema pelo outro lado e é
independente da escala.

Elas resolvem coisas diferentes e não são substitutas: a normalização conserta o `c_puct`
(que, com Q da ordem de dezenas, hoje é ruído — o PUCT vira `argmax Q` e o prior da rede
não importa), o FPU conserta o palpite inicial. Ver os braços `q_normalizado`, `fpu_pai` e
`tudo` no `93`.

## O γ é o amplificador, não a causa

O tamanho do buraco que o filho virgem precisa atravessar é o próprio valor, e o valor é
função de `γ`:

| γ | horizonte `1/(1−γ)` | valor no ponto fixo `1/(1−γ¹²)` | teto do bônus PUCT (32 sims) |
|---:|---:|---:|---:|
| 0,980 | 50 | 4,6 | 2,87 |
| 0,985 | 67 | 6,0 | 2,87 |
| 0,990 | 100 | 8,8 | 2,87 |
| **0,995** *(PPO, DQN, ACER, SOAP)* | 200 | 17,1 | 2,87 |
| **0,997** *(AlphaZero, MuZero, DreamerV3)* | 333 | **28,2** | 2,87 |

A coluna do meio é o **teto**: o ponto fixo se o agente comesse a cada 12 passos. O agente
real come a cada ~40 e chega a `V ≈ 3,5` com `γ = 0,997` — ou seja, opera bem abaixo desse
teto, e o buraco do FPU cresce junto com a competência dele. Baixar `γ` encolhe os dois
lados e *parece* consertar; não conserta, porque a comparação que importa é `V` contra
`c_puct · P · √N`, e `P` é pequeno justamente onde a busca precisaria discordar da rede. É
por isso que `gamma_995` existe como ablação de alinhamento com o resto do repositório, e
não como o conserto.

## O que `num_simulations` de fato compra

Medido com a busca não degenerada (`q_normalizado=True`), heurística de folha e **prior
uniforme**. A profundidade é a da variação principal — o ramo mais visitado, que é o que a
busca de fato usa:

| sims | prof. da árvore | prof. da PV | PV p95 | entropia das visitas |
|---:|---:|---:|---:|---:|
| 8 | 3,84 | 3,81 | 5 | 0,622 |
| 16 | 5,06 | 4,96 | 7 | 0,616 |
| 32 | 6,33 | 6,10 | 8 | 0,626 |
| 64 | 7,73 | 7,43 | 10 | 0,602 |
| 128 | 9,15 | 8,77 | 12 | 0,567 |

Duas leituras, e a segunda contraria o que se costuma dizer aqui:

1. **Dobrar as simulações compra ~1,3 plies, não o dobro de profundidade.** Se a hipótese
   for "a cobra não enxerga o beco de 12–15 passos", 64 não resolve — seria preciso ~256 a
   512. O argumento de horizonte não sustenta a subida de 32 para 64.
2. **A concentração quase não muda de 32 para 64** (entropia 0,626 → 0,602). O argumento de
   destilação — "com poucas visitas o alvo sai uniforme e não ensina nada", que é o do
   docstring do `alphazero.py` — sobrevive, mas o ganho aparece só em 128. **Caveat:** aqui
   o prior é uniforme; com uma rede treinada o prior é concentrado e as visitas concentram
   mais. Este número é um piso, não a medida do caso real. É por isso que `busca64` é um
   braço do `93` e não uma recomendação.

A coluna de score foi omitida de propósito: cada orçamento roda um número diferente de
passos (senão a tabela leva minutos) e os de 64 e 128 fecham 0 episódios. Comparar score
entre linhas aqui seria comparar amostras de tamanhos diferentes.

## O ruído de Dirichlet com 3 ações

A heurística do paper é `α ∝ 1/(ações legais)`, calibrada em ~10/n: Go usa 0,03 com ~250
ações (7,5), Xadrez 0,3 com ~35 (10,5), Shogi 0,15 com ~92 (13,8). **Para 3 ações isso dá
α ≈ 3,3.** O 0,5 de hoje já está bem abaixo disso; baixar mais vai na direção errada.

Amostrando 200 mil ruídos com 3 ações:

| α | máx. médio | P(máx > 0,8) | P(máx > 0,9) | entropia norm. |
|---:|---:|---:|---:|---:|
| 0,15 | 0,853 | 68,4% | 54,3% | 0,311 |
| 0,20 | 0,820 | 60,6% | 44,7% | 0,378 |
| 0,25 | 0,793 | 54,1% | 37,1% | 0,432 |
| 0,30 | 0,769 | 48,5% | 31,1% | 0,478 |
| **0,50** *(atual)* | 0,701 | 31,7% | **15,3%** | 0,607 |
| 1,00 | 0,611 | 12,0% | 3,0% | 0,758 |
| 2,00 | 0,535 | 2,0% | 0,1% | 0,865 |
| 3,33 *(10/n)* | 0,491 | 0,2% | 0,0% | 0,915 |

A patologia que se costuma atribuir ao α alto — "o ruído se concentra numa única ação
aleatória" — é o que o α **baixo** faz: com 0,2 mais de 90% da massa cai numa ação só em
45% dos lances, contra 15% em 0,5. O braço `dirichlet_1` sobe para 1,0, no meio do caminho
entre o valor de hoje e a heurística do paper.

## O segundo mecanismo: o tronco compartilhado é otimizado para o valor

A busca degenerada explica o *looping*. Ela não explica sozinha por que a **política pura**
fica perto do piso aleatório mesmo quando a busca vai bem — na execução de referência, 1 M
de passos com a busca em 17,8 e a rede em 2,45, contra um piso de 1,21.

A segunda causa é aritmética. O AlphaZero original treina o valor contra o resultado da
partida, em `[-1, 1]`: `perda_v` e `perda_pi` nascem na mesma ordem de grandeza e
`vf_coef = 1` é um número razoável. Aqui o alvo de valor é um retorno descontado **não
normalizado** — com `γ = 0,997` e uma maçã a cada ~37 passos (o regime medido em 1 M) ele
vale ~9 — enquanto a perda de política é uma entropia cruzada sobre 3 ações, presa perto de
`ln 3 ≈ 1,10`. As duas dividem o mesmo tronco convolucional, e o gradiente que chega lá é
dominado por uma delas.

Medido em `tools/diag_balanco_perdas.py` — razão entre as normas dos gradientes **no
tronco**, com o alvo de valor escalado para simular regimes de agente cada vez melhor:

| escala do alvo | \|z\| médio | ‖∇perda_v‖ / ‖∇perda_pi‖ | com `valor_symlog` |
|---:|---:|---:|---:|
| 1× *(agente que quase não come)* | 0,27 | 4,8× | 2,3× |
| 5× | 1,34 | 26,3× | 10,8× |
| 10× | 2,68 | 58,8× | 16,4× |
| **20×** *(o regime de 1 M de passos)* | **5,36** | **124,1×** | **23,0×** |
| 40× | 10,72 | 254,8× | 30,2× |

O problema não é só que a razão é grande: é que ela cresce **linearmente com a escala do
valor**, e a escala do valor cresce conforme o agente melhora. Quanto melhor o agente fica,
mais o tronco é puxado para prever o valor e menos sobra para a política — um laço que se
aperta sozinho. Com `symlog` a razão cresce só logaritmicamente.

O PPO deste repositório não sofre disso porque normaliza a vantagem por minilote
(`ppo.py:303`), o que torna o gradiente de política invariante à escala do valor, e usa
`vf_coef = 0,5` contra o `1,0` daqui. O AlphaZero não normaliza nada.

Daí dois braços a mais no `93`: `vf_025` (o botão que já existia, gratuito) e
`valor_symlog` (a transformação do DreamerV3, também desligada por padrão; a busca continua
recebendo o valor na escala real — `_frente` desfaz o symlog antes de devolver, porque o
backup do MCTS soma `recompensa + γ·valor` com recompensas de verdade).

**O que isto não prova.** A medição é da razão entre gradientes, não do resultado. Um tronco
puxado 124× para o valor pode ainda assim aprender a política, mais devagar — que é
compatível com o que a curva mostra. O que decide é o braço rodado contra o controle.

## O terceiro mecanismo: a temperatura destrói o alvo que a destilação deveria aprender

`temp_frac` não é o que parece. `temperatura()` lê `self.frac()`, que é
`global_step / total_steps` — **fração do treino inteiro**, não do episódio. Não existe
"metade do episódio estocástica": existe *metade do orçamento* jogada a τ = 1,0 e a outra
metade a τ = 0,25. As duas metades erram, em direções opostas.

A segunda metade é a que faz dano de verdade, porque a mesma distribuição temperada que
escolhe a ação **também é o alvo de treino** (`pi_b[t] = pi`). E τ = 0,25 eleva as
contagens de visita à quarta potência. Nas contagens medidas com 32 simulações (entropia
normalizada 0,626 — ver a tabela de profundidade acima):

| contagens de visita | alvo a τ = 1,0 | alvo a τ = 0,25 | entropia norm. |
|---|---|---|---:|
| `[24, 5, 3]` | `[0,750  0,156  0,094]` | `[0,9979  0,0019  0,0002]` | 0,662 → **0,015** |
| `[20, 8, 4]` | `[0,625  0,250  0,125]` | `[0,9735  0,0249  0,0016]` | 0,819 → 0,117 |
| `[16, 10, 6]` | `[0,500  0,312  0,188]` | `[0,8530  0,1302  0,0169]` | 0,932 → 0,428 |
| `[12, 11, 9]` | `[0,375  0,344  0,281]` | `[0,4944  0,3491  0,1564]` | 0,994 → 0,916 |

Da metade do treino em diante, o alvo de política vira **rótulo duro**: a rede é treinada
para ter confiança máxima no argmax de uma busca de 32 simulações. Isso joga fora
exatamente o que o AlphaZero destila — a distribuição de visitas *é* o produto da busca, e
a incerteza dela é informação, não ruído. E não há nada segurando: `ent_coef = 0,0`, porque
a exploração era para vir do Dirichlet.

No AlphaZero os dois papéis são separados: a temperatura é um botão de **exploração na
coleta**, e o alvo é a contagem crua. É o que `temp_alvo = 1.0` faz — o braço `alvo_cru`, o
mais barato de todos os consertos: uma linha, custo zero, e está dentro do `consertos`.

`temp_passos` é o agendamento canônico do paper (τ alto nos primeiros lances de **cada
episódio**, frio no resto) e resolve a primeira metade: hoje a cobra joga solta justamente
quando o tabuleiro aperta, porque o τ dela depende do calendário do treino e não da posição
no tabuleiro.

**Cuidado ao rodar os braços isolados:** `temp_por_lance` sozinho mantém `temp_fim = 0,25`,
então ele muda *quando* a temperatura cai, não o quanto. Já `temp_por_lance` combinado com
um `temp_fim` baixo **sem** `alvo_cru` deixaria o alvo duro a partir do lance 30 — pior que
hoje. Os dois andam juntos dentro do `consertos` por esse motivo.

## O que aconteceu depois

Os onze consertos **são o padrão** do `AlphaZeroConfig` desde que a medição os validou:
`fpu="pai"`, `q_normalizado`, `valor_symlog` com `vf_coef=0,5`, `temp_alvo=1,0`,
`temp_passos=30`, `epochs_por_iter=8`, `lr_final=5e-5`, `dirichlet_alpha=1,0`,
`desempate="aleatorio"` e `bootstrap_fim_janela`. Quem roda `06_alphazero` sem tocar em
nada roda o agente consertado.

Cada um continua desligável, e é isso que o `93_alphazero_ablacoes` mede — 17 braços que
**removem** uma coisa do padrão, não que a acrescentam. É a mesma inversão que o
`98_acktr_kl_nominal` sofreu quando a calibração da região de confiança venceu e virou o
padrão do `08`. Três braços removem um mecanismo inteiro e respondem a pergunta em três
execuções em vez de onze:

| braço | remove | seção |
|---|---|---|
| `sem_conserto_da_busca` | `fpu`, `q_normalizado` | §2.27 |
| `sem_conserto_do_tronco` | `valor_symlog`, `vf_coef` | §2.28 |
| `sem_conserto_do_alvo` | `temp_alvo`, `temp_passos` | §2.29 |
| `sem_correcoes` | tudo — é o agente anterior | as três |

A execução que motivou este documento continua na arena como
`alphazero/sims32_sem_correcoes/seed0`, renomeada para não dividir a identidade
`(algo, variant, seed)` com as novas; o `meta["renomeado_de"]` guarda o motivo e a
assinatura de código antiga, e o `caveat` diz do que ela é anterior.

**Duas armadilhas na própria coluna.** Quando a coluna com busca foi escrita, ela trouxe
uma cópia manual da contabilidade do `evaluate` — e a cópia errava exatamente onde o
`snakeai/eval.py` já tinha um comentário e um teste avisando. O score do episódio saía de
`env.score` lido **antes** do passo, o que perde um ponto em todo episódio que termina
comendo — ou seja, em **toda vitória por tabuleiro cheio**; e a `win_rate` saía de um
contador do laço, que continua somando os ambientes já fora da cota. As duas distorcem
justamente o regime em que um agente bom vive: com 84% de vitórias, quase toda a amostra.
A contabilidade agora é uma só, em `AgentBase.rodar_protocolo`, compartilhada pelos dois
agentes — a coluna com busca não pode mais divergir do protocolo oficial por cópia.

**Uma coluna que faltava.** O AlphaZero existe para buscar, e era avaliado só sem buscar.
`avaliar_com_busca` estava no agente desde sempre — protocolo oficial, 1.000 episódios,
greedy, semente 123 — e **nenhuma célula do notebook chamava**. Agora o `06` e o `93` têm a
célula, com dois orçamentos de simulação, e o resultado vai para o campo `busca` do
registro. A curva oficial continua sendo a rede pura, porque a busca gasta 33 avaliações de
rede por jogada contra 1 do PPO; a coluna separada é como se reporta isso sem trapacear no
eixo. Falta fazer o mesmo no MuZero — ver `docs/ANTES_DO_ARTIGO.md`.

## O que este documento **não** mede

* Nada aqui usou a rede treinada. A avaliação de folha é uma heurística, escolhida
  justamente porque permite deslocar o valor sem mudar o ranking. O que a rede de fato
  aprende a valer é medido pelo `valor_raiz` no registro da execução — se ele estiver na
  casa das dezenas e positivo, o regime é o da linha de score 0,00.
* Nenhum dos consertos foi rodado sob o contrato de 5 M passos. Os braços do
  `93_alphazero_ablacoes` existem para isso, contra `06_alphazero` **na mesma semente**.
* Os três mecanismos são independentes e podem ser cumulativos, mas **nenhum dos dois foi
  isolado sob o contrato de 5 M passos**. Concorrem ainda o orçamento de gradiente (o
  AlphaZero gasta ~4.900 atualizações contra as ~38.300 do PPO no mesmo orçamento de
  ambiente; ver `docs/ORCAMENTO_DE_GRADIENTE.md` e o braço `gradiente_8x`) e o alvo de
  política temperado (`temp_alvo`, braço `alvo_cru`). Quatro hipóteses, doze braços, um
  controle — é o que o `93` existe para resolver.
