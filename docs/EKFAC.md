# EK-FAC — a base do K-FAC, os autovalores medidos

O K-FAC faz duas coisas ao mesmo tempo, e só uma delas se justifica.

Da fatoração

```
A ⊗ G  =  (U_A ⊗ U_G) (S_A ⊗ S_G) (U_A ⊗ U_G)ᵀ
```

saem **uma base** — os autovetores `U_A ⊗ U_G`, chamados de KFE, *Kronecker-factored
eigenbasis* — e **uma escala para cada eixo** dessa base, `λ_A(j)·λ_G(i)`. A base é uma
aproximação defensável dos autovetores da Fisher verdadeira. As escalas são obrigadas a ter
forma de produto, e essa restrição não vem de lugar nenhum: ela saiu junto na conta e
ninguém a pediu.

O EK-FAC (George et al., 2018) fica com a base e joga fora as escalas, medindo no lugar
delas o segundo momento verdadeiro do gradiente projetado:

```
s*_{ji} = E_n[ ((U_Aᵀ ∇W_n U_G)_{ji})² ]
```

**Teorema 2 do paper:** `s*` é a melhor escala diagonal possível naquela base, em norma de
Frobenius. **Teorema 3:** portanto o EK-FAC nunca é pior que o K-FAC. Não é uma heurística
com um parâmetro a mais — é o mínimo de um problema de mínimos quadrados do qual o palpite
do K-FAC é um ponto qualquer.

Este documento registra **o que foi implementado, por que os defaults são o que são, e o
que a linha na arena pode e não pode responder**.

---

## 1. Por que sai barato

O gradiente **por amostra** de uma camada é um produto externo: `∇W_n = a_n g_nᵀ`. E
projetar um produto externo é projetar cada lado:

```
U_Aᵀ (a_n g_nᵀ) U_G  =  (U_Aᵀ a_n)(U_Gᵀ g_n)ᵀ
```

Então o quadrado da entrada `(j,i)` é `(U_Aᵀa_n)_j² · (U_Gᵀg_n)_i²`, e a média sobre o lote
inteiro é **um produto de matrizes** entre as duas projeções ao quadrado. Nada de
materializar `N` gradientes por amostra, nada de laço em Python — a implementação de
referência em PyTorch precisa de um laço sobre o lote no modo exato, e esta não.

O custo em relação ao K-FAC, medido em `kfac_ms`:

| onde | K-FAC | EK-FAC |
|---|---|---|
| a cada atualização | dois `cholesky_solve` | duas projeções `O(N·T·d²)` + quatro produtos de matriz |
| a cada `inv_every` | duas `cholesky` | duas `eigh` |
| memória | `A`, `G`, dois Cholesky | `U_A`, `U_G`, `λ_A`, `λ_G`, `s*` |

Numa `resnet_tiny` com lote de treino, isso deu **~2,1× o tempo do K-FAC** por
atualização. Como a atualização acontece uma vez por rollout, o custo se dilui — mas ele
não é desprezível, e `kfac_ms` está no registro justamente para que ninguém precise
supor.

---

## 2. O amortecimento, e o controle bit a bit

Esta é a parte que decide se a comparação com o ACKTR mede alguma coisa.

O K-FAC usa Tikhonov fatorado: `(A + √λ·π·I) ⊗ (G + √λ/π·I)`. Na base, isso dá a cada eixo
a escala `(λ_A + √λ·π)(λ_G + √λ/π)`, que expandida é

```
λ_A·λ_G  +  λ_A·√λ/π  +  λ_G·√λ·π  +  λ
```

O apêndice C do paper prescreve reproduzir exatamente essa estrutura em torno de `s*`:

```
denominador_{ji} = s*_{ji} + λ_A(j)·√λ/π + λ_G(i)·√λ·π + λ
```

Isso tem uma consequência que vale como teste, e é o teste mais importante da suíte: com
`s*` inicializado em `λ_A·λ_G` — o palpite do K-FAC, que é o que o EK-FAC assume antes de
medir qualquer coisa —, o denominador fica **idêntico** ao do K-FAC amortecido e as duas
direções coincidem até o arredondamento de float32.

`ACEKTRConfig(ema_escalas=1.0)` desliga a medição e expõe esse controle no nível do agente.
Se `test_without_measuring_ekfac_is_bit_for_bit_kfac` falhar, a diferença entre as curvas
passa a incluir uma convenção trocada — uma transposta, um `π` do lado errado, um fator `N`
— e deixa de ser atribuível à correção de autovalores. Nada quebraria; a curva só ficaria
diferente pelo motivo errado.

---

## 3. Os defaults, e por que eles não são os do paper

### 3.1 `ema_escalas = 0.5` (o `kfac_ema` é 0,95)

Parece inconsistente e não é. `A` e `G` são acumulados **através** das reconstruções da
base: nunca são zerados, e o que a média móvel absorve neles é só ruído de lote. `s*` não é
assim — ele descreve os eixos de **uma** base específica e é reiniciado no palpite do K-FAC
toda vez que a base muda.

Com `inv_every = 10`, uma média móvel de 0,95 gastaria a janela inteira saindo do palpite, e
o EK-FAC nunca chegaria a usar o que mediu — seria um K-FAC caro. Meia-vida de uma
atualização deixa as medições dominarem em três ou quatro passos, com folga dentro da
janela.

### 3.2 `inv_every = 10` — a amortização do paper **não** vale neste regime

O paper propõe amortizar: reconstruir a base raramente (50 a 500 passos) e recalcular as
escalas a cada passo. É de lá que vem metade do argumento — o EK-FAC não só aproxima
melhor, como fica mais barato que o K-FAC, porque a `eigh` cara é diluída em muitas
atualizações baratas.

Este documento chegou a dizer que `inv_every = 10` era um handicap deliberado e que o
regime do paper estava "a uma configuração de distância". A configuração foi rodada em
02/09. **Ela é pior, e a medição diz exatamente por quê.**

`ekfac_desvio` é um dente de serra — cai a zero em cada reconstrução da base e cresce até a
próxima. Comparando as duas execuções pelo número da atualização:

| atualização | 1 | 15 | 29 | 43 | **51** (base) | 57 |
|---|---|---|---|---|---|---|
| `inv_every = 50` | 0,06 | 35,3 | 59,7 | **69,6** | — | 0,29 |
| `inv_every = 10` | 0,36 | 0,37 | 0,37 | 0,23 | — | 0,17 |

Duas ordens de grandeza — e elas **escalam com o comprimento da janela**. Esse é o ponto:
se o número medisse violação de Kronecker, que é o que o EK-FAC existe para corrigir, ele
não dependeria de quando a base foi construída. Ele estava medindo **base velha**. (A §4
abaixo dizia que as duas coisas "não se separam neste número"; separam — basta variar
`inv_every` e ver o que se move.)

E base velha no EK-FAC é pior que base velha no K-FAC, o que não é simétrico:

* o **K-FAC** com `A`, `G` de 10 passos atrás ainda é `A⁻¹∇G⁻¹` — um pré-condicionador PSD
  coerente, que descreve uma curvatura antiga;
* o **EK-FAC** mistura `s*` medido **agora** com eixos de 50 passos atrás. Nesses eixos
  `s*` pode ser minúsculo onde a curvatura real é grande, e o denominador do apêndice C
  divide por quase nada numa direção de curvatura alta. Não é uma aproximação pior de
  `F⁻¹`; é outra coisa.

O `kl_fator` da calibração mediu isso, e é o mesmo número da §5:

| execução | `kl_fator` mediano | p90 |
|---|---|---|
| ACKTR, K-FAC, base a cada 10 | 18,7 | 22,5 |
| ACEKTR, base a cada 10 (T=5) | 20,0 | 29,3 |
| **ACEKTR, base a cada 50** | **46,2** | **58,1** |

A execução fechou em 74,47 de score com **0,4% de tabuleiros cheios** — média maior que a
da anterior (71,07) e taxa de vitória 44× menor (17,6%).

A premissa da amortização é que o modelo muda pouco entre reconstruções. Vale em
aprendizado supervisionado, com milhares de passos de gradiente. Aqui o treino inteiro tem
**610 atualizações** e cada uma anda `kl_max` de KL por construção: 50 atualizações são 8%
da execução e uma política inteiramente diferente. `inv_every` voltou a 10.

### 3.3 `escalas_acumuladas = True` — o conserto certo da queixa original

A queixa que levou ao `inv_every = 50` era real: com uma janela de 10 e `ema_escalas = 0,5`,
`s*` mal saía do palpite do K-FAC antes de a base ser reconstruída, e o que o EK-FAC usava
no meio do caminho era uma medição de ~2 lotes. Um autovalor **subestimado por ruído** vai
para o denominador e amplifica exatamente a direção que o lote não soube estimar.

Mas a resposta é trocar o **estimador**, não a janela. Dentro de 10 atualizações a rede
quase não muda, então não há deriva a esquecer — e é justamente contra deriva que serve uma
média exponencial. A média **acumulada**, com o palpite do K-FAC entrando como uma
pseudo-observação,

```
s*_k  =  (λ_A⊗λ_G  +  Σ_{i≤k} s_i) / (1 + k)
```

dá o prior a 100% em `k = 0` (o controle bit a bit continua valendo), 50% em `k = 1`, 10%
em `k = 9`, e variância caindo como `1/k` em vez de estacionar em ~2 lotes. Mesmo frescor
do K-FAC, autovalores de fato medidos.
`test_the_accumulated_estimator_is_less_noisy_than_the_exponential_one` trava a
comparação — com a base congelada, senão os dois `m2` descreveriam eixos diferentes.

`ema_escalas` continua existindo como o caminho exponencial (`escalas_acumuladas=False`) e,
em `1.0`, como o controle que desliga a medição e faz o EK-FAC virar exatamente o K-FAC.

**O nome.** Enquanto a execução com este estimador era uma só, a variante levava `+s_acum`
mesmo sendo o default — as duas execuções superadas ocupavam `resnet_small` e `+base50`, e
alguma marca tinha que separar as três na identidade `(algo, variant, seed)`. Com as três
sementes de 03/09 gravadas, a marca inverteu para a regra que vale no resto do repositório:
**o padrão não ganha marca, quem desvia é quem aparece**. Hoje o default é
`acektr/resnet_small`, a média móvel da implementação de referência é `+s_ema`, e as duas
superadas viraram `resnet_small+kl_cal_v1+s_ema_T5` (01/09) e `resnet_small+base50+s_ema`
(02/09) — cada nome derivado do config que aquela execução de fato rodou.

### 3.4 `kl_cal_debias = True` — a região de confiança não pode levar 8% do treino para acordar

`_fator_kl` é uma média móvel com `kl_cal_ema = 0,98`: constante de tempo de ~50
atualizações. O orçamento inteiro tem **610**. Partindo de 1,0, a calibração gasta ~8% do
treino subindo até o fator verdadeiro — que as execuções longas mediram entre 15 e 25 — e
nesse intervalo o alvo efetivo é até 20× maior que o pedido, exatamente quando a política
ainda é aleatória e o estrago é permanente.

Medido em 96 ambientes, 20 atualizações, mesma semente:

| | `kl_fator` na it. 10 | entropia na it. 10 | entropia na it. 15 |
|---|---|---|---|
| EMA crua, prior 1,0 | 1,10 | 0,29 | 0,25 |
| debiasada, prior 15 | assenta em ~5 na it. 5 | 0,90 (it. 5) | — |

A EMA crua ainda estava em 1,10 na décima atualização — ou seja, **não tinha começado a
corrigir** — enquanto a entropia já havia caído de 1,06 para 0,29. A versão debiasada
mantém `s` e o peso `w` acumulado e usa `s/w`, como o `1 − βᵗ` do Adam.

Na execução de 02/09 isto funcionou como projetado, e é o único dos três ajustes daquela
rodada que sobrevive: a entropia em 1 M de passos foi **0,196**, a mais alta de todas as
execuções desta família (ACKTR: 0,066 · 0,143 · 0,084; ACEKTR anterior: 0,085).

Fica **desligado no ACKTR** (`kl_cal_debias = False`), porque as três execuções gravadas de
`acktr/resnet_small` rodaram sem ele e mudar o default faria o `08_acktr` deixar de
reproduzi-las.

## 4. `ekfac_desvio`: o número que distingue "não ajudou" de "não fez nada"

O risco específico deste algoritmo é que um EK-FAC quebrado é indistinguível de um K-FAC: as
duas curvas coincidem, e a leitura natural — "a correção não valeu nada neste problema" — é
a leitura errada. `ekfac_desvio` existe para separar os dois casos:

```
‖s* − λ_A⊗λ_G‖_F / ‖λ_A⊗λ_G‖_F,  média sobre as camadas
```

É **o tamanho da correção que está sendo aplicada**. Ele mede duas coisas somadas, e vale
saber quais: quanto a Fisher deste problema deixa de ser um produto de Kronecker naquela
base, e quanto a base envelheceu desde que foi construída. As duas são o que o EK-FAC existe
para absorver, mas elas não se separam neste número.

O formato é um **dente de serra**: cai a zero em cada reconstrução da base e cresce até a
próxima. Ler uma atualização isolada não diz nada; o que interessa é o pico antes de cada
reinício.

| leitura | conclusão |
|---|---|
| picos grandes, curva igual à do ACKTR | a Fisher **não** é Kronecker aqui, o EK-FAC corrigiu bastante, e mesmo assim não mudou o score — resultado legítimo e informativo |
| grudado em zero | o EK-FAC não está corrigindo nada. Ou o problema é Kronecker de verdade, ou há um bug — e o bug é a explicação mais provável das duas |
| picos grandes e curva melhor | o caso que o paper prevê |

Nas primeiras iterações de uma `resnet_tiny` os picos ficaram na casa de dezenas, o que diz
que a hipótese de Kronecker é **bem** violada neste domínio. É a primeira medição desse tipo
sobre o `snake-arena` — a §2.7 da revisão especulava sobre a qualidade de `F̃` sem nenhum
número.

---

## 5. A previsão falsificável — ainda em aberto, e por quê

O docstring do `ACKTR` registra uma medição incômoda de uma execução de 5 M passos: a KL
**entregue** ficou sistematicamente acima da pedida — 11,8× no primeiro quinto, caindo para
4,4× no último. O diagnóstico escrito lá foi:

> `Δᵀ∇ = ΔᵀF̃Δ`, com `F̃` a Fisher *aproximada*. A KL medida é a da política de verdade.
> Onde `F̃` subestima a curvatura real, `Δ` fica grande demais naquelas direções e a KL
> prevista sai baixa.

Se esse diagnóstico estiver certo, o EK-FAC tem que **encolher o fator**: ele aproxima `F`
melhor, por teorema, na mesma base. E o número já é registrado a cada atualização —
`kl_fator`, da calibração da região de confiança.

| resultado | o que se conclui |
|---|---|
| `kl_fator` do ACEKTR mais perto de 1 | o diagnóstico se sustenta, e a correção de autovalores era a peça que faltava |
| `kl_fator` igual ao do ACKTR | o desvio vem de outro lugar — a diagonalidade por blocos, a homogeneidade espacial da convolução, ou a própria aproximação quadrática da KL — e a §região de confiança precisa ser reescrita |
| `kl_fator` **maior** que o do ACKTR | não estava previsto quando esta tabela foi escrita, e é o que a §5.2 mediu: 29,62 contra 19,11. Aproximar melhor a matriz piorou o número, o que derruba a primeira linha e agrava a segunda |

### 5.1 A primeira tentativa não conta

Em 01/09 as duas execuções existiam e a leitura parecia pronta: mediana de `kl_fator`
**18,71** no ACKTR contra **19,98** no ACEKTR — igual, não mais perto de 1. Segunda linha da
tabela, diagnóstico derrubado.

Só que o par não estava pareado. O `A2CConfig.rollout` foi de 16 para 5 no commit `7cdfe2c`
(21/08), um dia **depois** das três execuções gravadas do ACKTR, e o `ACKTRConfig` herdava
esse campo em silêncio. A execução do ACEKTR rodou com `T = 5` e as três do ACKTR com
`T = 16` — dois orçamentos de crédito diferentes, e portanto duas distribuições de `Δᵀ∇`
diferentes. A §7 pede "mesma semente, resto congelado"; o resto não estava congelado.

A mesma confusão contamina a leitura de score. Interpolando `train_score_mean` na grade:

| execução | 1,0 M | 1,5 M | 2,0 M | 3,0 M | 5,0 M |
|---|---|---|---|---|---|
| faixa das 3 sementes, `T = 16` | 26–29 | 31–37 | 40–64 | 67–72 | 73–81 |
| ACEKTR, `T = 5` | 29,3 | 36,8 | 44,1 | 55,5 | 63,5 |

Até 1,5 M o ACEKTR está no **topo** da faixa. A separação começa em 2 M, logo depois de
`shaping_frac` zerar o shaping em 1,25 M — quando a recompensa deixa de ser densa e o
crédito passa a depender da janela do GAE. Com `γλ = 0,945`, `0,945⁵ = 76%` do peso fica no
bootstrap contra `0,945¹⁶ = 40%`.

E não foi falta de passo, que era a suspeita óbvia: somando `√KL` sobre as atualizações, o
ACEKTR acumulou **202** contra 57–73 das três sementes do ACKTR. Ele andou 3,6× mais e
chegou 20 pontos abaixo — o que também descarta subir `kl_max` como conserto, e é o único
achado desta execução que sobrevive.

`ACKTRConfig` voltou a declarar `rollout = 16`.

### 5.2 O par limpo, e a resposta

Ele existe desde 03/09. O controle **não** é `acktr/resnet_small`: aquelas três sementes
rodaram antes de `kl_cal_debias` existir, e o ACEKTR parte de prior 15 com a calibração
debiasada. O controle certo é `acktr/resnet_small+kl_cal_debias_definitiva`, que difere do
default do ACEKTR em exatamente **um** campo — o pré-condicionador. Mesmo `lr_start = 0,5`,
mesmo `minibatches = 1`, mesmo `rollout = 16`, mesmo `inv_every = 10`, mesmo
`kl_max = 0,015`, mesma calibração, mesmo prior.

| semente | ACKTR score | ACEKTR score | ACKTR `kl_fator` | ACEKTR `kl_fator` |
|---|---:|---:|---:|---:|
| 0 | 69,37 | 74,30 | 18,01 | 31,27 |
| 1 | 79,04 | 80,02 | 20,20 | 21,47 |
| 2 | — | 82,01 | — | 29,62 |
| **mediana** | **74,20** | **80,02** | **19,11** | **29,62** |

**A previsão da §5 falhou, e falhou na direção contrária.** O que se esperava era o EK-FAC
*encolher* o fator; ele o **dobrou**, e subiu nas duas sementes compartilhadas. Nenhuma das
duas linhas da tabela de resultados descreve isto: não é "mais perto de 1" nem "igual". A
leitura que sobra é a segunda, agravada — o desvio sistemático da região de confiança **não**
vem de `F̃` subestimar a curvatura na média, porque aproximar melhor a matriz piorou o
número. Vem de algum dos três suspeitos que a §6 já listava, e o primeiro deles é o desta
casa: `s*` é exato *dentro* da hipótese de homogeneidade espacial da convolução, não sobre
ela.

Há uma explicação mecânica compatível com o sinal, e ela é falsificável: `Δᵀ∇ = ΔᵀF̃Δ` usa a
Fisher aproximada para *prever* a KL, mas o passo também é *tomado* nela. Corrigir os
autovalores encolhe `F̃` nas direções em que o K-FAC os inflava, o passo cresce justamente
ali, e a KL entregue cresce junto — a melhor aproximação da matriz produz um passo maior, não
uma previsão melhor. Quem separa as duas coisas é medir a KL prevista contra a entregue por
faixa de autovalor corrigido, que é medição nova e ainda não foi feita.

**No score, nada se separa.** A mediana vai de 74,20 para 80,02 e o ACEKTR ganha nas duas
sementes compartilhadas, mas a dispersão entre sementes do próprio ACKTR neste ambiente é de
19 pontos — maior que a diferença inteira. É exatamente o que a §8 promete que **não** se
pode prometer, e continua valendo: duas e três sementes não separam um efeito deste tamanho.

## 6. O que "exato" quer dizer numa convolução

Numa `Dense`, `s*` é o segundo momento exato — sem aproximação nenhuma.

Numa `Conv2D`, não. O gradiente por amostra é a **soma sobre as posições espaciais** de
produtos externos, e o quadrado de uma soma não se decompõe. Aqui, como no KFC (Grosse &
Martens, 2016) que o K-FAC deste repositório já usa, cada posição é tratada como uma amostra
independente, e `s*` é o segundo momento exato **sob essa hipótese**.

Ou seja: o EK-FAC corrige os autovalores *dentro* da hipótese de homogeneidade espacial, não
a hipótese. Está registrado porque "autovalores exatos numa camada convolucional" é uma
frase que promete mais do que entrega — e porque, se a previsão da §5 falhar, esta é a
primeira suspeita.

A convenção de escala que faz isso fechar: a soma sobre as `T` posições entra dividindo por
`N`, e não por `N·T`. É o que põe `s*` na mesma escala de `λ_A·λ_G` e faz o amortecimento do
apêndice C reproduzir o Tikhonov fatorado. Trocar os dois deixaria `s*` menor por um fator
`T` — 100 na primeira convolução de um tabuleiro 10×10 — e o EK-FAC viraria um K-FAC com
amortecimento gigante, que **treina**, só que pior.
`test_the_conv_scales_keep_the_same_convention` trava isso.

---

## 7. O que comparar com o quê

| par | o que a diferença mede |
|---|---|
| `12_acektr` com `escalas_acumuladas=False, ema_escalas=0.5, kl_cal_debias=False, kl_fator_inicial=1.0` × `08_acktr`, mesma semente | a correção de autovalores, com todo o resto congelado — o par de uma variável só |
| `12_acektr` no default (`resnet_small`, sem marca) × `08_acktr+kl_cal_debias_definitiva` | **o par limpo, e o que a §5.2 responde.** Uma variável só: o pré-condicionador. Todo o resto — `lr`, `minibatches`, `rollout`, `inv_every`, `kl_max`, calibração e prior — é idêntico |
| `12_acektr` no default × `08_acktr` (as três sementes de agosto) | mede **duas** coisas somadas: autovalores e a partida da região de confiança, porque aquelas execuções rodaram antes de `kl_cal_debias`. É a leitura de desempenho, não a de atribuição |
| `12_acektr` × `12_acektr` com `ema_escalas=1` | o mesmo, com o controle *dentro* do algoritmo: a segunda execução é o K-FAC bit a bit |
| `12_acektr` × `12_acektr+base50+s_ema` | o eixo de amortização do paper. **Já foi rodado e a resposta é não** — ver §3.2. O par continua na tabela porque a medição vale, não porque a pergunta esteja aberta |
| `12_acektr` × `04_a2c` | a soma dos dois: vale a pena aproximar a curvatura, e vale a pena corrigir os autovalores dela |

---

## 8. O que **não** se pode prometer

Que o ACEKTR ganhe do ACKTR na arena.

Aproximar melhor a Fisher é uma afirmação sobre a matriz, não sobre o score final. E a
primeira execução longa do ACKTR mostrou que neste domínio a dispersão entre sementes é
enorme — três sementes deram 89,78 · 70,67 · 78,13, desvio 9,63, contra 1,80 do PPO. Uma
diferença plausível de pré-condicionador desaparece dentro dessa dispersão, e três sementes
não bastam para separá-la.

O que o algoritmo entrega com certeza são **duas medidas** que hoje não existem: o quanto a
Fisher deste problema deixa de ser um produto de Kronecker (`ekfac_desvio`), e se o desvio
sistemático da região de confiança do ACKTR vem daí (`kl_fator`). Se a curva também subir,
melhor; mas não é por isso que ele está aqui.

---

## Referência

Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent. *Fast
Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis*, NeurIPS 2018.
[arXiv:1806.03884](https://arxiv.org/abs/1806.03884)

Implementação de referência (PyTorch):
[Thrandis/EKFAC-pytorch](https://github.com/Thrandis/EKFAC-pytorch). Ela oferece três modos
— `ra` (média móvel do gradiente do minilote, multiplicada pelo tamanho do lote), `intra`
(exato, com laço sobre o lote) e a base congelada. Esta implementação é o modo exato **sem**
o laço, pelo argumento do produto externo da §1.

O K-FAC que serve de base está em `snakeai/kfac.py` (Martens & Grosse, 2015; Grosse &
Martens, 2016 para as convoluções), e o agente que o usa é o `ACKTR` (Wu et al., 2017).
