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

### 3.2 `inv_every = 10`, o mesmo do ACKTR — e é aqui que o EK-FAC está handicapado

O paper propõe **amortizar**: reconstruir a base raramente (50 a 500 passos) e recalcular as
escalas a cada passo. É de lá que vem metade do argumento — o EK-FAC não só aproxima melhor,
como fica mais barato que o K-FAC, porque a `eigh` cara é diluída em muitas atualizações
baratas.

Aqui o default é `10`, o mesmo do ACKTR, e a escolha é deliberada: **é o que faz
`08_acktr × 12_acektr` isolar uma variável só**. Se o EK-FAC também mudasse a frequência de
reconstrução, a diferença entre as curvas passaria a misturar "autovalores melhores" com
"base mais velha", e o repositório teria mais uma comparação de duas variáveis — que é
exatamente o que o `96_ppo_orcamento_esparso` documenta como o defeito do braço de controle
antigo.

O regime do paper está a uma configuração de distância: `ACEKTRConfig(inv_every=50)`, que
ganha a variante `+base50`. É a execução que responde "e se o EK-FAC rodar como ele foi
projetado?" — e ela **não** compete com o `08_acktr` na mesma leitura, porque mexe em dois
botões.

---

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

## 5. A previsão falsificável

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

Nos dois casos o repositório troca uma suposição por uma medida. É a comparação mais barata
deste documento: os dois números já existem, é só rodar as duas execuções na mesma semente.

---

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
| `12_acektr` × `08_acktr`, mesma semente | a correção de autovalores, com todo o resto congelado — a leitura principal |
| `12_acektr` × `12_acektr` com `ema_escalas=1` | o mesmo, com o controle *dentro* do algoritmo: a segunda execução é o K-FAC bit a bit |
| `12_acektr` × `12_acektr+base50` | o eixo de amortização do paper: base rara, escalas sempre |
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
