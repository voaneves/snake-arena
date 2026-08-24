# LBC — trocar exploração agendada por exploração escolhida

Os outros nove algoritmos deste repositório exploram por **regra fixa**. O ε do DQN desce
numa reta, o coeficiente de entropia do PPO desce noutra, o σ da `NoisyDense` encolhe
sozinho conforme a rede fica confiante. Nenhuma dessas regras olha para o resultado: o ε de
0,3 no passo 1 M é 0,3 porque o agendamento diz, e não porque explorar tanto ali esteja
rendendo alguma coisa.

Isso é uma escolha de projeto que o repositório nunca mediu. As três curvas de ablação que
existem hoje — orçamento de gradiente, canal de fome, região de confiança do ACKTR — mexem
em como o agente *aprende*, nunca em como ele *decide o que experimentar*. O LBC (Fan et
al., ICLR 2023) é o algoritmo que faz dessa decisão um problema de otimização, e é por isso
que ele entra na arena.

Este documento registra **o que foi implementado, o que foi desviado do paper e por quê**,
para que a linha do LBC na arena possa ser lida sem que ninguém precise abrir o código.

---

## 1. O que o algoritmo faz

Três peças, na ordem em que atuam num passo de treino.

### 1.1 A população

`N = 3` políticas, cada uma com o seu fator de desconto: **0,99 · 0,995 · 0,999**. Míope,
o do contrato, paciente. Elas compartilham o tronco e diferem nas cabeças — ver §2.1.

Uma política com γ baixo prefere a comida à vista; uma com γ alto aceita rodeios para não
se prender. São comportamentos qualitativamente diferentes sobre o mesmo estado, e é essa
diferença que dá matéria-prima ao espaço de comportamento.

### 1.2 O mapeamento híbrido

O comportamento — a distribuição que **de fato** escolhe as ações — é uma mistura de
Boltzmann sobre a população inteira:

```
μ_ψ(a|s) = Σ_i ω_i · softmax(τ_i · logits_i(s))        ψ = (τ_1..τ_N, ω_1..ω_N)
```

É aqui que o LBC se separa do Agent57. Lá, o meta-controlador escolhia **qual política da
população usar**, e o espaço de comportamento tinha exatamente `N` elementos — para
aumentá-lo era preciso treinar mais políticas, que é caro. Aqui o espaço é contínuo e tem
dimensão `2N`: os `τ` controlam a entropia política a política (τ → 0 é uniforme, τ grande
é guloso) e os `ω` decidem quanto cada política contribui. Com `ω` one-hot e `τ = 1`
recupera-se exatamente o caso Agent57, que passa a ser um ponto dentro do espaço em vez de
ser o espaço todo.

**Detalhe de implementação que não é detalhe:** a máscara de ação é aplicada *depois* de
multiplicar os logits por `τ`. Mascarar antes multiplicaria o `MASK_NEG = −1e9` por `τ`, e
com `τ` pequeno o valor encolhe na direção do zero — uma ação letal voltaria a ter
probabilidade não desprezível. Não levanta exceção nenhuma: a cobra só passa a bater na
parede de vez em quando. `test_the_mixture_never_puts_mass_on_a_lethal_action` trava isso.

### 1.3 A seleção por bandit

Ψ é contínuo e o bandit precisa de braços, então Ψ é discretizado em **regiões**. A palavra
é importante: o braço não é um `ψ`, é um conjunto de `ψ`, e puxar o braço sorteia um `ψ` de
dentro dele. Se o braço fosse um ponto, o espaço de comportamento voltaria a ser finito —
exatamente a limitação que o LBC existe para remover.

A discretização é o produto de dois eixos:

| eixo | valores | o que varia |
|---|---|---|
| faixa de `τ` | 4 intervalos log-uniformes em `[0,25 · 4,0]` | quão determinístico é o comportamento |
| padrão de `ω` | 3 concentrados (um por política) + 1 uniforme | qual política domina a mistura |

São **16 braços**. Um produto cartesiano completo sobre os `2N` eixos daria `4³ × …`, que o
bandit não conseguiria estimar dentro de 5 M passos: com mais braços que episódios por
janela, todo braço fica sem valor estimado e o UCB vira sorteio uniforme caro.

Cada ambiente carrega um braço e um `ψ` próprios, trocados **quando o episódio daquele
ambiente acaba** — nunca no meio. É o que torna a atribuição de crédito exata: o retorno
não descontado do episódio (o score, a métrica do contrato) vai para o braço que esteve no
ar durante o episódio inteiro. Trocar por iteração pareceria mais simples e creditaria o
resultado a um braço que só apareceu no fim; o bandit aprenderia a preferir o comportamento
errado, com curvas saudáveis o tempo todo.

### 1.4 E por isso, V-trace

A mistura `μ` não é nenhuma das políticas que estão sendo treinadas. Os dados são
off-policy **por construção**, e não por acidente: quanto mais o meta-controlador explora,
mais longe `μ` fica de cada `π_i`.

`vtrace()` corrige com pesos de importância `π_i/μ` truncados — `ρ̄ = 1` para o alvo de
valor, `c̄ = 1` para a propagação temporal. Os dois fazem coisas diferentes e confundi-los é
o bug clássico: `ρ̄` decide **para que ponto fixo** o crítico converge; `c̄` decide só a
variância da propagação para trás no tempo.

Há um efeito colateral que interessa ao §2.1 da revisão. Como `μ` está **gravado**, várias
épocas sobre o mesmo rollout continuam corretas à medida que `π_i` se afasta: o orçamento de
gradiente sai do próprio estimador. No PPO, o mesmo orçamento precisa do clipping para ser
comprado. Por isso o LBC roda 4 épocas × 32 minilotes, como o PPO, sem nada parecido com
clipping no código.

---

## 2. Os desvios declarados

Três em relação ao paper, e dois no meta-controlador. Todos por causa do contrato deste
repositório, e nenhum silencioso.

### 2.1 Tronco compartilhado entre as políticas

**Paper:** cada política é um modelo independente (Assumption 1 — mesma estrutura,
parâmetros próprios).
**Aqui:** um tronco, `N` pares de cabeças (política, valor).

O motivo é o contrato. Todos os algoritmos recebem 5 M passos de **ambiente**; três ResNets
separadas triplicariam o custo por passo, e o LBC entraria na arena competindo com o mesmo
orçamento de ambiente e três vezes mais computação. É exatamente a comparação que este
repositório existe para não fazer — a mesma razão pela qual o número do AlphaZero com busca
fica numa coluna à parte.

O que se perde é diversidade de **representação**: as três políticas veem as mesmas
features. O que se mantém — e é o que constrói o espaço de comportamento não-degenerado do
§4.1 — é diversidade de **objetivo**: cada cabeça tem o seu γ, o seu crítico e o seu alvo
V-trace.

Se alguém quiser medir o custo desse desvio, o experimento é direto: três instâncias de
`build_actor_critic_populacao(n_politicas=1)` treinadas em paralelo, com o mesmo `ψ`. Fica
registrado como pergunta aberta, não como algo que já sabemos.

### 2.2 `H` reduzido ao fator de desconto

**Paper:** `h_i = (γ_i, RS_i)` — desconto **e** um método de *reward shaping* por política.
**Aqui:** `h_i = γ_i`.

O eixo de shaping não foi cortado por preguiça: o `VecSnake` aplica o shaping potencial
dentro do `step()` e devolve **uma** recompensa. Ter um shaping por política exigiria ou
mudar a assinatura do ambiente — que é a fonte única de verdade de dez algoritmos — ou
reimplementar o potencial dentro do agente, que é precisamente o erro que a §1.5 da revisão
documenta (`avaliar_com_busca` reimplementou o protocolo e reintroduziu dois bugs já
corrigidos).

O shaping continua existindo, compartilhado e com o mesmo agendamento do PPO, e usa o γ da
política avaliada. Como ele decai a zero em 25% do treino, a escolha do γ do shaping afeta
só o primeiro quarto da execução.

Consequência a registrar: o espaço de comportamento deste LBC é **menor** que o do paper em
um eixo. Pela Fig. 5 do próprio paper, reduzir o espaço degrada o desempenho — então o
número daqui deve ser lido como um piso do que o método daria com `H` completo.

### 2.3 Um bandit, não uma população de bandits

**Paper (§4.2):** vários MABs com `c` e granularidade de discretização diferentes,
ensemble, e substituição periódica dos membros para atenuar a não-estacionariedade.
**Aqui:** um `BanditUCB` com janela deslizante.

A janela já trata a não-estacionariedade que importa neste domínio — a política mudando
embaixo do bandit — e a população de bandits acrescenta uma camada de meta-meta-seleção
cujo efeito seria indistinguível do ruído entre sementes num benchmark de três sementes.
Fica como extensão natural, não como dívida escondida.

### 2.4 O valor do braço é normalizado antes do bônus

O score do UCB soma um valor com um bônus de exploração. No Atari os retornos variam por
ordens de grandeza entre jogos, e o paper resolve isso jogo a jogo, pela população de
bandits. Aqui o problema aparece **dentro de uma execução**: no passo 100 mil todo braço
rende ~1 ponto e o bônus domina (seleção quase aleatória); no passo 5 M os bons rendem 80 e
o bônus vira ruído (seleção gulosa). O mesmo `c`, dois regimes opostos, sem ninguém ter
mexido em nada — a exploração do meta-controlador viraria função de quanto o agente já sabe
jogar.

Normalizar os valores para `[0, 1]` pelo mínimo e máximo **entre os braços** deixa `c` com
um significado estável: *quanto de incerteza vale o intervalo inteiro entre o pior e o
melhor braço*. `test_normalization_makes_the_selection_independent_of_the_reward_scale` e o
seu controle fixam as duas metades disso.

### 2.5 A softmax de seleção tem temperatura

Consequência direta de §2.4, e ela precisa ser dita porque é o tipo de coisa que quebra em
silêncio: com os scores comprimidos em `[0, 1]`, uma softmax de temperatura 1 nunca daria a
um braço mais que ~2,7× a probabilidade de outro. O bandit ficaria preso perto do uniforme
**por construção**, e a curva do LBC seria a da ablação de seleção aleatória sem que nada
acusasse. A temperatura padrão é 0,1 — uma diferença de 20% do intervalo entre o pior e o
melhor braço vale ~7× em probabilidade.

---

## 3. O que comparar com o quê

| par | o que a diferença mede |
|---|---|
| `10_lbc` × `01_ppo`, mesma semente | trocar exploração agendada por exploração selecionada — a pergunta principal |
| `10_lbc` × `10_lbc` com `selecao="aleatoria"` | quanto a parte *learnable* vale (ablação de seleção, Fig. 5) |
| `10_lbc` × `10_lbc` com `n_politicas=1` | quanto a população vale (ablação "reduzir H", Fig. 5) |

O par principal foi montado para isolar uma coisa só. Mesma rede, mesmo ambiente, mesmo
orçamento de passos, mesmo orçamento de gradiente (4 × 32) e — de propósito — **o mesmo
γ = 0,995 na política avaliada**, que é o `indice_alvo = 1`. Sem esse cuidado, a diferença
entre as curvas incluiria fator de desconto e o número não responderia nada.

As duas ablações são configuração, não código novo: `LBCConfig(selecao="aleatoria")` e
`LBCConfig(n_politicas=1, gammas=(0.995,), indice_alvo=0)`. As duas ganham sufixo próprio
na variante (`+selecao_aleatoria`, `+pop1`), porque `load_all` agrupa por
`(algo, variant, seed)` e identidade repetida vira uma curva só na arena.

---

## 4. O que olhar no log

O LBC tem mais jeitos de falhar em silêncio que os outros agentes daqui, porque ele tem uma
peça — o meta-controlador — que produz curvas plausíveis mesmo quando não está fazendo
nada. Quatro números no registro existem para isso:

| campo | leitura saudável | patologia |
|---|---|---|
| `razao_truncada` | fração não desprezível e não saturada | **0** = `μ` colou nas políticas alvo e o V-trace não corrige nada — o LBC virou um A2C caro. **~1** = a correção está saturada e o gradiente é o de um on-policy enviesado |
| `mab_entropia` | cai ao longo do treino, sem zerar | grudada em `log 16 = 2,77` = o bandit nunca decidiu; a curva deveria coincidir com a da seleção aleatória, e se **não** coincidir há algo errado em outro lugar |
| `omega_entropia` | passeia entre 0 e `log 3` | presa em `log 3` = a mistura nunca concentra e o mapeamento híbrido degenerou no uniforme; presa em 0 = degenerou em política única (o caso Agent57) |
| `tau_medio` | sobe ao longo do treino | plano = o eixo de entropia não está sendo usado |

`ev` (variância explicada) é lido como nos outros agentes: mede o crítico da política
**avaliada**, não a média da população.

---

## 5. O que ainda não foi medido

Nada nesta seção é afirmação — é a lista do que o repositório ainda não sabe sobre este
algoritmo.

* **Se o LBC ganha do PPO aqui.** Snake 10×10 com máscara de ação é um domínio de
  exploração fácil: o PPO já fecha ~90% dos tabuleiros. O LBC foi feito para jogos de
  Atari com exploração dura, e o resultado honesto pode perfeitamente ser "não compensa
  neste domínio" — que é um resultado, e entra na arena com a nota.
* **Se a parte *learnable* faz diferença.** Se `selecao="ucb"` e `selecao="aleatoria"`
  derem a mesma curva, o mérito estava no espaço de comportamento e não no bandit.
* **Quanto custa o tronco compartilhado** (§2.1).
* **Quanto o eixo de shaping por política valeria** (§2.2).
* **Custo em tempo de parede.** Uma época a mais de forward por atualização (o
  recálculo dos alvos V-trace) mais o custo das `N` cabeças. Não foi perfilado ainda; o
  `tools/perfil_dispositivo.py` é o lugar de fazê-lo.

---

## Referência

Jiajun Fan, Yuzheng Zhuang, Yuecheng Liu, Jianye Hao, Bin Wang, Jiangcheng Zhu, Hao Wang,
Shutao Xia. *Learnable Behavior Control: Breaking Atari Human World Records via
Sample-Efficient Behavior Selection*. ICLR 2023.
[arXiv:2305.05239](https://arxiv.org/abs/2305.05239)

Peças de apoio: V-trace em Espeholt et al., 2018
([arXiv:1802.01561](https://arxiv.org/abs/1802.01561)); UCB com janela deslizante em
Garivier & Moulines, 2008 ([arXiv:0805.3415](https://arxiv.org/abs/0805.3415)); o
antecessor que o LBC generaliza, Agent57, em Badia et al., 2020
([arXiv:2003.13350](https://arxiv.org/abs/2003.13350)).
