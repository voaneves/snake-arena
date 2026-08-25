# SOAP — memória discreta para uma observação que não é markoviana

O Snake deste repositório *parece* um MDP: tabuleiro completo, informação perfeita,
determinístico. Não é — e a prova está no próprio repositório desde
[`CANAL_DE_FOME.md`](CANAL_DE_FOME.md).

A observação do contrato tem 5 canais: corpo, cabeça, decaimento da cauda, comida e
comprimento. **Nenhum deles é a fome**, e o episódio é truncado em
`100 + 2·comprimento` passos sem comer. Ou seja: dois estados pixel a pixel idênticos, um
com fome 5 e outro com fome 105, têm valores diferentes e exigem ações diferentes — e a
rede não tem como distingui-los. Isso é a definição de observabilidade parcial.

Há três respostas possíveis, e o repositório já gastou uma:

| resposta | onde | resultado |
|---|---|---|
| acrescentar o relógio à **observação** | `97_ppo_canal_de_fome` | 7,8 pontos **abaixo**, atrás em 17 dos 18 pontos de avaliação, e `comparable=False` de brinde |
| dar **memória contínua** ao agente | não implementado (LSTM/GRU) | — |
| dar **memória discreta** ao agente | `11_soap` | esta página |

O SOAP é a terceira, e é a única que cabe **dentro** do contrato: os 5 canais continuam
sendo 5 canais, a entrada da rede não muda, a curva compete sem asterisco. O que muda é que
o agente passa a carregar uma variável latente entre os passos.

---

## 1. O que o algoritmo faz

### 1.1 A fatoração

Duas políticas, sobre `Z = 4` opções discretas:

* **sub-política** `π_θ(a|s,z)` — como agir, dado que a opção corrente é `z`;
* **transição de opção** `π_ψ(z'|s,a,z)` — para qual opção ir, dado o estado, a ação
  tomada e a opção anterior.

A segunda é a contribuição de fatoração do paper. No Option-Critic a opção é escolhida a
partir do estado (`π(z|s)`) e uma função de terminação decide quando trocar; aqui a próxima
opção depende explicitamente da **anterior**, e é isso que permite a uma opção persistir por
conta própria — sem um mecanismo de terminação separado, e sem ser re-sorteada a cada passo.

Um crítico por opção, `V(s,z)`, fecha o conjunto.

### 1.2 A crença

O agente nunca sabe em que opção está: `z` não é observado nem supervisionado. O que ele
carrega é `ζ_t(z) := p(z_t | s_{0:t}, a_{0:t-1})`, a crença para a frente, atualizada pelo
filtro:

```
α_t        = Σ_z ζ_t(z) π_θ(a_t|s_t,z)                                (o marginal da ação)
ζ_{t+1}(z') = Σ_z ζ_t(z) π_θ(a_t|s_t,z) π_ψ(z'|s_t,a_t,z) / α_t
```

`α_t` é a probabilidade da ação que o ambiente de fato viu — é ela que faz o papel de
`π(a|s)` num PPO comum, e é o denominador de tudo.

**Causal, e é o ponto.** O mesmo paper propõe um segundo algoritmo, o PPOEM, que usa o
*forward-backward* inteiro e atribui opções em retrospecto — com informação que o agente não
terá na hora de agir. O paper mostra que isso degrada conforme a sequência cresce, e o SOAP
existe justamente para otimizar a atribuição que o agente **vai conseguir fazer** ao vivo.

`ζ` começa uniforme no início de cada episódio, e volta ao uniforme quando o episódio
acaba. Deixá-la atravessar a morte da cobra faria o agente começar a partida convencido de
estar num regime que pertence à anterior — e nada quebraria; a curva só ficaria pior.
`test_the_belief_resets_when_the_episode_ends` trava isso.

### 1.3 A vantagem que propaga

O gradiente de `log α_t` atravessa `ζ_t`, que depende de todos os passos anteriores. Derivar
isso a mão daria uma retropropagação pelo tempo. O paper mostra (§5.2) que ela colapsa numa
recursão para trás fechada:

```
A^GOA_t(z') = Σ_z A^GAE_t(z)·ζ_t(z)  +  (1−d_t)·[ U_{t+1}(z') − E_{ζ_{t+1}}U_{t+1} ]
U_t(z)      = Σ_{z'} A^GOA_t(z')·p_Θ(a_t,z'|s_t,z) / α_t
```

Em português: escolher `a_t` e terminar o passo na opção `z'` vale a **vantagem imediata**
do passo (média sobre a crença atual) mais o quanto `z'` é uma opção melhor que a média para
o futuro. O primeiro termo não depende de `z'` — é o que um PPO comum já teria. Todo o
conteúdo de opção está no segundo.

Duas propriedades desse segundo termo importam, e as duas têm teste:

* ele é **centrado** sob `ζ_{t+1}`. Se não fosse, deslocaria o gradiente da sub-política em
  vez de só redistribuir entre as opções, e o SOAP viraria um PPO com baseline enviesada;
* ele **morre na fronteira do episódio** (`1−d_t`). Sem isso a utilidade do episódio
  seguinte vaza para dentro deste — o mesmo defeito da §1.1 da revisão, agora no eixo das
  opções.

A normalização por `α_t` dentro de `U` também não é decorativa: sem ela, uma ação improvável
no marginal amplifica o sinal de opção e a recursão diverge nas primeiras iterações, quando
`α` é pequeno.

### 1.4 A perda, na forma implementada

O paper escreve a perda com `p_Θ` cru e o clipping em torno de `p_Θ_velho`. Aqui ela está na
forma algebricamente equivalente:

```
L = − Σ_{z,z'} w(z,z') · min( ρ·A^GOA(z'), clip(ρ, 1±ε)·A^GOA(z') )
w(z,z') = ζ(z)·p_velho(a_t,z'|s_t,z) / α_t          ρ = p_novo / p_velho
```

`w` é a **responsabilidade** do par `(z_t, z_{t+1})`: a posteriori dado o histórico e a ação
tomada. Ela soma exatamente 1 sobre os pares — `test_the_responsibilities_form_a_distribution`
confere —, então a perda é uma média ponderada de perdas de PPO e `ρ` vive perto de 1, como
num PPO comum. Multiplicar `p_Θ` cru dá o mesmo gradiente a menos do fator constante
`p_velho`, mas com a escala variando por amostra.

O crítico é regredido **sob a crença**: uma opção que o agente acha improvável naquele
estado não deve puxar o valor dela para o retorno observado.

---

## 2. O controle é exato, e isso é o ponto

Com `n_opcoes=1` o SOAP não é *parecido* com o PPO — ele **é** o PPO:

| peça | com `Z = 1` |
|---|---|
| crença | `ζ ≡ 1` |
| marginal | `α_t = π(a_t\|s_t)` |
| transição | `π_ψ ≡ 1` |
| GAE por par | colapsa no `compute_gae` do PPO, termo a termo |
| vantagem de opção | `A^GOA = A^GAE` (o termo centrado é identicamente zero) |
| responsabilidade | `w ≡ 1` |
| perda | `−min(ρA, clip(ρ)A)` — o PPO |

`tests/test_soap.py` prova as igualdades numericamente, não por argumento. Isso é o que
torna a linha da arena legível: a diferença entre `11_soap` e `11_soap+op1` é atribuível às
opções e a mais nada, sem depender de ninguém acreditar na implementação. E a diferença
entre `11_soap` e `01_ppo` fica sendo opções **mais** a diferença de arquitetura das cabeças
— que é pequena e está declarada.

---

## 3. Colapso de opções: os quatro sintomas

É o modo de falha clássico de todo método de opções, e ele **não levanta exceção**: as
opções colapsam, o SOAP vira um PPO com 4× as cabeças, e a curva parece só um PPO um pouco
pior. Quatro números no registro existem para que isso seja visto em vez de suposto.

| campo | leitura saudável | patologia |
|---|---|---|
| `opcao_divergencia` | cresce e se estabiliza acima de ~0,1 | perto de zero = as `Z` sub-políticas fazem a mesma coisa. **É o sintoma mais direto**: as opções existem no papel e não no comportamento |
| `opcao_uso_entropia` | abaixo de `log Z` mas bem acima de zero | igual a `log Z` o treino inteiro = a crença nunca se decide, `ζ` fica no uniforme e as sub-políticas são sempre misturadas. Zero = uma opção só |
| `opcao_persistencia` | alta e crescente | perto de `1/Z` = a opção é re-sorteada a cada passo, e "temporalmente estendida" — a razão de existir do arcabouço — deixou de valer |
| `goa_amplitude` | não desprezível | zero = `A^GOA` é constante em `z'`, `π_ψ` não recebe gradiente e a política de troca é ruído |

No **início** do treino os quatro estão no estado degenerado de propósito: os logits nascem
com ganho 0,01, então as sub-políticas são quase idênticas e a crença é quase uniforme.
`test_the_options_start_indistinguishable_and_the_diagnostic_says_so` fixa esse ponto de
partida, para que o diagnóstico signifique alguma coisa depois.

Se o colapso acontecer, o primeiro botão é `ent_opcao_coef` — um bônus de entropia sobre
`π_ψ`, **zero por padrão** porque o paper não o tem. A execução que o gira ganha marca
própria na variante (`+entz…`), porque ela não é mais o algoritmo do paper.

---

## 4. O que comparar com o quê

| par | o que a diferença mede |
|---|---|
| `11_soap` × `11_soap` com `n_opcoes=1` | o valor das opções, com a implementação inteira congelada — o controle exato |
| `11_soap` × `01_ppo` | o mesmo, mais a diferença de cabeças; a comparação de leitura direta na arena |
| `11_soap` × `97_ppo_canal_de_fome` | **memória contra observação**: duas respostas para a mesma não-markovianidade. A segunda já foi medida e perdeu; esta é a rodada de volta |

O terceiro par é o mais interessante e o único que exige cuidado: `97` roda com 6 canais e é
`comparable=False`, então a comparação é **contra o PPO de cada lado**, não direta. Isto é,
o que se compara é *quanto cada resposta rendeu em relação ao seu próprio controle*.

---

## 5. Detalhes de implementação que não são detalhes

**A política avaliada é o marginal.** O protocolo do contrato é `argmax`, e o argmax de
`Σ_z ζ(z)π_θ(a|s,z)` é a ação que a política de fato escolhe. O argmax de qualquer
`π_θ(·|s,z)` isolada seria outra política, e o número publicado seria de um agente que nunca
jogou.

**A avaliação tem memória.** `PoliticaComOpcoes` implementa o mesmo contrato de duas metades
que o `PoliticaRecorrente` do DreamerV3: `__call__` devolve os logits e `apos_passo` recebe a
ação que **de fato** saiu — que pode não ser o argmax, se o filtro de segurança agiu — e onde
o episódio terminou. Avaliar com `ζ` congelado no uniforme não daria erro; daria um número
mais baixo, e a conclusão "opções não ajudam aqui" viria do defeito de medição.

**O GIF também.** `quadros_do_episodio` não chamava `apos_passo`, então o vídeo de qualquer
política com memória era gravado com o estado interno congelado no valor inicial — o agente
do vídeo não era o agente da curva. Corrigido junto com este algoritmo; valia, sem que
ninguém notasse, para o DreamerV3.

**A exportação não afirma paridade.** Um `.tflite` que recebe só a observação não consegue
reproduzir uma política cuja ação depende de estado interno. Os arquivos continuam sendo
gerados e medidos; a conferência de paridade de ação é pulada, com aviso, para qualquer
política que exponha `apos_passo`.

**Fome é truncamento, e o crítico é por opção.** O bootstrap `γ·V(s_final, z')` depende de
`z'`, então a recompensa do rollout carrega um eixo de opção. Um bootstrap único ensinaria a
todas as opções o valor terminal de uma delas.

---

## 6. O que ainda não foi medido

* **Se as opções emergem neste domínio.** A hipótese é que elas se separem por regime —
  "caçar comida" contra "desenrolar o corpo", ou algo que só o GIF revela. O resultado
  honesto pode perfeitamente ser o colapso, e aí a leitura é que 106 passos de horizonte de
  fome não bastam para justificar memória discreta.
* **Se `Z = 4` é o número certo.** O paper usa 4 no Atari. Aqui não há razão a priori;
  `n_opcoes` é um eixo de ablação barato.
* **SOAP contra uma GRU.** A comparação que fecharia a pergunta "memória discreta ou
  contínua?" exige um baseline recorrente que o repositório não tem.
* **Custo em tempo de parede.** Uma cabeça `Z × A × Z` e um rollout que carrega `ζ` em
  NumPy. Não foi perfilado; `tools/perfil_dispositivo.py` é o lugar.

---

## Referência

Shu Ishida, João F. Henriques. *SOAP-RL: Sequential Option Advantage Propagation for
Reinforcement Learning in POMDP Environments*, 2024.
[arXiv:2407.18913](https://arxiv.org/abs/2407.18913) ·
[código dos autores](https://github.com/shuishida/SoapRL)

O arcabouço de opções vem de Sutton, Precup & Singh (1999); a fatoração que o SOAP
substitui é a do Option-Critic, Bacon, Harb & Precup
([arXiv:1609.05140](https://arxiv.org/abs/1609.05140)).
