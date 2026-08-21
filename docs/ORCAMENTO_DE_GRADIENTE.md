# O orçamento de gradiente

**Com o mesmo orçamento de ambiente, gastar 5 M passos em ~38.000 atualizações em vez de
~2.400 levou o PPO de 62,19 para 80,90 pontos — e de 4,4% para 60,1% de jogos perfeitos.**
A dispersão entre sementes caiu por um fator de 5,4. Nenhuma linha do ambiente, da
observação, da rede ou do protocolo de avaliação foi tocada.

![ablação de orçamento](../assets/orcamento_light.png)

## O que foi comparado

Uma variável, com os dois braços em três sementes cada:

| | esparso (o padrão até agosto) | denso (o padrão de hoje) |
|---|---:|---:|
| `rollout` | 96 | 32 |
| `epochs` × `minibatches` | 3 × 8 | 4 × 32 |
| amostras por iteração | 49.152 | 16.384 |
| tamanho do minilote | 6.144 | 512 |
| iterações em 5 M passos | ~102 | ~305 |
| **atualizações de gradiente** | **~2.400** | **~38.300** |
| tempo de parede | 0,41 h | 0,83 h |

Tudo o mais idêntico: `num_envs=512`, γ 0,995, GAE(λ) 0,95, clip 0,2, entropia 0,02→0,002,
`lr` 3e-4→5e-5, KL alvo 0,03, `resnet_small` com 180.464 parâmetros, 5 M passos, avaliação
com 1.000 episódios greedy na semente 123.

Registros em `runs/ppo/resnet_small_esparso/seed{0,1,2}` (antes) e `runs/ppo/resnet_small/seed{0,1,2}` (padrão de hoje).

## O resultado

| execução | score final | mediana | tabuleiro cheio | melhor ckpt | pico em |
|---|---:|---:|---:|---:|---:|
| esparso seed0 | 64,56 | 71 | 0,0% | 62,72 | 95% |
| esparso seed1 | 70,58 | 79 | 13,1% | 69,46 | 95% |
| esparso seed2 | 51,43 | 60 | 0,0% | 51,23 | 95% |
| **esparso média** | **62,19** | 70 | **4,4%** | — | — |
| denso seed0 | 81,50 | 97 | 61,4% | 85,86 | 68% |
| denso seed1 | 78,87 | 97 | 54,7% | 81,98 | 84% |
| denso seed2 | 82,32 | 97 | 64,1% | 81,41 | 89% |
| **denso média** | **80,90** | 97 | **60,1%** | — | — |

## Três leituras, em ordem de importância

**1. A mediana bate no teto.** Nas três sementes densas a mediana é **97** — mais da metade
dos episódios são jogos perfeitos. A média de 80,90 não mede mais "quão bem o agente joga";
ela é essencialmente uma função da taxa de vitória, porque o teto está saturado. Daqui para
cima, a métrica que discrimina é `tabuleiro cheio`, e a média vira um resumo enganoso.

**2. A dispersão colapsou.** Desvio padrão de 9,79 no esparso contra **1,80** no denso, uma
razão de 5,4. Amplitude de 19,15 contra 3,45. Isto é mais útil para quem reproduz do que o
ganho de média: no orçamento esparso, cada atualização carrega muito peso e o destino final
depende de onde a trajetória caiu; com passos pequenos e numerosos, a média emerge e a
semente importa pouco. Uma execução do orçamento esparso não é uma medida do algoritmo — é
uma amostra de uma distribuição larga.

**3. A separação é completa.** A pior semente densa (78,87) supera a melhor do esparso
(70,58) por 8,3 pontos. Num teste de permutação exato com 3×3, separação completa dá
p = 0,10 bilateral — que é o **menor valor que este desenho pode produzir**, e não um sinal
fraco. Perseguir p < 0,05 exigiria mais sementes; o que sustenta a conclusão aqui é a
magnitude (+18,71 pontos, +55,7 pontos percentuais de vitória) contra um ruído entre
sementes de 1,80.

## Duas observações que o experimento não previa

**O crítico estava saudável o tempo todo.** A variância explicada fica entre 0,88 e 0,96
nas três sementes densas. A suspeita registrada na revisão (§2.2) — de que o `vf_clip` em
unidades absolutas travava o crítico — **não se confirma neste orçamento**. A explicação
econômica é que as duas suspeitas eram a mesma: o clip limita o valor a ±0,2 por
atualização, e com 128 atualizações por iteração em vez de 24 ele deixa de morder. Fica
como hipótese não testada para o orçamento esparso, cujas execuções não registram `ev` —
com a ressalva que a réplica no A2C acrescenta, mais abaixo.

**Em duas de três sementes a política piorou no fim.** Seed 0 caiu de 85,86 (68% do
orçamento) para 81,50, com o tabuleiro cheio indo de 74,1% para 61,4%; seed 1 caiu 3,11
pontos; seed 2 não caiu. Nas que caíram, a entropia terminou em ~0,138 de um máximo de
1,099, o `clipfrac` desabou para 0,03 e a KL medida para 0,003 — o último terço endurece a
política em vez de melhorá-la, e a versão endurecida joga pior o final de partida, que é
onde o agente precisa de flexibilidade para não se fechar no próprio corpo. O agendamento
de entropia e de `lr` é o suspeito, e é um experimento próprio.

Sobra folga, aliás: a KL medida fica de 2 a 10 vezes abaixo do alvo de 0,03, e o early-stop
por KL só corta épocas em 3% a 5% das iterações. O orçamento efetivo é o nominal, e os
passos ainda são conservadores.

## O que isto obriga

O contrato fixa os passos de ambiente e se cala sobre as atualizações — que hoje variam por
um fator de 64 entre os algoritmos do repositório: ~610 no ACKTR, ~1.953 no A2C, ~19.500 no
DQN, ~39.000 no PPO. Depois deste resultado, comparar algoritmos sem declarar esse eixo mede
orçamento de otimização, não algoritmo.

Igualar não é possível: A2C e ACKTR fazem uma atualização por rollout **por definição**, e
o teto estrutural deles fica perto de 10.000 sem descaracterizar o método. A regra
defensável é **declarar e estreitar**: cada algoritmo recebe o maior orçamento que a sua
definição permite, o número vai para `meta["atualizacoes"]` de cada execução, e a coluna
aparece na tabela. Disparidade reportada é informação; disparidade silenciosa é confundidor.

## A réplica no A2C

O resultado acima vem de uma família só, e no PPO o botão do orçamento é na verdade três
(`rollout`, `epochs`, `minibatches`) girados juntos. O A2C serve de réplica com **uma
variável só**: ele não tem épocas nem minilotes para reaproveitar o rollout, então mudar
`rollout` muda exatamente o número de atualizações e mais nada.

| | esparso | padrão de hoje |
|---|---:|---:|
| `rollout` | 16 | 5 |
| amostras por atualização | 8.192 | 2.560 |
| **atualizações em 5 M passos** | **~610** | **~1.953** |

O 5 não é escolha de conveniência: é o `t_max` canônico do A3C, o valor do artigo
original. O 16 era o que estava no repositório antes de o orçamento virar eixo declarado.

Os dois braços saem de `04_a2c` (padrão) e `95_a2c_orcamento_esparso` (controle), que fixa
`A2CConfig.esparso`. A previsão foi registrada **antes** de qualquer medição: efeito menor
que o do PPO, porque a razão entre os braços é de 3,2× contra 16× lá, e com um teto
estrutural que nem o braço denso alcança.

### O que já foi medido

| execução | atualizações | score final | mediana | p95 | tabuleiro cheio | pico em |
|---|---:|---:|---:|---:|---:|---:|
| esparso seed0 | 611 | 55,47 | 59 | 71 | 0,0% | 100% |
| denso seed{0,1,2} | ~1.953 | — | — | — | — | — |

Uma semente de um braço não é resultado: é o primeiro ponto. O número entra na conclusão
quando os dois braços tiverem três sementes cada — e a leitura do experimento do PPO vale
aqui em dobro, porque foi ela que mostrou que **uma execução do orçamento esparso é uma
amostra de uma distribuição larga**, não uma medida do algoritmo.

Três observações que já se sustentam sozinhas:

**O crítico está saudável — e não é o mesmo teste do PPO.** A variância explicada termina em
**0,986** com 611 atualizações. Isto fecha metade da lacuna deixada acima: não é a contagem
baixa de atualizações, por si, que estraga o crítico. Mas **não** testa a suspeita do
`vf_clip`, porque o A2C tem `vf_clip=0` — ele nem tem o botão. A hipótese aberta em
"Duas observações que o experimento não previa" continua aberta, e só um PPO esparso que
registre `ev` a fecha.

**A política ainda estava subindo no fim.** O melhor checkpoint é o último, a 100% do
orçamento, e os quatro pontos anteriores são 46,07 / 49,67 / 48,42 / 55,47. Não há o
endurecimento tardio que derrubou duas das três sementes densas do PPO — a entropia termina
em 0,105 de um máximo de 1,099, mas sem o custo que aparecia lá. Ou o A2C esparso é curto
demais para chegar à fase em que aquilo acontece, ou a fase é um artefato do reaproveitamento
de rollout. As duas leituras são testáveis; nenhuma está testada.

**O tempo de parede dos dois braços não é comparável.** A correção de retracing (§2.6 da
revisão, escalares como tensores) entrou **entre** esta execução e o braço denso. O esparso
gastou 0,778 h recompilando o grafo uma vez por atualização; o denso não vai recompilar
nenhuma. A coluna existe para o PPO, onde os dois braços rodaram no mesmo código — aqui ela
mede a correção, não o orçamento.

Uma nota de proveniência: esta execução saiu com `assinatura_pacote=782a8b8aa4af004f`, e o
`95_a2c_orcamento_esparso.ipynb` no HEAD carrega `df6c8eb2b2ca2f58`. A diferença entre as
duas é a correção de retracing e o próprio `A2CConfig.esparso()` — nenhuma delas muda um
número, mas reproduzir hoje não devolve a mesma assinatura. Foi também por isso que o
registro nasceu com `variant="resnet_small"` e precisou ser reetiquetado para
`resnet_small_esparso`: `A2CConfig.esparso()`, que carimba o sufixo, ainda não existia.

## Reprodução

```bash
python - <<'PY'
import json, numpy as np
for v in ("resnet_small_esparso", "resnet_small"):
    s = [json.load(open(f"runs/ppo/{v}/seed{i}/history.json"))["final"]["score_mean"]
         for i in range(3)]
    print(v, [round(x, 2) for x in s], "média", round(np.mean(s), 2),
          "dp", round(np.std(s, ddof=1), 2))
PY
```

Para o A2C, trocando `ppo` por `a2c` e o alcance das sementes pelo que já existe:

```bash
python - <<'PY'
import json
d = json.load(open("runs/a2c/resnet_small_esparso/seed0/history.json"))
print(d["variant"], d["config"]["rollout"], d["meta"]["atualizacoes"],
      round(d["final"]["score_mean"], 2))
PY
```

O gráfico sai de `tools/fig_orcamento.py`, que lê os mesmos `history.json`.
