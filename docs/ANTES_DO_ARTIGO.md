# O que falta antes de congelar os resultados do artigo

Revisão do estado do repositório com uma pergunta só: **o que hoje entra num artigo e o
que não entra**. É uma lista de decisões, não de tarefas de código.

## Resposta curta

Dá para escrever agora tudo que não depende dos números: introdução, trabalhos
relacionados, o ambiente, o contrato de comparabilidade, o protocolo de avaliação, a
metodologia e — inteira — a ablação do canal de fome, que é o único experimento fechado do
repositório.

**A tabela de resultados não pode ser congelada ainda.** Não por falta de execuções, mas
porque as três que existem foram medidas com **réguas diferentes**, e a que aparece em
primeiro lugar é a mais frágil das três.

## O que está sólido

**PPO, três sementes, protocolo atual.** 64,56 / 70,58 / 51,43, média 62,19.

E há uma pergunta natural que já está respondida: *as correções da revisão invalidam essas
execuções?* Não. Reconstruí a versão pré-correção dos arquivos que o PPO percorre no
treino e rodei quatro iterações com a mesma semente nas duas versões: `pg`, `vf`, `ent`,
`kl`, `clipfrac`, a soma dos pesos e o score de avaliação saem **idênticos até o último
dígito**. As mudanças no caminho do PPO são de registro (variância explicada, contagem de
atualizações) e de avaliação de checkpoint — nenhuma toca no laço de treino. As três
curvas valem.

## O que não entra como está

### 1. ACKTR 83,91 — o número mais alto do repositório, e o mais frágil

`runs/acktr/resnet_small/0`. Dois problemas independentes:

**Uma semente.** O contrato pede três (`COMPARABILITY.md`), mas isso é convenção escrita,
não código: `validate()` não conta sementes, e a arena publica a linha com "sementes 1" ao
lado de um PPO com 3. A ablação do canal de fome mediu a amplitude entre sementes do PPO
em **19,1 pontos** — a distância entre 83,91 e 64,56 é 19,4. Uma semente não sustenta a
conclusão "ACKTR ganha do PPO", que é justamente a manchete que essa linha produz.

**Régua antiga.** A execução é de 12/08; a correção do protocolo de avaliação entrou em
14/08 (commit `edac33d`, o mesmo que trouxe a maçã final do episódio vencedor e a
decomposição por causa de fim). Duas evidências no próprio arquivo:

* `final` não tem `fim_fome` / `fim_colisao` / `fim_tabuleiro_cheio` — as chaves que toda
  execução posterior tem;
* `win_rate = 0,672` com `score_max = 96`. Sob a definição atual, `win_rate` é a fração de
  episódios com score igual a 97: com máximo 96, teria que ser **zero**. O número veio de
  outra fórmula.

O `score_max = 96` num tabuleiro cujo perfeito é 97 é a assinatura do episódio vencedor a
que faltava a última maçã. Ou seja: o 83,91 provavelmente **subestima** o ACKTR, e ainda
assim não pode ser publicado — não é o mesmo instrumento que mediu o PPO. E `validate()`
deixou passar, porque confere `episodes == 1000` e `completo`, não *qual protocolo*
produziu os números.

### 2. DQN semente 0 — medida do agente com o defeito que acabamos de corrigir

A execução já está em `runs/dqn/base/seed0/` e a arena a incluiu — **mas ela é
pré-correção**, e três evidências independentes dizem isso: `created_at` de 18/08 às 14:48,
ausência de `meta["atualizacoes"]` (que só o código corrigido grava) e o notebook daquela
execução (`44d39e0`) sem `desfaz_truncamento`.

Ou seja: ela treinou com a fome gravada como terminação **e** com a observação do episódio
seguinte no lugar do estado final. O resultado tem exatamente a assinatura disso:

| | DQN semente 0 (pré-correção) | PPO (3 sementes) |
|---|---:|---:|
| score final | 57,43 | 51,4 – 70,6 |
| morte por fome | **34,3%** | 1,9 – 4,7% |
| tabuleiro cheio | 0,3% | 0 – 13,1% |
| melhor checkpoint | 65,66 (passo 4,50 M) | 62,7 |
| avaliação em 4,75 M | 27,98 | — |
| horas de parede | 2,4 | 0,4 |

Trinta e quatro por cento dos episódios terminando por inanição, contra 2–5% do PPO, é o
sintoma que a correção do truncamento ataca: sem bootstrap, o alvo de TD ensina que
sobreviver muito termina em −0,5. Publicar "o DQN é instável e morre de fome" como
*achado*, tendo um defeito conhecido que produz precisamente essa assinatura, é a primeira
pergunta que um revisor faria — e a resposta honesta seria "não sabemos".

O 65,66 → 27,98 → 57,43 nas últimas avaliações é instabilidade real de DQN, mas a revisão
também aponta uma causa mecânica não medida (§2.4: a rede alvo tem ~8 atualizações de
defasagem, não ~2.000). Vale saber qual é qual antes de escrever a interpretação.

Uma boa notícia no meio: essa execução **foi medida com a régua atual** — tem as chaves de
causa de fim e `score_max = 97`. O problema dela é o agente, não o instrumento. E o custo
de refazer é conhecido: **2,4 horas** de parede (contra 0,4 h do PPO), porque `learn_every`
faz uma atualização a cada 256 passos de ambiente com laço Python na memória. Três sementes
de DQN são ~7 horas — o suficiente para valer decidir §2.4 antes de gastá-las.

### 3. A ablação do canal de fome não aparece na arena

`comparable=False` faz a execução sumir das três listas de `arena.py` — não entra no
gráfico, não entra na tabela, e não entra nem na seção "execuções que não entraram"
(§1.9 da revisão, ainda em aberto). Para o artigo isso não é fatal, porque o
`CANAL_DE_FOME.md` é autossuficiente, mas a figura da arena não conta a história inteira
enquanto for assim.

## Três decisões antes de gastar GPU

Cada execução são ~40 minutos de P100. Nove algoritmos × três sementes são ~18 horas. Não
vale começar essa fila antes de decidir:

**1. O orçamento de gradiente (§2.1).** Se `PPOConfig.denso()` levar o PPO de 62 para, por
exemplo, 75, a tabela inteira muda de sentido — e a comparação só é justa se todos os
algoritmos receberem o mesmo tratamento. Isto é *uma* execução de decisão, e ela vem
antes das dezoito. Junto vem a medição da variância explicada (§2.2): se o crítico do PPO
estiver explicando ~0 do retorno, o `vf_clip` sobe na frente de tudo e a fila muda de novo.

**2. O truncamento nos agentes sequenciais (§1.1, parte pendente).** ACER, DreamerV3,
AlphaZero e MuZero ainda tratam fome como terminação. Se eles entram no artigo, entram
depois da correção — senão o artigo compara "algoritmo + tratamento de truncamento", que é
a crítica que o próprio `COMPARABILITY.md` levanta contra o repositório antigo.

**3. Congelar o commit.** Todas as execuções do artigo precisam sair da **mesma** versão do
pacote, e o `history.json` grava `meta["commit"] = "desconhecido"` quando o notebook roda
fora de um clone git — que é o caso no Kaggle. Sem isso, daqui a três meses não há como
provar qual código produziu qual curva. Registrar a assinatura do notebook (que o gerador
já calcula) resolveria: ela identifica o pacote inteiro.

## Ordem sugerida

1. **Marcar a execução de DQN que já está lá como o "antes"** — ela é evidência útil do
   defeito, desde que o registro diga isso. Hoje ela entra na arena como resultado oficial,
   ao lado do PPO, sem nada que a distinga.
2. **Uma execução de decisão do PPO** com `denso()`, semente 0, e a variância explicada
   ligada. Compara com a `seed0` que já existe (mesma semente, mesmo tudo).
3. **Decidir a configuração final** à luz de (2) e corrigir o truncamento dos sequenciais.
4. **Congelar o commit** e rodar a fila: 3 sementes por algoritmo, incluindo **re-rodar o
   ACKTR** (é a manchete e está com 1 semente e régua antiga) e o DQN.
5. **Escrever a seção de resultados** com a arena regenerada.

## Enquanto isso, o que já dá para escrever

O ambiente e o contrato; a discussão de comparabilidade (é a contribuição metodológica mais
forte do trabalho, e independe dos números); o protocolo de avaliação, incluindo a
correção do viés de amostra que moveu o piso de 1,08 para 1,21; a ablação do canal de fome
inteira, com o gráfico; e a seção de limitações — para a qual esta revisão é matéria-prima
direta.

## Emendas pequenas que evitariam repetir tudo isso

Nenhuma é urgente, todas são baratas:

* **`validate()` exigir as chaves de causa de fim.** Elas nasceram junto com o protocolo
  atual, então exigi-las reprova automaticamente qualquer registro medido com a régua
  antiga — o caso do ACKTR teria sido pego sozinho.
* **Contar sementes.** A arena já mostra a coluna; falta um aviso quando `n < 3`, na mesma
  linguagem das outras violações.
* **Gravar a assinatura do pacote no registro.** O gerador já a calcula por notebook; ela é
  o `commit` que o Kaggle não tem.
* **`.gitattributes` com `* text=auto eol=lf`.** O repositório não tem nenhum e o
  `core.autocrlf` está indefinido — foi assim que os arquivos voltaram com CRLF.
