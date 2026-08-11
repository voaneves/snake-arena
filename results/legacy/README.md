# Curvas históricas do `colab-rl`

Seis execuções de DQN de **10.000 episódios** cada, resgatadas de
`colab-rl/models/benchmarking_models/`, com as colunas nomeadas (o original era
`,0,1,2,3`) e normalizadas pela regra **`score = comprimento - 3`**.

Ficam em CSV, não em JSON: são a fonte de dados bruta. `snakeai.record.from_legacy_csv`
converte para o esquema do repositório na hora de plotar, sem duplicar 10 MB no git.

## Estas curvas não competem

Todas nascem com `comparable=False`. Foram medidas no ambiente antigo, com recompensa
`+comprimento` ao comer, estado ordinal de um canal (com a cabeça sobrescrita pelo corpo),
cinco ações absolutas e o eixo em **episódios**, não em passos. No gráfico da arena
aparecem como linhas tracejadas cinza — contexto histórico, nunca competidor.

Segunda ressalva, igualmente importante: estes são **scores de treino, com exploração**
(eps-greedy), não o benchmark greedy de 1.000 episódios do contrato. O número honesto do
DQN sairá do DQN portado, na Fase 3.

## O que havia lá

| variante | o que era | episódios | melhor média móvel (100) | média final | máx |
|---|---|---|---|---|---|
| `epsgreedy.csv` | DQN + experience replay, eps-greedy | 10,000 | **17.41** | 15.01 | 38 |
| `epsgreedy_3steps.csv` | DQN + ER, eps-greedy, retornos de 3 passos | 10,000 | **14.49** | 13.17 | 32 |
| `epsgreedy_double.csv` | Double DQN + ER, eps-greedy | 10,000 | **13.27** | 11.94 | 26 |
| `epsgreedy_target_3steps.csv` | DQN + ER, rede alvo, 3 passos | 10,000 | **13.33** | 12.79 | 27 |
| `epsgreedy_per.csv` | DQN + prioritized experience replay | 10,000 | **18.32** | 16.81 | 38 |
| `epsgreedy_target_per.csv` | DQN + PER, rede alvo | 10,000 | **15.32** | 14.09 | 29 |

O melhor deles, `epsgreedy_per`, encostou em **18,3** de média móvel durante o treino, com
máximo de 38 — num tabuleiro onde o score perfeito é 97.

Isso vira o **critério de fidelidade da Fase 3**: o DQN portado, rodando a variante
equivalente, precisa reproduzir ou superar esse patamar. Se ficar muito acima, a diferença
é atribuível ao ambiente e ao estado corrigidos — e isso merece um parágrafo no README, não
uma comemoração silenciosa.

## Colunas

| coluna | o que é |
|---|---|
| `episode` | índice do episódio de treino (0 a 9999) |
| `length` | comprimento da cobra ao fim do episódio — **começa em 3** |
| `episode_steps` | passos que o episódio durou |
| `loss` | perda do DQN naquele episódio |
| `reward` | recompensa acumulada, na escala antiga (`+comprimento` ao comer) |
| `score` | `length - 3` — a métrica do contrato, adicionada na normalização |
