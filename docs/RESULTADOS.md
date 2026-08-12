# Resultados

Gerado por `python -m snakeai.arena --all`. Não editar à mão.

![arena](../assets/arena_light.png)

| algoritmo | rede | params | sementes | passos | score médio | amplitude | mediana | máx | cheio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| _piso aleatório_ | — | — | — | 0 | **1,21** | — | 1 | — | 0% |

Score perfeito no 10×10: **97**.

## Execuções que não entraram na arena

Estão registradas em `runs/`, com curva e artefatos, mas não competem — o
motivo está em `meta["contract_violations"]` de cada uma.

- `acer/resnet_tiny/seed0`: avaliação final com 300 episódios, contrato exige 1000
- `alphazero/sims12/seed0`: avaliação final com 200 episódios, contrato exige 1000
