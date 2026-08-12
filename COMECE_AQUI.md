# Comece aqui

Estado em 12 de agosto de 2026. **O código está pronto e testado; o benchmark ainda não foi
rodado.** Este documento diz exatamente o que fazer, nessa ordem.

## 1. Versionar o que está no disco (2 minutos)

O repositório local tem um commit só e tudo o mais está solto. No PowerShell:

```powershell
cd D:\GitHub\snake-arena
git add -A
git commit -m "snake-arena: pacote completo, 7 algoritmos, 347 testes"
```

Se sobrou uma pasta `_to_delete\`, pode apagar — são notebooks com a numeração antiga.

## 2. Publicar no GitHub (5 minutos)

**Isto é pré-requisito para os badges "Open in Colab" funcionarem.** Eles apontam para
`github.com/voaneves/snake-arena`, que ainda não existe. Sem isso os notebooks continuam
utilizáveis — é só fazer upload manual no Colab — mas o botão não abre.

```powershell
gh repo create voaneves/snake-arena --public --source=. --push
```

## 3. Rodar o benchmark (é aqui que o gráfico se preenche)

Cada notebook é autocontido: **suba o `.ipynb` no Colab e rode**. Nada de clonar.

`Runtime → Change runtime type → GPU (T4)` antes de começar.

Ordem sugerida, por retorno sobre o tempo:

| # | notebook | por quê |
|---|---|---|
| 1 | `06_alphazero.ipynb` | candidato mais forte. **Use `num_simulations` entre 64 e 128** — com 12 a destilação não funciona, e isso está medido |
| 2 | `01_ppo.ipynb` | a referência; tudo é lido contra ela |
| 3 | `03_rainbow.ipynb` | o topo da linhagem DQN |
| 4 | `02_dqn.ipynb` | a base, para o Rainbow ter contra o que ser comparado |
| 5 | `04_a2c.ipynb` | o controle experimental do PPO |
| 6 | `07_muzero.ipynb` | responde "quanto custa não ter o simulador" |
| 7 | `05_acer.ipynb` | o mais difícil; já se sabe que converge |
| 8 | `99_ablacoes.ipynb` | arquitetura e otimizador, os dois eixos que 2019 nunca mediu |

Cada um roda três vezes, com `SEMENTE` em 0, 1 e 2. Deixe `PASSOS` no padrão: o contrato
exige o mesmo orçamento para todos, e o notebook avisa se você mudar.

Ligue `USAR_DRIVE = True`. A sessão do Colab vai cair; com o Drive o treino continua de onde
parou.

## 4. Montar a arena

Baixe as pastas `runs/` do Drive, coloque em `runs/` do repositório e:

```powershell
python -m snakeai.arena --all
```

Ele regenera `assets/arena_light.png`, `assets/arena_dark.png` e `docs/RESULTADOS.md`.
Execuções que não obedecerem ao contrato aparecem listadas como excluídas, com o motivo.

## O que ainda não existe

- **Nenhum resultado oficial.** O gráfico está vazio de propósito.
- **`models/`** vazio — se enche quando houver o que exportar.
- **CI** não configurado. `pytest` roda local em ~6 minutos.
- **Os repositórios antigos** (`snake-on-pygame`, `colab-rl`) ainda não apontam para cá, e
  os três bugs de estado do `snake.py` continuam lá.

## Se algo quebrar

`pytest` da raiz. São 347 testes e a mensagem de falha foi escrita para dizer o que
quebrou, não só que quebrou.
