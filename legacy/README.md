# Arquivo histórico — os 13 notebooks do `colab-rl`

Nenhum destes roda. Estão aqui por três motivos, nessa ordem de importância:

1. **Para o erro ficar registrado.** "Não funciona" não é diagnóstico. Cada notebook abaixo
   tem, no cabeçalho, a exceção exata da última execução salva — e foi dela que saiu boa
   parte do que o repositório novo conserta.
2. **Para o GitHub finalmente renderizá-los.** Todos foram salvos **sem extensão `.ipynb`**,
   e por isso nunca apareceram como notebook em lugar nenhum. Aqui têm a extensão.
3. **Para a substituição ser rastreável.** Cada um aponta para o que ocupou o lugar dele.

Eles **não são comparáveis** com o benchmark atual — mediam `snake.length` em vez de
`score`, com outra recompensa, outras ações e o tempo em episódios. As curvas de treino que
sobreviveram estão normalizadas em [`../results/legacy/`](../results/legacy/).

| notebook | pasta original | como morreu | substituído por |
|---|---|---|---|
| [`kfac_optimizer_test.ipynb`](nao_funcionavam/kfac_optimizer_test.ipynb) | `Not Working` | ImportError: `tensorflow.contrib.kfac` não existe desde o TF2 | `99_ablacoes.ipynb` — a pergunta 'o otimizador importa?' virou ablação medida com Adam/AdamW/RMSprop/Lion/SGD |
| [`new_kfac.ipynb`](nao_funcionavam/new_kfac.ipynb) | `Not Working` | ImportError: `tensorflow.contrib.kfac` não existe desde o TF2 | `99_ablacoes.ipynb` — a pergunta 'o otimizador importa?' virou ablação medida com Adam/AdamW/RMSprop/Lion/SGD |
| [`snakeai_acer.ipynb`](nao_funcionavam/snakeai_acer.ipynb) | `Not Working` | TypeError: KerasTensor passado a uma API TF sem dispatch — a matemática do ACER estava montada dentro do grafo funcional do Keras | `05_acer.ipynb` — reescrito do zero em Keras 3, com Retrace(λ) e região de confiança. Converge |
| [`snakeai_keras.ipynb`](nao_funcionavam/snakeai_keras.ipynb) | `Not Working` | AttributeError: `keras.backend.set_image_dim_ordering` foi removido do Keras | o pacote `snakeai/` inteiro, portado para Keras 3 com `channels_last` |
| [`snakeai_acer.ipynb`](tf1/snakeai_acer.ipynb) | `Working 1.x` | ValueError: expected shape=(None, 256, 100), found (None, 100) — a dimensão de tempo se perdeu entre a coleta e o update | `05_acer.ipynb` — reescrito do zero em Keras 3, com Retrace(λ) e região de confiança. Converge |
| [`snakeai_dqn_kfac_cnn3.ipynb`](tf1/snakeai_dqn_kfac_cnn3.ipynb) | `Working 1.x` | InvalidArgumentError: input depth must be evenly divisible by filter depth: 10 vs 3 | `99_ablacoes.ipynb` — a pergunta 'o otimizador importa?' virou ablação medida com Adam/AdamW/RMSprop/Lion/SGD |
| [`snakeai_dqn_kfac_kl_divergence_cnn3.ipynb`](tf1/snakeai_dqn_kfac_kl_divergence_cnn3.ipynb) | `Working 1.x` | sem erro na última execução salva | `99_ablacoes.ipynb` — a pergunta 'o otimizador importa?' virou ablação medida com Adam/AdamW/RMSprop/Lion/SGD |
| [`snakeai_dqn_adam_cnn3.ipynb`](tf2/snakeai_dqn_adam_cnn3.ipynb) | `Working 2.x` | NameError: name 'game' is not defined — célula de teste órfã, colada de outro notebook | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |
| [`snakeai_dqn_rmsprop_cnn3.ipynb`](tf2/snakeai_dqn_rmsprop_cnn3.ipynb) | `Working 2.x` | NameError: name 'nb_frames' is not defined — célula de teste órfã | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |
| [`snakeai_dqn_rmsprop_cnn2_kl_divergence.ipynb`](tf2/snakeai_dqn_rmsprop_cnn2_kl_divergence.ipynb) | `Working 2.x` | AttributeError: 'float' object has no attribute 'params' — `print(loss.params)` num float | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |
| [`snakeai_dqn_rmsprop_cnn4.ipynb`](tf2/snakeai_dqn_rmsprop_cnn4.ipynb) | `Working 2.x` | sem erro na última execução salva | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |
| [`snakeai_dqn_rmsprop_cnn4_1.ipynb`](tf2/snakeai_dqn_rmsprop_cnn4_1.ipynb) | `Working 2.x` | NameError: name 'nb_frames' is not defined — célula de teste órfã | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |
| [`snakeai_dqn_rmsprop_per_dueling_cnn4.ipynb`](tf2/snakeai_dqn_rmsprop_per_dueling_cnn4.ipynb) | `Working 2.x` | IndexError: list assignment index out of range — a memória PER nunca era pré-alocada | `02_dqn.ipynb` e `03_rainbow.ipynb` — os seis notebooks viraram flags de uma configuração |

## O que cada erro ensinou

- **`IndexError` da PER** virou `tests/test_memory.py::test_buffer_preallocates_and_never_raises_on_first_insert`.
- **`KerasTensor` do ACER** virou `tests/test_acer.py::test_the_model_is_only_input_to_outputs`.
- **A dimensão de tempo perdida** virou `test_stored_segment_keeps_the_time_axis`.
- **`set_image_dim_ordering`** virou `tests/test_nets.py::test_channels_last_everywhere`.
- **As células órfãs com `NameError`** viraram o gerador de notebooks: hoje nenhuma célula
  é escrita à mão, então não há como colar uma de outro lugar.
