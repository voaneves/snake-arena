# Referências — o que este repositório implementa, e de onde veio

Todo algoritmo, componente e escolha de projeto do `snake-arena` que tem origem publicada,
com a implementação correspondente ao lado. Serve a três leitores diferentes:

* quem quer **conferir** que a implementação faz o que o paper diz;
* quem vai **escrever o artigo** e precisa da bibliografia sem reconstruí-la a partir dos
  docstrings;
* quem chegou por um algoritmo só e quer saber **o que mais está aqui**.

## Como estes identificadores foram conferidos

Cada `arXiv:ID` foi aberto e comparado com o título e os autores do trabalho que a
implementação diz seguir. Isso não é zelo excessivo: **um identificador trocado leva a um
paper existente e plausível**, que é o pior tipo de erro de citação — não quebra nada, e
ninguém confere. `tests/test_notebooks.py::test_the_algorithm_table_links_the_defining_paper`
fixa o vínculo entre cada notebook e o paper que o define, e
`test_every_arxiv_link_in_the_readme_is_in_the_bibliography` impede que a tabela do README e
esta página divirjam.

Trabalhos sem arXiv aparecem com DOI ou com a referência da conferência.

---

## 1. Os algoritmos com notebook próprio

Um por notebook, na ordem em que aparecem na arena.

| # | Notebook | Trabalho | Onde está a implementação |
|---|---|---|---|
| 01 | PPO | Schulman, Wolski, Dhariwal, Radford & Klimov, 2017 — *Proximal Policy Optimization Algorithms* · [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | `snakeai/agents/ppo.py` |
| 02 | DQN | Mnih et al., 2013 — *Playing Atari with Deep Reinforcement Learning* · [arXiv:1312.5602](https://arxiv.org/abs/1312.5602) · versão Nature 2015 · [doi:10.1038/nature14236](https://www.nature.com/articles/nature14236) | `snakeai/agents/dqn.py` |
| 03 | Rainbow | Hessel et al., 2017 — *Rainbow: Combining Improvements in Deep Reinforcement Learning* · [arXiv:1710.02298](https://arxiv.org/abs/1710.02298) | `snakeai/agents/rainbow.py` |
| 04 | A2C | Mnih et al., 2016 — *Asynchronous Methods for Deep Reinforcement Learning* · [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) | `snakeai/agents/a2c.py` |
| 05 | ACER | Wang et al., 2016 — *Sample Efficient Actor-Critic with Experience Replay* · [arXiv:1611.01224](https://arxiv.org/abs/1611.01224) | `snakeai/agents/acer.py` |
| 06 | AlphaZero | Silver et al., 2017 — *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* · [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) | `snakeai/agents/alphazero.py`, `snakeai/search/` |
| 07 | MuZero | Schrittwieser et al., 2019 — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* · [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | `snakeai/agents/muzero.py`, `snakeai/nets/muzero.py` |
| 08 | ACKTR | Wu, Mansimov, Liao, Grosse & Ba, 2017 — *Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation* · [arXiv:1708.05144](https://arxiv.org/abs/1708.05144) | `snakeai/agents/acktr.py` |
| 09 | DreamerV3 | Hafner, Pasukonis, Ba & Lillicrap, 2023 — *Mastering Diverse Domains through World Models* · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | `snakeai/agents/dreamerv3.py`, `snakeai/nets/dreamer.py` |
| 10 | LBC | Fan, Zhuang, Liu, Hao, Wang, Zhu, Wang & Xia, 2023 — *Learnable Behavior Control: Breaking Atari Human World Records via Sample-Efficient Behavior Selection* · [arXiv:2305.05239](https://arxiv.org/abs/2305.05239) | `snakeai/agents/lbc.py`, `snakeai/bandit.py` |
| 11 | SOAP | Ishida & Henriques, 2024 — *SOAP-RL: Sequential Option Advantage Propagation for Reinforcement Learning in POMDP Environments* · [arXiv:2407.18913](https://arxiv.org/abs/2407.18913) | `snakeai/agents/soap.py` |
| 12 | ACEKTR | George, Laurent, Bouthillier, Ballas & Vincent, 2018 — *Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis* · [arXiv:1806.03884](https://arxiv.org/abs/1806.03884) | `snakeai/agents/acektr.py`, classe `EKFac` em `snakeai/kfac.py` |

Os notebooks `95`–`99` são **ablações deste repositório**, não algoritmos publicados, e por
isso não têm paper. Dar a eles o paper do algoritmo base sugeriria que a variação é do paper,
e não é. Ver `docs/ORCAMENTO_DE_GRADIENTE.md` e `docs/CANAL_DE_FOME.md`.

---

## 2. As peças que compõem os algoritmos

Componentes que não têm linha própria na arena, mas sem os quais os de cima não são o que
dizem ser.

### 2.1 Estimadores e correções off-policy

| Peça | Onde é usada | Trabalho |
|---|---|---|
| GAE(λ) | vantagem do PPO, A2C, ACKTR, ACEKTR e SOAP | Schulman, Moritz, Levine, Jordan & Abbeel, 2015 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation* · [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| Retrace(λ) | o estimador off-policy do ACER | Munos, Stepleton, Harutyunyan & Bellemare, 2016 — *Safe and Efficient Off-Policy Reinforcement Learning* · [arXiv:1606.02647](https://arxiv.org/abs/1606.02647) |
| V-trace | o estimador off-policy do LBC (`vtrace()`) | Espeholt et al., 2018 — *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures* · [arXiv:1802.01561](https://arxiv.org/abs/1802.01561) |

Os três resolvem o mesmo problema — aprender com dados que outra política gerou — e as
diferenças entre eles são a razão de o repositório ter os três. Retrace estima `Q`, V-trace
estima `V`, GAE não corrige nada porque assume dados on-policy. `tests/test_lbc.py` amarra o
V-trace ao GAE no caso on-policy, e é assim que se sabe que ele está certo.

### 2.2 A família DQN, componente a componente

| Flag | Trabalho |
|---|---|
| `double` | van Hasselt, Guez & Silver, 2015 — *Deep Reinforcement Learning with Double Q-learning* · [arXiv:1509.06461](https://arxiv.org/abs/1509.06461) |
| `dueling` | Wang, Schaul, Hessel, van Hasselt, Lanctot & de Freitas, 2015 — *Dueling Network Architectures for Deep Reinforcement Learning* · [arXiv:1511.06581](https://arxiv.org/abs/1511.06581) |
| `per` | Schaul, Quan, Antonoglou & Silver, 2015 — *Prioritized Experience Replay* · [arXiv:1511.05952](https://arxiv.org/abs/1511.05952) |
| `c51` | Bellemare, Dabney & Munos, 2017 — *A Distributional Perspective on Reinforcement Learning* · [arXiv:1707.06887](https://arxiv.org/abs/1707.06887) |
| `noisy` | Fortunato et al., 2017 — *Noisy Networks for Exploration* · [arXiv:1706.10295](https://arxiv.org/abs/1706.10295) |

As cinco são ortogonais por construção em `snakeai/agents/dqn.py` — é isso que permite medir
cada uma isolada, e é a resposta aos seis notebooks quase idênticos do repositório antigo.
Ligar todas é o Rainbow.

### 2.3 Curvatura de segunda ordem

| Peça | Onde é usada | Trabalho |
|---|---|---|
| K-FAC | `snakeai/kfac.py`, camadas densas | Martens & Grosse, 2015 — *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* · [arXiv:1503.05671](https://arxiv.org/abs/1503.05671) |
| KFC | `snakeai/kfac.py`, convoluções | Grosse & Martens, 2016 — *A Kronecker-factored Approximate Fisher Matrix for Convolution Layers* · [arXiv:1602.01407](https://arxiv.org/abs/1602.01407) |
| EK-FAC | `snakeai/kfac.py`, classe `EKFac` | George et al., 2018 · [arXiv:1806.03884](https://arxiv.org/abs/1806.03884) |

A implementação canônica do Google (`tensorflow/kfac`) foi **arquivada em 19/04/2026** e usa
`tensorflow.compat.v1`: serviu de referência, não de dependência. A implementação de
referência do EK-FAC em PyTorch é [Thrandis/EKFAC-pytorch](https://github.com/Thrandis/EKFAC-pytorch);
a daqui usa o mesmo estimador exato **sem** o laço sobre o lote — ver `docs/EKFAC.md` §1.

### 2.4 Exploração e meta-controle

| Peça | Onde é usada | Trabalho |
|---|---|---|
| Agent57 | o antecessor que o LBC generaliza | Badia, Piot, Kapturowski, Sprechmann, Vitvitskyi, Guo & Blundell, 2020 — *Agent57: Outperforming the Atari Human Benchmark* · [arXiv:2003.13350](https://arxiv.org/abs/2003.13350) |
| UCB com janela deslizante | `snakeai/bandit.py`, o meta-controlador do LBC | Garivier & Moulines, 2008 — *On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems* · [arXiv:0805.3415](https://arxiv.org/abs/0805.3415). A versão ALT 2011, *…for Switching Bandit Problems*, é a citada pelo LBC · [doi:10.1007/978-3-642-24412-4_16](https://doi.org/10.1007/978-3-642-24412-4_16) |

### 2.5 Abstração temporal

| Peça | Onde é usada | Trabalho |
|---|---|---|
| Opções / semi-MDP | o arcabouço que o SOAP instancia | Sutton, Precup & Singh, 1999 — *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*, Artificial Intelligence 112(1–2) · [doi:10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1) |
| Option-Critic | a fatoração que o SOAP substitui | Bacon, Harb & Precup, 2016 — *The Option-Critic Architecture* · [arXiv:1609.05140](https://arxiv.org/abs/1609.05140) |

---

## 3. O que sustenta o ambiente e o protocolo

Estas não são "referências de contexto": cada uma corresponde a uma linha do contrato de
comparabilidade ou a uma decisão de projeto que muda um número publicado.

| Decisão | Trabalho |
|---|---|
| **Shaping potencial que decai a zero.** É o que garante que a política ótima do problema com shaping seja a mesma do problema real — sem essa garantia, o shaping do `VecSnake` mudaria o que o agente está otimizando | Ng, Harada & Russell, 1999 — *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML |
| **Score como métrica, não comprimento.** A convenção do ALE de reportar o retorno do jogo, que é o que torna as curvas comparáveis com a literatura | Bellemare, Naddaf, Veness & Bowling, 2013 — *The Arcade Learning Environment: An Evaluation Platform for General Agents*, JAIR 47 · [doi:10.1613/jair.3912](https://doi.org/10.1613/jair.3912) |
| **Os 37 detalhes do PPO.** A lista que decide se um PPO aprende ou vira ruído; o docstring de `snakeai/agents/ppo.py` diz quais estão implementados | Huang, Dossa, Raffin, Kanervisto & Wang, 2022 — *The 37 Implementation Details of Proximal Policy Optimization* · [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |

---

## 4. Arquitetura das redes

| Peça | Onde é usada | Trabalho |
|---|---|---|
| ResNet | os troncos `resnet_tiny`/`small`/`base` | He, Zhang, Ren & Sun, 2015 — *Deep Residual Learning for Image Recognition* · [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| GroupNorm | a normalização de todos os troncos residuais | Wu & He, 2018 — *Group Normalization* · [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) |

GroupNorm e não BatchNorm, e a razão é o RL: o lote de um rollout é altamente correlacionado
(512 ambientes no mesmo instante de treino), e a estatística de lote da BatchNorm passa a
depender de **quantos** ambientes rodam em paralelo — um parâmetro de execução vazando para
dentro do modelo.

---

## 5. Otimizadores de primeira ordem

O eixo do `99_ablacoes`, e o sucessor declarado do K-FAC como "eixo de otimizador" — ver
`snakeai/otimizadores.py`.

| `optimizer=` | Trabalho |
|---|---|
| `adam` | Kingma & Ba, 2014 — *Adam: A Method for Stochastic Optimization* · [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| `adamw` | Loshchilov & Hutter, 2017 — *Decoupled Weight Decay Regularization* · [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| `lion` | Chen et al., 2023 — *Symbolic Discovery of Optimization Algorithms* · [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) |
| `rmsprop` | Tieleman & Hinton, 2012 — *Lecture 6.5: RMSProp*, COURSERA: Neural Networks for Machine Learning (sem publicação formal) |
| `sgd` | com momento de Nesterov, conforme Sutskever, Martens, Dahl & Hinton, 2013 — *On the importance of initialization and momentum in deep learning*, ICML |

---

## 6. Código de terceiros

Não são papers, e por isso ficam separados: são implementações que serviram de referência ou
de ponto de partida.

| Origem | O que veio de lá |
|---|---|
| [`farizrahman4u/qlearning4k`](https://github.com/farizrahman4u/qlearning4k) | o código de Snake que virou a base original do jogo |
| [`chuyangliu/snake`](https://github.com/chuyangliu/snake) | a ideia das ações relativas, que colapsa as simetrias de rotação |
| [`Kaixhin/Rainbow`](https://github.com/Kaixhin/Rainbow) | a `CNN3`, portada de `colab-rl/models/utilities/networks.py` |
| [`Thrandis/EKFAC-pytorch`](https://github.com/Thrandis/EKFAC-pytorch) | referência da implementação do EK-FAC |
| `tensorflow/kfac` (arquivado) | referência da implementação do K-FAC |
| [`voaneves/colab-rl`](https://github.com/voaneves/colab-rl) | os 13 notebooks de origem, preservados em `legacy/` |
| [`voaneves/snake-on-pygame`](https://github.com/voaneves/snake-on-pygame) | o jogo jogável por humanos e o leaderboard |

---

## 7. Onde cada paper vira teste

O repositório não confia em "implementei conforme o paper" — cada afirmação central tem um
teste que falha se ela deixar de valer. A lista completa está nos módulos de teste; estas são
as que valem por todas:

| Afirmação | Teste |
|---|---|
| o K-FAC pré-condiciona de verdade: com a Fisher exata, **um** passo de tamanho 1 aterrissa no ótimo de mínimos quadrados — o que nenhum método de primeira ordem faz, com learning rate nenhum | `test_kfac.py::test_one_kfac_step_with_the_exact_fisher_lands_on_the_optimum` |
| o EK-FAC nunca é pior que o K-FAC em norma de Frobenius (Teorema 3) | `test_ekfac.py::test_ekfac_is_never_worse_than_kfac_in_frobenius_norm` |
| o V-trace on-policy é exatamente o GAE(λ=1) do PPO | `test_lbc.py::test_on_policy_vtrace_is_gae_with_lambda_one` |
| o SOAP com uma opção **é** o PPO | `test_soap.py::test_with_one_option_the_option_gae_is_the_ppo_gae` |
| o ACEKTR sem medir é o ACKTR bit a bit | `test_ekfac.py::test_without_measuring_ekfac_is_bit_for_bit_kfac` |
| o ACKTR é o A2C mais K-FAC e nada mais | `test_acktr.py::test_acktr_is_a2c_plus_kfac_and_nothing_else` |
| a Fisher usa a ação **amostrada**, não a tomada — a diferença entre Fisher e Fisher empírico | `test_kfac.py::test_categorical_fisher_uses_a_sampled_action_not_the_taken_one` |

---

## 8. Como citar este repositório

Ainda não há artigo. Enquanto não houver, o repositório se cita pelo commit — e a
**assinatura do pacote** de cada execução (`meta["assinatura_pacote"]` no `history.json`) é o
que amarra um número a um código específico. Ver `docs/PROCEDENCIA.md`.
