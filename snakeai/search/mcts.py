"""MCTS com PUCT sobre o simulador **real**.

Por que isto é a jogada certa em Snake
---------------------------------------
MuZero e EfficientZero gastam a maior parte da complexidade deles aprendendo um modelo do
mundo — porque em Atari o simulador não está disponível durante a busca. Aqui está: Snake é
determinístico, de informação perfeita, tem 3 ações, e o `VecSnake` faz ~286 mil passos por
segundo. Aprender um modelo do que já se pode simular exatamente seria pagar caro por uma
aproximação pior.

Então este módulo faz busca em árvore com o jogo de verdade. É o AlphaZero sem a parte de
adivinhar a física.

Como a busca fica em lote
-------------------------
MCTS é naturalmente sequencial, e uma avaliação de rede por simulação com lote 1 seria
lentíssimo na GPU. O truque: rodar **N árvores independentes em paralelo**, uma por
ambiente, e sincronizá-las por número de simulação. Na simulação `k`, as N árvores estão
todas esperando avaliar um nó — e aí a rede recebe um lote de N. A busca custa
`num_simulations` chamadas de rede, não `N × num_simulations`.

O estado de cada nó é o dicionário de `VecSnake.get_state()`. Restaurar um nó é escrever
esses arrays de volta num ambiente de busca descartável — barato, e exato.
"""

from __future__ import annotations

import numpy as np

from ..env.vec_snake import N_ACTIONS
from .dinamica import DinamicaReal

__all__ = ["No", "MCTS"]


class No:
    """Um nó da árvore. Guarda o estado do jogo e as estatísticas do PUCT."""

    __slots__ = ("estado", "prior", "visitas", "soma_valor", "filhos", "recompensa",
                 "terminal", "mask", "expandido")

    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.visitas = 0
        self.soma_valor = 0.0
        self.filhos = {}
        self.estado = None
        self.recompensa = 0.0
        self.terminal = False
        self.mask = None
        self.expandido = False

    @property
    def valor(self):
        return self.soma_valor / self.visitas if self.visitas else 0.0


class MCTS:
    """Busca em árvore com PUCT, em lote sobre N árvores.

    Parâmetros
    ----------
    avaliar : callable
        `avaliar(obs, mask) -> (priors, valores)`, com `priors` já normalizado e
        mascarado. É a rede; o MCTS não conhece Keras.
    board_size, gamma : contrato do ambiente.
    num_simulations : int
        Orçamento de busca por jogada. É o botão que troca computação por qualidade.
    c_puct : float
        Peso da exploração no PUCT. Maior = confia mais no prior, explora mais largo.
    dirichlet_alpha, dirichlet_frac : float
        Ruído na raiz, só durante a coleta. Sem ele a busca fica determinística e o agente
        nunca descobre uma jogada que a rede ainda não gosta — é o análogo do ε-greedy.
    """

    def __init__(self, avaliar, board_size=10, gamma=0.997, num_simulations=32,
                 c_puct=1.5, dirichlet_alpha=0.5, dirichlet_frac=0.25, rng=None,
                 starve_base=None, dinamica=None):
        self.avaliar = avaliar
        self.board_size = int(board_size)
        #: O ambiente de busca TEM que ser configurado igual ao de treino. Se o
        #: `starve_base` diferir, a árvore simula um jogo com outra regra de fome — e
        #: planeja sobre um mundo que não é o que o agente vai jogar. Não levanta erro:
        #: só produz decisões ligeiramente erradas, o tempo todo.
        self.starve_base = starve_base
        #: O que a árvore percorre. Trocar isto — e só isto — é a diferença entre
        #: AlphaZero e MuZero neste repositório.
        self.dinamica = dinamica or DinamicaReal(board_size, starve_base)
        self.gamma = float(gamma)
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._sim = None      # ambiente de busca, criado sob demanda no tamanho certo
        self._ultimas_raizes = []

    # ---------------------------------------------------------------- ambiente
    def _ambiente(self, n):
        """Compatibilidade: expõe o `VecSnake` da dinâmica real, quando houver."""
        return self.dinamica._ambiente(n)

    def _expandir(self, nos, priors, valores):
        for no, p in zip(nos, priors):
            if no is None:
                continue
            no.expandido = True
            permitidas = np.nonzero(no.mask)[0] if no.mask is not None else range(N_ACTIONS)
            for a in permitidas:
                no.filhos[int(a)] = No(prior=float(p[a]))

    def _selecionar(self, no):
        """PUCT: `Q(s,a) + c · P · √N / (1 + n)`, com `Q(s,a) = r + γ·V(filho)`.

        A recompensa de **chegar** ao filho tem que entrar no Q — e é fácil esquecer,
        porque o nó guarda o valor do estado dele, não do movimento. Sem esse termo, um
        filho alcançado morrendo tem valor 0 (episódio acabou, sem futuro) e parece tão
        atraente quanto um filho seguro: a busca escolhe a morte e fica **pior que
        aleatória**. Foi exatamente o que aconteceu na primeira versão deste arquivo.
        """
        if not no.filhos:
            return None
        raiz_n = np.sqrt(max(no.visitas, 1))
        melhor, melhor_pont = None, -np.inf
        for a, filho in no.filhos.items():
            u = self.c_puct * filho.prior * raiz_n / (1 + filho.visitas)
            q = (filho.recompensa + self.gamma * filho.valor) if filho.visitas else 0.0
            pont = q + u
            if pont > melhor_pont:
                melhor, melhor_pont = a, pont
        return melhor

    # -------------------------------------------------------------------- busca
    def run(self, estado_raiz, mask_raiz, obs_raiz, adicionar_ruido=False):
        """Roda a busca a partir de N estados. Devolve `(visitas, valores_raiz)`.

        `visitas` é `(N, 3)` — a contagem de visitas por ação, que é a política melhorada
        pela busca. `valores_raiz` é `(N,)`, o valor que a busca atribuiu à posição.
        """
        n = mask_raiz.shape[0]
        din = self.dinamica

        raizes = [No() for _ in range(n)]
        for i, r in enumerate(raizes):
            r.estado = din.fatiar(estado_raiz, i)
            r.mask = mask_raiz[i]

        priors, valores = self.avaliar(obs_raiz, mask_raiz)
        priors = np.asarray(priors, dtype=np.float64)
        if adicionar_ruido:
            ruido = self.rng.dirichlet([self.dirichlet_alpha] * N_ACTIONS, size=n)
            priors = (1 - self.dirichlet_frac) * priors + self.dirichlet_frac * ruido
            priors = np.where(mask_raiz, priors, 0.0)
            priors /= np.maximum(priors.sum(1, keepdims=True), 1e-12)
        self._expandir(raizes, priors, valores)
        for r, v in zip(raizes, np.asarray(valores).ravel()):
            r.visitas, r.soma_valor = 1, float(v)

        for _ in range(self.num_simulations):
            caminhos = []          # (lista de nós, ação escolhida) por árvore
            pais, acoes = [], []
            for i, raiz in enumerate(raizes):
                no, caminho = raiz, [raiz]
                a = self._selecionar(no)
                while a is not None and no.filhos[a].expandido and not no.filhos[a].terminal:
                    no = no.filhos[a]
                    caminho.append(no)
                    a = self._selecionar(no)
                caminhos.append((caminho, a))
                pais.append(no)
                acoes.append(a if a is not None else 1)

            # --- um passo da dinâmica, em lote, a partir dos N pais
            estados = din.empilhar([p.estado for p in pais])
            estados_filhos, obs, mask, rew, done = din.passo(estados, acoes)

            folhas, a_avaliar_obs, a_avaliar_mask, idx_avaliar = [], [], [], []
            for i, ((caminho, a), pai) in enumerate(zip(caminhos, pais)):
                if a is None:
                    folhas.append(None)
                    continue
                filho = pai.filhos[a]
                filho.recompensa = float(rew[i])
                filho.terminal = bool(done[i])
                folhas.append(filho)
                if filho.terminal:
                    # ARMADILHA: o `VecSnake` reseta sozinho ao terminar, então `obs[i]` e
                    # `estados_filhos[i]` já são de um episódio NOVO. Guardá-los aqui
                    # plantaria uma partida aleatória dentro da árvore. Um nó terminal não
                    # precisa de estado — vale 0 e nunca é expandido.
                    continue
                filho.estado = din.fatiar(estados_filhos, i)
                filho.mask = mask[i] if din.usa_mascara else None
                a_avaliar_obs.append(obs[i])
                a_avaliar_mask.append(mask[i])
                idx_avaliar.append(i)

            valores_folha = np.zeros(n, dtype=np.float64)
            if idx_avaliar:
                p_f, v_f = self.avaliar(np.stack(a_avaliar_obs), np.stack(a_avaliar_mask))
                p_f = np.asarray(p_f, dtype=np.float64)
                v_f = np.asarray(v_f, dtype=np.float64).ravel()
                self._expandir([folhas[i] for i in idx_avaliar], p_f, v_f)
                for k, i in enumerate(idx_avaliar):
                    valores_folha[i] = v_f[k]

            # --- backup
            # `v` é sempre o valor estimado DO nó que está recebendo o crédito. Subindo,
            # ele vira `recompensa_de_entrar_no_nó + γ·v`. Nó terminal vale 0: o episódio
            # acabou, não há retorno futuro nenhum para descontar.
            for i, (caminho, a) in enumerate(caminhos):
                if a is None:
                    continue
                folha = folhas[i]
                v = 0.0 if folha.terminal else float(valores_folha[i])
                for no in reversed([*caminho, folha]):
                    no.visitas += 1
                    no.soma_valor += v
                    v = no.recompensa + self.gamma * v

        #: guardado para inspeção em teste — a árvore some ao fim do `run`
        self._ultimas_raizes = raizes

        visitas = np.zeros((n, N_ACTIONS), dtype=np.float64)
        for i, raiz in enumerate(raizes):
            for a, filho in raiz.filhos.items():
                visitas[i, a] = filho.visitas
        valores_raiz = np.array([r.valor for r in raizes], dtype=np.float32)
        return visitas, valores_raiz

    @staticmethod
    def politica_das_visitas(visitas, temperatura=1.0):
        """Converte contagens de visita em distribuição.

        `temperatura → 0` vira argmax (jogo forte); `1` mantém a proporção das visitas
        (bom para explorar e para o alvo de treino).
        """
        if temperatura <= 1e-6:
            p = (visitas == visitas.max(axis=1, keepdims=True)).astype(np.float64)
        else:
            p = np.power(np.maximum(visitas, 0), 1.0 / temperatura)
        soma = p.sum(axis=1, keepdims=True)
        return np.where(soma > 0, p / np.maximum(soma, 1e-12), 1.0 / visitas.shape[1])
