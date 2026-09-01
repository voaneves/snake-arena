"""Bandits como meta-controladores — quem escolhe o comportamento, e com que critério.

Todo algoritmo deste repositório resolve a exploração com uma **regra fixa**: o ε do DQN
decai numa reta, o coeficiente de entropia do PPO decai noutra, o ruído da `NoisyDense`
encolhe sozinho. Nenhuma dessas regras olha para o resultado — o ε de 0,3 no passo 1 M é
0,3 porque o agendamento diz, não porque explorar tanto ali esteja rendendo alguma coisa.

Um bandit inverte isso: cada configuração de comportamento vira um **braço**, o retorno do
episódio jogado com ela vira a recompensa do braço, e a escolha passa a ser aprendida. É a
peça que o Agent57 introduziu e que o LBC generaliza — e é a única parte deste repositório
em que a exploração é medida em vez de agendada.

Por que a janela deslizante, e não a média de sempre
----------------------------------------------------
O problema **não é estacionário**, e por um motivo que não é sutil: a política muda embaixo
do bandit. Um braço muito exploratório é ótimo nos primeiros 500 mil passos e péssimo no
fim; um braço quase guloso é o contrário. Um UCB clássico, que faz média sobre todas as
puxadas desde o início, leva centenas de milhares de passos para esquecer que o braço
exploratório já foi bom — e nesse intervalo ele continua sendo escolhido.

A correção padrão é a de Garivier & Moulines (2011): média sobre uma **janela dos retornos
recentes**, e a contagem do bônus de exploração também dentro da janela. Um braço que era
bom e deixou de ser cai em `janela` puxadas, não em ninguém sabe quantas.

Por que os valores são normalizados antes do bônus
--------------------------------------------------
O score do UCB soma duas coisas de naturezas diferentes: um valor (aqui, o score do
episódio de Snake, que vai de 0 a 97) e um bônus de exploração da ordem de `c·√log t`, que
raramente passa de 3. Somar os dois crus faz o `c` deixar de significar nada — no começo do
treino, com todos os braços perto de 1 ponto, o bônus domina e a escolha é aleatória; no
fim, com braços em 60 e 80 pontos, o bônus é ruído e a escolha é gulosa. O mesmo `c`, dois
regimes opostos, sem que ninguém tenha mexido nele.

Normalizar os valores para `[0, 1]` pelo mínimo e máximo **entre os braços** resolve: o `c`
passa a ser lido como "quanto de incerteza vale um intervalo inteiro de desempenho entre o
pior e o melhor braço", que é uma grandeza estável do começo ao fim. É um desvio do paper
do LBC, que usa o retorno cru — lá as escalas do Atari variam por ordens de grandeza entre
jogos e o mesmo problema existe, só que resolvido jogo a jogo pelo `c` da população de
bandits. Ver `docs/LBC.md`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["BanditUCB"]


class BanditUCB:
    """UCB não-estacionário com janela deslizante, sobre `n_bracos` braços.

    Parâmetros
    ----------
    n_bracos : int
        Quantos braços. No LBC, quantas regiões o espaço de comportamento foi dividido.
    c : float
        Peso do bônus de exploração. Com os valores normalizados para `[0, 1]` (o padrão),
        `c = 1` significa "uma incerteza máxima vale tanto quanto ser o melhor braço".
    janela : int
        Quantos retornos recentes cada braço guarda. Curta demais e o valor é ruído;
        longa demais e o bandit demora a perceber que o comportamento envelheceu.
    temperatura : float
        Temperatura da softmax que transforma o score em `P_Ψ`. Ela existe **por causa**
        da normalização: com os scores comprimidos em `[0, 1]`, uma softmax de temperatura
        1 nunca dá a um braço mais que ~2,7 vezes a probabilidade de outro, e o bandit
        fica preso perto do uniforme por construção — decidir deixaria de ser possível.
        Baixa demais e a seleção vira `argmax`, que é o outro extremo: todos os atores no
        mesmo braço e nenhuma evidência nova sobre os outros. 0,1 dá ~7× de vantagem para
        uma diferença de 20% do intervalo entre o pior e o melhor braço.
    normalizar : bool
        Normaliza os valores para `[0, 1]` antes de somar o bônus. Ver o docstring do
        módulo — desligar reproduz a fórmula crua do paper.
    rng : np.random.Generator, opcional
        Gerador próprio, para reprodutibilidade.
    """

    def __init__(self, n_bracos, c=1.0, janela=64, temperatura=0.25, normalizar=True,
                 min_puxadas=8, piso_uniforme=0.1, rng=None):
        if int(n_bracos) < 1:
            raise ValueError("um bandit precisa de pelo menos um braço")
        if not 0.0 <= float(piso_uniforme) < 1.0:
            raise ValueError("piso_uniforme tem que estar em [0, 1)")
        self.n = int(n_bracos)
        self.c = float(c)
        self.janela = int(janela)
        self.temperatura = float(temperatura)
        self.normalizar = bool(normalizar)
        self.min_puxadas = int(min_puxadas)
        self.piso_uniforme = float(piso_uniforme)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._retornos = [deque(maxlen=self.janela) for _ in range(self.n)]
        #: Puxadas desde sempre. Não entra no bônus — serve só para o relatório, porque é
        #: o número que diz se o bandit convergiu para um braço ou continua varrendo.
        self.puxadas_totais = np.zeros(self.n, dtype=np.int64)

    def __len__(self):
        return self.n

    # ------------------------------------------------------------------ leitura
    def visitas(self):
        """Puxadas **dentro da janela** — é esta a contagem que o bônus usa."""
        return np.array([len(d) for d in self._retornos], dtype=np.float64)

    def valores(self):
        """Retorno médio recente por braço. `NaN` onde a janela está vazia.

        `NaN` e não zero: um braço nunca puxado não tem valor estimado, e escrever zero
        ali seria afirmar que ele é ruim — exatamente o contrário do que a ausência de
        dados autoriza.
        """
        return np.array([np.mean(d) if len(d) >= self.min_puxadas else np.nan
                         for d in self._retornos], dtype=np.float64)

    def score(self):
        """`V_k + c·√(log(1 + Σ_{j≠k} N_j) / (1 + N_k))` — o score do LBC (eq. do §4.2).

        O somatório exclui o próprio braço de propósito: o bônus cresce com o que os
        **outros** braços foram puxados, que é a formulação de Garivier & Moulines para o
        caso comutante. Um braço puxado o tempo todo não infla o próprio bônus.
        """
        v = self.valores()
        n = self.visitas()
        total = n.sum()

        if self.normalizar:
            v = self._normaliza(v, self._ruido())
        # Otimismo diante da incerteza: um braço sem visita entra com o melhor valor
        # observado, não com a média. Sem isso, um braço que ninguém puxou compete com
        # valor `NaN` e some do argmax mesmo com o bônus máximo.
        finitos = v[np.isfinite(v)]
        v = np.where(np.isfinite(v), v, finitos.max() if finitos.size else 0.0)

        bonus = self.c * np.sqrt(np.log(1.0 + (total - n)) / (1.0 + n))
        return v + bonus

    def distribuicao(self):
        """`softmax(score)` — a distribuição de seleção `P_Ψ` do paper.

        Softmax e não `argmax`: com vários atores escolhendo ao mesmo tempo, o argmax
        colocaria todos no mesmo braço e o bandit deixaria de receber dados sobre os
        outros. A cauda da softmax é o que mantém a estimativa dos braços perdedores viva
        o suficiente para que uma virada seja percebida — e é ela, não o bônus do UCB, que
        garante que todo braço continue sendo medido de vez em quando.

        `temperatura` decide a dureza dessa softmax. Ver o parâmetro no construtor: com os
        valores normalizados, sem ela a distribuição não conseguiria concentrar.
        """
        s = self.score() / max(self.temperatura, 1e-12)
        e = np.exp(s - s.max())
        p = e / e.sum()
        if self.piso_uniforme > 0.0:
            p = (1.0 - self.piso_uniforme) * p + self.piso_uniforme / self.n
        return p

    # ------------------------------------------------------------------ escolha
    def amostrar(self, tamanho=None):
        """Sorteia braços de `P_Ψ`. `tamanho=None` devolve um inteiro."""
        p = self.distribuicao()
        return self.rng.choice(self.n, size=tamanho, p=p)

    # -------------------------------------------------------------- atualização
    def registrar(self, braco, retorno):
        """Guarda o retorno de um episódio jogado com o braço `braco`."""
        b = int(braco)
        self._retornos[b].append(float(retorno))
        self.puxadas_totais[b] += 1

    def registrar_lote(self, bracos, retornos):
        for b, r in zip(np.asarray(bracos).ravel(), np.asarray(retornos).ravel()):
            self.registrar(b, r)

    # ---------------------------------------------------------------- relatório
    def resumo(self):
        """Números para o registro da execução — o que responde "o bandit está fazendo algo?".

        `entropia` é a da distribuição de seleção, em nats: perto de `log K` o bandit
        ainda está varrendo, perto de zero ele decidiu. Uma execução em que ela nunca sai
        de `log K` é uma execução em que a seleção não aprendeu nada, e a curva deveria
        ser comparável à da ablação de seleção aleatória — que é justamente a ablação da
        Fig. 5 do paper.
        """
        p = self.distribuicao()
        v = self.valores()
        finitos = np.isfinite(v)
        return {
            "mab_entropia": float(-(p * np.log(p + 1e-12)).sum()),
            "mab_entropia_max": float(np.log(self.n)),
            "mab_braco_top": int(np.argmax(p)),
            "mab_p_top": float(p.max()),
            "mab_valor_top": float(v[np.argmax(p)]) if finitos[np.argmax(p)] else float("nan"),
            "mab_bracos_visitados": int(finitos.sum()),
            #: Diferença entre o melhor e o pior braço em unidades de erro padrão. Abaixo
            #: de ~2 o bandit não tem evidência para separar braço nenhum, e a seleção
            #: **deve** estar perto do uniforme — se `mab_entropia` estiver baixa com este
            #: número baixo, o meta-controlador travou em ruído.
            "mab_sinal_ruido": float(
                (np.nanmax(v) - np.nanmin(v)) / self._ruido()) if (
                    finitos.sum() >= 2 and self._ruido() > 0) else float("nan"),
        }

    def _ruido(self):
        """Erro padrão **combinado** das médias por braço — a régua do que é diferença real.

        A normalização min–max é cega à incerteza: ela estica a distância entre o pior e o
        melhor braço para o intervalo inteiro `[0, 1]` *sempre*, mesmo quando essa
        distância é menor que o próprio erro amostral. No começo do treino, com todos os
        braços rendendo ~0,02 ponto, o que separa o "melhor" do "pior" é ruído puro — e a
        normalização o promove a um sinal de amplitude máxima, que a temperatura de 0,1
        então transforma em quase-`argmax`. Foi assim que a execução `seed0` travou em
        `mab_p_top = 0,999` no passo 800 mil, com 512 ambientes no mesmo braço e nenhuma
        evidência nova sobre os outros quinze.

        Este número é o denominador mínimo: enquanto a diferença entre os braços não passar
        de ~2 erros padrão, os valores normalizados ficam pequenos e o bônus do UCB domina.
        O bandit só decide quando tem motivo para decidir.
        """
        sems = [np.std(d, ddof=1) / np.sqrt(len(d))
                for d in self._retornos if len(d) >= 2]
        return float(np.mean(sems)) if sems else 0.0

    @staticmethod
    def _normaliza(v, ruido=0.0):
        """Min–max **entre os braços**, ignorando os não visitados.

        Entre os braços, e não contra uma escala fixa, porque a escala do retorno muda
        durante o treino inteiro: no passo 100 mil todo braço rende ~1 ponto, no passo
        5 M os bons rendem 80. O que interessa ao bandit é a ordem e a distância
        *relativa*, e essas duas sobrevivem à normalização.
        """
        finitos = v[np.isfinite(v)]
        if finitos.size == 0:
            return np.zeros_like(v)
        lo, hi = finitos.min(), finitos.max()
        # o denominador nunca encolhe abaixo de ~2 erros padrão: sem esse piso, uma
        # diferença de ruído entre os braços vira um sinal de amplitude 1 — ver `_ruido`
        escala = max(hi - lo, 2.0 * float(ruido))
        if escala < 1e-12:
            return np.where(np.isfinite(v), 0.0, np.nan)
        return (v - lo) / escala
