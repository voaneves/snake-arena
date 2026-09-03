"""Andaime comum a todos os agentes.

O que fica aqui é o que **precisa** ser idêntico entre algoritmos para que a comparação
valha: a cadência da avaliação, o formato do registro, o critério de "melhor checkpoint",
e os agendamentos lineares. O que varia — como o agente aprende — fica em cada módulo.

Foi essa separação que faltou no repositório antigo: cada notebook tinha o próprio laço de
treino, a própria noção de época e o próprio jeito de avaliar, e por isso as curvas nunca
puderam ser sobrepostas.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from ..env.vec_snake import VecSnake
from ..eval import evaluate, keras_policy, random_baseline
from ..plataforma import resumo_plataforma
from ..record import CONTRATO, Recorder, validate

__all__ = ["BaseConfig", "AgentBase"]


@dataclass
class BaseConfig:
    """Os campos que todo agente do benchmark tem. Cada algoritmo estende com os seus."""

    board_size: int = CONTRATO["board_size"]
    net: str = "resnet_small"
    seed: int = 0

    #: Orçamento oficial. O contrato exige o **mesmo** valor para todos os algoritmos.
    total_steps: int = 5_000_000

    #: Avaliação periódica durante o treino, no protocolo oficial.
    eval_every_steps: int = 250_000
    eval_episodes: int = CONTRATO["eval_episodes"]
    eval_envs: int = 250

    ckpt_dir: str = "checkpoints"
    runs_dir: str = "runs"
    log_every_steps: int = 50_000

    #: Artefatos gerados no fim do treino. O GIF custa segundos e responde a pergunta que
    #: nenhuma curva responde: *como* o agente joga.
    salvar_grafico: bool = True
    #: Sufixo acrescentado à variante. Serve para que uma execução que muda
    #: hiperparâmetros — e portanto **compete**, mas não é a mesma coisa — não divida a
    #: identidade `(algo, variant, seed)` com a do padrão. `load_all` agrupa por essa
    #: tripla, então identidade repetida vira uma curva só, com as duas misturadas.
    sufixo_variante: str = ""

    salvar_gif: bool = True
    gif_seeds: tuple = (7, 21, 42)

    #: Marque `False` numa execução que muda o ambiente ou o protocolo de propósito — uma
    #: ablação. Ela continua sendo gravada e plotada, mas **fora da arena**, e `caveat`
    #: passa a ser obrigatório: uma curva incomparável sem o motivo escrito é pior que
    #: nenhuma curva, porque alguém vai compará-la mesmo assim.
    comparable: bool = True
    caveat: str = ""

    def __post_init__(self):
        if not self.comparable and not self.caveat:
            raise ValueError(
                "comparable=False exige `caveat` dizendo por que esta execução não "
                "compete. Sem isso a curva vira uma armadilha para quem ler depois.")
        if self.board_size != CONTRATO["board_size"]:
            raise ValueError(
                f"board_size={self.board_size} viola o contrato "
                f"({CONTRATO['board_size']}). Mude o contrato conscientemente, "
                "não a execução."
            )


def proximo_multiplo(passo, cadencia):
    """O menor múltiplo de `cadencia` **estritamente acima** de `passo`.

    É a diferença entre uma grade absoluta e uma que reancora: `passo + cadencia` faz cada
    avaliação cair um bloco depois da anterior, e o desvio se acumula — na execução padrão
    do PPO a última avaliação aconteceu 513 mil passos além do ponto nominal, 10% do
    orçamento. Como cada algoritmo avança em blocos de tamanho diferente (8.192 no A2C,
    49.152 no PPO padrão), as grades divergem entre si e a coluna `passos até 40`, que o
    contrato lê **sem interpolação**, passa a comparar medições feitas em lugares
    diferentes. Ver `docs/REVISAO_ALGORITMOS.md` §1.7.
    """
    return (int(passo) // int(cadencia) + 1) * int(cadencia)


class AgentBase:
    """Laço de treino comum: agendamentos, avaliação, checkpoint e registro.

    A subclasse implementa `iterate()` — um passo de aprendizado, que devolve estatísticas
    do rollout — e o resto vem de graça, igual para todo mundo.
    """

    algo = "base"

    #: Tamanho da janela da média móvel do treino, **em episódios**. 500 é da mesma ordem
    #: dos 1.000 da avaliação oficial: grande o bastante para o número não pular com um
    #: episódio de sorte, pequeno o bastante para acompanhar o agente melhorando.
    JANELA_EPISODIOS = 500

    #: Sufixo da variante para execuções fora do contrato de observação. A identidade de
    #: uma execução é `(algo, variant, seed)` — é por ela que `load_all` agrupa, não pelo
    #: caminho. Sem o sufixo, uma execução com `canal_fome=True` fica com a **mesma
    #: identidade** da execução de contrato da mesma rede e semente: hoje elas só não se
    #: misturam porque `comparable=False` as tira da arena, o que é proteção por acidente,
    #: não por construção. Ver `docs/CANAL_DE_FOME.md`.
    SUFIXO_FOME = "_fome"

    def __init__(self, cfg, variant="default"):
        self.cfg = cfg
        if getattr(cfg, "canal_fome", False):
            variant = self._com_sufixo(variant, self.SUFIXO_FOME)
        self.variant = self._com_sufixo(variant, getattr(cfg, "sufixo_variante", ""))
        self.model = None
        self.global_step = 0
        self.episodes = 0
        self.iteration = 0
        self.history = []
        self.evals = []
        self.baseline = None
        self.melhor = -np.inf
        self._proximo_eval = 0
        self._proximo_log = 0
        self._atualizacoes = 0
        #: Janela de episódios recentes para a média móvel do treino. Sem ela, o log
        #: imprime a média dos episódios que por acaso terminaram **naquela** iteração —
        #: uma amostra de tamanho 0 a 3. É o que produzia a sequência
        #: `2,50 · 10,00 · — · — · 2,00 · 11,00`, que parece instabilidade do algoritmo e
        #: é só tamanho de amostra. O `—` é literalmente "nenhum episódio acabou agora".
        #:
        #: A janela é medida em **episódios**, não em iterações, e a diferença não é
        #: cosmética. Uma iteração de PPO são 512 × 96 = 49.152 passos e ~200 episódios;
        #: uma de DQN são ~1.000 passos e 2 ou 3 episódios. Um limite fixo de iterações
        #: cobriria a execução inteira num caso e alguns segundos no outro — e no primeiro
        #: a "média móvel" viraria **média acumulada**, arrastada para baixo pelos
        #: episódios ruins do começo para sempre.
        self._janela = deque()
        #: Totais acumulados desde o início e no instante do log anterior. A janela móvel
        #: descarta pela esquerda, então a diferença entre dois logs **não** dá para ser
        #: reconstruída dela — e é essa diferença que responde "o que aconteceu nos
        #: últimos N episódios", que a média móvel de 500 esconde por construção. Numa
        #: iteração que produz ~80 episódios por log, a móvel arrasta 6 logs de história:
        #: uma degradação em curso aparece nela achatada e com atraso.
        self._acumulado = {"n": 0.0, **{c: 0.0 for c in self.CAMPOS_JANELA}}
        self._acumulado_no_log = dict(self._acumulado)
        self._legenda_impressa = False
        self._registrou_causas = False
        #: O `Recorder` da execução em curso, para o `salvar()` conseguir gravar a curva
        #: junto do modelo. Sem ele, retomar devolve os pesos e perde o registro.
        self._rec = None
        self._curva_retomada, self._wall_retomado = [], 0.0
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------- agendamentos
    #: Os campos somados na janela. Todos são **contagens ou somas** por bloco, nunca
    #: médias: só assim a média da janela pode ser ponderada pelo número de episódios de
    #: cada bloco, que é o que impede uma iteração com 2 episódios de pesar igual a uma
    #: com 200.
    CAMPOS_JANELA = ("score", "vitorias", "fome", "colisao", "passos")

    def _registra_episodios(self, media, n, **somas):
        """Guarda `n` episódios de score médio `media` e descarta o que saiu da janela.

        Descarta pela esquerda enquanto o que sobra ainda cobre `JANELA_EPISODIOS`, e
        nunca esvazia: com um algoritmo cuja iteração já produz mais episódios que a
        janela inteira, o certo é a janela ser aquela iteração — e não ficar vazia.
        """
        bloco = {"n": n, "score": media * n}
        bloco.update({c: float(somas.get(c, 0.0)) for c in self.CAMPOS_JANELA
                      if c != "score"})
        self._janela.append(bloco)
        self._acumulado["n"] += n
        for c in self.CAMPOS_JANELA:
            self._acumulado[c] += bloco[c]
        total = sum(b["n"] for b in self._janela)
        while len(self._janela) > 1 and total - self._janela[0]["n"] >= self.JANELA_EPISODIOS:
            total -= self._janela.popleft()["n"]

    def registra_fim(self, info):
        """Contabiliza os episódios que acabaram neste passo, **por causa**.

        Chamado de dentro do laço de coleta de cada agente, com o `info` que `VecSnake.step`
        devolve. É o único lugar onde a causa da morte existe: a memória de treino guarda
        `cont = 0` e nada mais, e depois do reset não há como saber se a cobra bateu ou
        passou fome.

        E a diferença entre as duas é justamente o que o score sozinho esconde. Um agente
        preso em 1,2 pontos pode estar batendo em tudo (não aprendeu a sobreviver) ou
        andando em círculo até morrer de fome (aprendeu a sobreviver e não a comer) — dois
        problemas opostos, com o mesmo número na curva. Foi exatamente essa ambiguidade que
        custou horas no diagnóstico do Dreamer.
        """
        n = int(info["scores"].size)
        if not n:
            return
        self._registrou_causas = True
        self._registra_episodios(
            float(info["scores"].mean()), n,
            vitorias=info["wins"],
            fome=info["starved"],
            # `deaths` conta colisão; vitória e fome não entram nele
            colisao=info["deaths"],
            passos=float(info["lengths"].sum()),
        )

    def media_movel(self):
        """Score médio dos últimos ~`JANELA_EPISODIOS` episódios, ponderado.

        `None` só quando nenhum episódio terminou ainda.
        """
        n = self.episodios_na_janela()
        return sum(b["score"] for b in self._janela) / n if n else None

    def episodios_na_janela(self):
        return sum(b["n"] for b in self._janela)

    def resumo_janela(self):
        """As frações que o score sozinho não conta, sobre a mesma janela de episódios.

        `{}` enquanto nenhum episódio terminou. As causas só aparecem se o agente chamar
        `registra_fim` — quem ainda não chama continua reportando só o score, sem inventar
        um zero que pareceria "nunca morre de fome".
        """
        n = self.episodios_na_janela()
        if not n:
            return {}
        soma = {c: sum(b.get(c, 0.0) for b in self._janela) for c in self.CAMPOS_JANELA}
        r = {"janela_episodios": n, "train_score_mean": soma["score"] / n}
        if self._registrou_causas:
            r.update({
                "win_rate": soma["vitorias"] / n,
                "frac_fome": soma["fome"] / n,
                "frac_colisao": soma["colisao"] / n,
                "passos_por_episodio": soma["passos"] / n,
            })
        return r

    def resumo_bloco(self):
        """As mesmas médias de `resumo_janela`, mas só sobre os episódios **novos**.

        "Novos" = terminados desde o log anterior. É o número que mostra a direção: a
        média móvel de 500 episódios responde "onde o agente está", o bloco responde "para
        onde está indo", e num treino que degrada as duas discordam por muito tempo antes
        de a móvel virar. Prefixadas com `bloco_` no registro para não colidirem com as
        chaves da janela, que são as que a arena e as curvas leem.

        `{}` quando nenhum episódio terminou desde o log anterior.
        """
        n = self._acumulado["n"] - self._acumulado_no_log["n"]
        if n <= 0:
            return {}
        d = {c: self._acumulado[c] - self._acumulado_no_log[c] for c in self.CAMPOS_JANELA}
        r = {"bloco_episodios": int(n), "bloco_train_score_mean": d["score"] / n}
        if self._registrou_causas:
            r.update({
                "bloco_win_rate": d["vitorias"] / n,
                "bloco_frac_fome": d["fome"] / n,
                "bloco_frac_colisao": d["colisao"] / n,
                "bloco_passos_por_episodio": d["passos"] / n,
            })
        return r

    def _marcar_bloco(self):
        """Fecha o bloco atual. Chamado depois de cada log, e só de lá."""
        self._acumulado_no_log = dict(self._acumulado)

    def frac(self):
        """Fração do orçamento já gasta, em [0, 1]. Base de todo agendamento linear."""
        return min(1.0, self.global_step / max(1, self.cfg.total_steps))

    def linear(self, inicio, fim):
        return inicio + self.frac() * (fim - inicio)

    # ----------------------------------------------------------- truncamento
    @staticmethod
    def desfaz_truncamento(info, prox_obs, prox_mask, done):
        """Devolve `(prox_obs, prox_mask, done)` com a morte por fome tratada como o que
        ela é: **truncamento**, não terminação.

        O `VecSnake` marca `done` para fome porque o episódio de fato acaba ali, mas
        exporta `trunc_idx`, `final_obs` e `final_mask` justamente para que o agente possa
        continuar o valor. Quem guarda a transição crua — DQN, Rainbow — tinha dois
        problemas de uma vez: gravava `done=1`, jogando fora o `γ·V(s')`, e gravava como
        `s'` a observação **do episódio seguinte**, porque o ambiente já resetou. O
        segundo é o pior: não havia como corrigir depois, o estado certo não estava no
        buffer.

        Aqui os dois somem: `s'` volta a ser o estado final verdadeiro e `done` volta a
        ser 0, que é exatamente o alvo de TD correto. Ver `docs/REVISAO_ALGORITMOS.md`
        §1.1.

        Não altera as entradas — devolve cópias.
        """
        ti = info.get("trunc_idx")
        if ti is None or len(ti) == 0:
            return prox_obs, prox_mask, done
        prox_obs = np.array(prox_obs, copy=True)
        prox_mask = np.array(prox_mask, copy=True)
        done = np.array(done, copy=True)
        prox_obs[ti] = info["final_obs"]
        prox_mask[ti] = info["final_mask"]
        done[ti] = 0.0
        return prox_obs, prox_mask, done

    @staticmethod
    def bootstrap_truncados(info, recompensas, valores_finais, gamma):
        """Soma `γ·V(s_final)` à recompensa dos episódios truncados por fome.

        A outra metade do tratamento de truncamento, para quem guarda **retornos** em vez
        de transições soltas. O `desfaz_truncamento` serve a quem tem um buffer de
        `(s, a, r, s')` e pode simplesmente devolver o `s'` verdadeiro; num rollout ou num
        segmento, o passo seguinte já pertence a outro episódio, e o valor do estado final
        precisa entrar **na recompensa** — que é como o PPO faz desde sempre.

        O `done` continua 1: a fronteira do episódio é real dentro do buffer, e é ela que
        impede o retorno de atravessar para o episódio seguinte. O que muda é que o
        retorno daquele passo deixa de valer −0,5 e passa a valer −0,5 + γ·V(s_final).

        Devolve uma cópia; sem truncamento, devolve a entrada intacta. Ver
        `docs/REVISAO_ALGORITMOS.md` §1.1.
        """
        ti = info.get("trunc_idx")
        if ti is None or len(ti) == 0:
            return recompensas
        saida = np.array(recompensas, copy=True, dtype=np.float32)
        saida[ti] += float(gamma) * np.asarray(valores_finais, dtype=np.float32)
        return saida

    # -------------------------------------------------------------- avaliação
    def politica(self):
        """A função de política que `snakeai.eval` consome. Sobrescreva se precisar."""
        return keras_policy(self.model)

    def politica_do_modelo(self, modelo):
        """A política de um modelo que veio **de fora** — um checkpoint, tipicamente.

        Existe porque `avaliar_melhor` trocava `self.model` e chamava `avaliar()`, o que
        só funciona para quem joga por `self.model`. O MuZero declara `model` como
        propriedade com setter vazio e o DreamerV3 joga por `self.ator` dentro de uma
        política recorrente: nos dois, a troca não fazia nada e a coluna `melhor` do
        registro virava uma segunda medição do modelo **final**, gravada com o passo do
        checkpoint `best`. Ver `docs/REVISAO_ALGORITMOS.md` §1.4.

        Quem não consegue jogar a partir de um `.keras` sozinho deve levantar
        `NotImplementedError` com o motivo — `avaliar_melhor` transforma isso numa coluna
        ausente e explicada, que é honesto, em vez de um número errado.
        """
        return keras_policy(modelo)

    def avaliar(self, episodes=None, safety=False, politica=None):
        """Roda o protocolo oficial. **Nunca** com exploração — é o número honesto."""
        stats, _ = evaluate(
            politica or self.politica(),
            board_size=self.cfg.board_size,
            episodes=episodes or self.cfg.eval_episodes,
            num_envs=self.cfg.eval_envs,
            greedy=CONTRATO["eval_greedy"],
            safety=safety,
            seed=CONTRATO["eval_seed"],
            # o ambiente de avaliação tem que ter os mesmos canais que o de treino
            canal_fome=getattr(self.env, "canal_fome", False),
        )
        return stats

    def rodar_protocolo(self, escolher, episodes=1000, seed=123, num_envs=None,
                        max_segundos=None, verbose=False):
        """O protocolo oficial com uma regra de escolha que precisa de mais que `(obs, mask)`.

        `snakeai.eval.evaluate` recebe uma política `(obs, mask) → logits`. Uma **busca**
        não cabe nessa interface: o AlphaZero precisa do estado do ambiente para restaurar
        nós da árvore, o MuZero precisa do latente da representação. Este laço é a mesma
        contabilidade do `evaluate` — episódios, semente, greedy, causas de fim — com a
        escolha delegada a `escolher(env, obs, mask) → ações`.

        Existe para que a coluna "com busca" não seja uma segunda implementação do
        protocolo. As duas armadilhas abaixo estavam nas cópias manuais que este método
        substitui, e as duas produzem um número silenciosamente **baixo justamente nas
        vitórias** — que é o regime em que um agente bom passa a maior parte do tempo:

        * **o score sai de `info["scores"]`, não de `env.score` lido antes do passo.** O
          episódio que termina comendo — toda vitória por tabuleiro cheio é assim — perde
          exatamente um ponto na segunda forma. Ver
          `test_eval.py::test_a_winning_episode_scores_the_last_apple`;
        * **`win_rate` sai da amostra coletada**, `(scores == perfeito).mean()`, e não de um
          contador do laço. O laço continua rodando os ambientes que já cumpriram a cota, e
          somar as vitórias deles daria uma taxa que não corresponde aos episódios medidos.

        `max_segundos` existe porque este laço **não tem um custo previsível**: ele roda até
        cada ambiente fechar a cota, e um agente bom faz episódios longos — a coleta de
        1.000 episódios com busca chega a horas. Sem uma trava, a única saída é cancelar a
        célula e perder tudo. Com ela, o que deu tempo de medir volta com
        `completo=False`, que é o campo que o `validate()` já usa para recusar uma
        avaliação parcial: o número existe para você olhar, e não entra na arena por
        engano. `verbose` imprime o progresso, porque uma espera de uma hora sem uma linha
        na tela é indistinguível de um travamento.
        """
        import time as _time
        cfg = self.cfg
        n = num_envs or min(cfg.eval_envs, 64)
        env = VecSnake(n, cfg.board_size, rng=np.random.default_rng(seed),
                       canal_fome=getattr(self.env, "canal_fome", False))
        obs, mask = env.reset()
        por_env = int(np.ceil(episodes / n))
        coletados = [[] for _ in range(n)]
        motivos = {"fome": 0, "colisao": 0, "tabuleiro_cheio": 0}
        perfeito = cfg.board_size * cfg.board_size - 3
        faltam, passos = n, 0
        t0 = _time.time()
        proximo_aviso = 30.0
        esgotou = False

        while faltam > 0:
            obs, mask, _, done, info = env.step(escolher(env, obs, mask))
            passos += n
            gasto = _time.time() - t0
            if verbose and gasto >= proximo_aviso:
                proximo_aviso = gasto + 30.0
                feitos = sum(len(c) for c in coletados)
                alvo = n * por_env
                # Os `n` ambientes correm em sincronia, então os episódios fecham em
                # levas. Extrapolar com uma leva incompleta dá um número absurdo — na
                # primeira rodada o "faltam" chegava a 256 min para um trabalho de 12.
                # Só estima depois que a primeira leva fechou.
                if feitos >= n:
                    falta = (gasto / feitos) * (alvo - feitos)
                    quanto = f"faltam ~{falta / 60:.0f} min"
                else:
                    quanto = (f"primeira leva ainda correndo — a estimativa aparece "
                              f"quando os {n} ambientes fecharem o 1º episódio")
                print(f"    ... {feitos}/{alvo} episódios · {gasto / 60:.1f} min · "
                      f"{quanto}", flush=True)
            if max_segundos is not None and gasto > max_segundos:
                esgotou = True
                break
            truncados = set(info["trunc_idx"].tolist())
            for j, i in enumerate(np.nonzero(done)[0]):
                if len(coletados[i]) >= por_env:
                    continue
                s_final = int(info["scores"][j])
                coletados[i].append(s_final)
                if i in truncados:
                    motivos["fome"] += 1
                elif s_final == perfeito:
                    motivos["tabuleiro_cheio"] += 1
                else:
                    motivos["colisao"] += 1
                if len(coletados[i]) == por_env:
                    faltam -= 1

        scores = np.array([s for l in coletados for s in l][:episodes], dtype=np.int32)
        if scores.size == 0:
            raise RuntimeError(
                "nenhum episódio terminou dentro do tempo — aumente `max_segundos` ou "
                "reduza `num_simulations`")
        total = max(1, sum(motivos.values()))
        return {
            "episodes": int(scores.size),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_std": float(scores.std()),
            "score_max": int(scores.max()),
            "score_p95": float(np.percentile(scores, 95)),
            "win_rate": float((scores == perfeito).mean()),
            "perfect_possible": perfeito,
            "env_steps_used": int(passos),
            "segundos": round(_time.time() - t0, 1),
            #: `False` quando o tempo acabou antes da cota. O `validate()` recusa uma
            #: avaliação parcial, e é isso que se quer: o número serve para olhar, não
            #: para entrar na arena por engano.
            "completo": not esgotou,
            **{f"fim_{k}": v / total for k, v in motivos.items()},
        }

    def piso(self):
        if self.baseline is None:
            self.baseline = random_baseline(
                self.cfg.board_size, self.cfg.eval_episodes, self.cfg.eval_envs,
                seed=CONTRATO["eval_seed"],
            )
        return self.baseline

    # ------------------------------------------------------------- checkpoint
    def _caminho(self, tag, ext):
        return os.path.join(self.cfg.ckpt_dir, f"{self.algo}_{tag}.{ext}")

    def modelos_extra(self):
        """Modelos/camadas além de `self.model` sem os quais a execução não se reproduz.

        O `salvar()` grava `self.model`, o que basta para quem joga com uma rede só. O
        DreamerV3 não é assim: `self.model` é o **ator**, e um ator sem o modelo do mundo
        não joga nada — a pasta da execução guardava um `.keras` que não reproduz o número
        da curva, e `retomar()` voltava com o RSSM aleatório enquanto o `global_step`
        continuava contando. Ver `docs/REVISAO_ALGORITMOS.md` §1.4.

        Devolve `{nome: modelo}`. Os pesos vão para um `.npz` ao lado do `.keras`, em vez
        de um `.keras` por peça: preserva a identidade dos objetos, e portanto as
        `tf.function` já traçadas que os capturaram.
        """
        return {}

    def _pesos_extra(self):
        return {f"{nome}/{i}": np.asarray(v)
                for nome, m in self.modelos_extra().items()
                for i, v in enumerate(m.weights)}

    def _salvar_extra(self, tag):
        pesos = self._pesos_extra()
        if pesos:
            np.savez(self._caminho(tag, "npz"), **pesos)

    def _carregar_extra(self, tag):
        """Devolve `True` se havia pesos extras para carregar."""
        caminho = self._caminho(tag, "npz")
        extras = self.modelos_extra()
        if not extras or not os.path.exists(caminho):
            return False
        with np.load(caminho) as dados:
            for nome, m in extras.items():
                for i, v in enumerate(m.weights):
                    chave = f"{nome}/{i}"
                    if chave in dados:
                        v.assign(dados[chave])
        return True

    def salvar(self, tag="last"):
        self.model.save(self._caminho(tag, "keras"))
        self._salvar_extra(tag)
        estado = {
            "global_step": self.global_step, "episodes": self.episodes,
            "iteration": self.iteration, "history": self.history,
            "evals": self.evals, "baseline": self.baseline, "melhor": self.melhor,
            "config": asdict(self.cfg), "variant": self.variant,
            # A curva **como o `Recorder` a gravou**, com o `wall_s` de cada ponto, e o
            # relógio acumulado. É o que permite a execução retomada continuar o mesmo
            # `history.json` em vez de começar outro — ver `Recorder.semear`. Guardar
            # `history`/`evals` não bastava: as linhas de avaliação da curva não estão em
            # `history`, e nenhum dos dois tem o relógio.
            "curva": list(self._rec.record.curve) if self._rec is not None else [],
            "wall_s": round(self._rec.wall_s, 3) if self._rec is not None else 0.0,
        }
        with open(self._caminho(tag, "json"), "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False)

    def retomar(self, tag="last", verbose=True):
        """Retoma do checkpoint. O Colab derruba a sessão — é questão de quando.

        **Só retoma o que é a mesma execução.** `_caminho` não tem semente nem variante no
        nome — os checkpoints são `{algo}_{tag}.keras`, um par por algoritmo, de propósito:
        a pasta é compartilhada e sobrescrita pela execução seguinte. O preço disso é que
        um checkpoint da semente 0, ou de outra região de confiança, mora exatamente onde
        a próxima execução vai procurar.

        E o modo de falhar é o pior possível, porque nada quebra: `retomar()` também
        restaura o `global_step`. Um checkpoint de uma execução de 5 M passos faz o laço de
        treino **sair na primeira verificação**, e o notebook termina em segundos —
        avaliando o modelo velho e gravando o resultado dele em
        `runs/<algo>/<variante nova>/seed<N>/`. Um número plausível, com o nome da
        configuração que nunca rodou.

        Então a identidade `(variante, semente)` é conferida antes de qualquer coisa ser
        carregada, e um checkpoint de outra identidade é **recusado**, não adotado. Quem
        quiser mesmo continuar de um modelo estranho carrega à mão; o caminho automático
        não faz isso por acidente.
        """
        import keras

        m, s = self._caminho(tag, "keras"), self._caminho(tag, "json")
        if not (os.path.exists(m) and os.path.exists(s)):
            return False
        with open(s, encoding="utf-8") as f:
            estado = json.load(f)

        antiga = (estado.get("variant"), estado.get("config", {}).get("seed"))
        atual = (self.variant, self.cfg.seed)
        # `(None, None)` é um checkpoint anterior a este campo: não há o que conferir, e
        # recusá-lo quebraria retomadas legítimas de execuções antigas.
        if antiga != (None, None) and antiga != atual:
            if verbose:
                print(f"[checkpoint] ignorado: {self._caminho(tag, 'keras')} é de "
                      f"variante={antiga[0]!r} semente={antiga[1]!r}, e esta execução é "
                      f"variante={atual[0]!r} semente={atual[1]!r}.")
                print("             Começando do zero. Apague a pasta de checkpoints "
                      "para não ver este aviso de novo.")
            return False

        self.model = keras.models.load_model(m)
        self._carregar_extra(tag)
        self.on_model_reloaded()
        self.global_step = estado["global_step"]
        self.episodes = estado["episodes"]
        self.iteration = estado["iteration"]
        self.history = estado["history"]
        self.evals = estado.get("evals", [])
        self.baseline = estado.get("baseline")
        self.melhor = estado.get("melhor", -np.inf)
        self._curva_retomada = estado.get("curva") or []
        self._wall_retomado = float(estado.get("wall_s") or 0.0)
        self._proximo_eval = proximo_multiplo(self.global_step,
                                              self.cfg.eval_every_steps)
        self._proximo_log = self.global_step
        return True

    @staticmethod
    def _com_sufixo(variant, sufixo):
        """Acrescenta `sufixo` à variante, sem duplicar quando ela já o traz."""
        if not sufixo:
            return variant
        sufixo = sufixo if sufixo.startswith("_") else f"_{sufixo}"
        return variant if variant.endswith(sufixo) else variant + sufixo

    def on_model_reloaded(self):
        """Gancho: o otimizador antigo aponta para as variáveis do modelo antigo."""

    # ------------------------------------------------------------------ treino
    def iterate(self):
        raise NotImplementedError

    def train(self, verbose=True, ate_passos=None):
        """Roda até o orçamento, avaliando na cadência oficial. Devolve o `RunRecord`."""
        alvo = ate_passos or self.cfg.total_steps
        # O `env_spec` descreve o ambiente que **de fato** rodou, não o contrato: uma
        # execução com `canal_fome=True` gravava `n_channels: 5` no registro, e o
        # arquivo mentia sobre a própria observação. Ele continua idêntico ao contrato
        # em qualquer execução de 5 canais, que é o caso normal.
        env_spec = dict(CONTRATO)
        canais = getattr(getattr(self, "env", None), "n_channels", None)
        if canais:
            env_spec["n_channels"] = int(canais)

        rec = Recorder(self.algo, variant=self.variant, seed=self.cfg.seed,
                       net=self.cfg.net,
                       params=self.model.count_params() if self.model else 0,
                       config=asdict(self.cfg), env_spec=env_spec,
                       root=self.cfg.runs_dir)
        # Uma execução retomada é uma chamada nova de `train()`, e portanto um `Recorder`
        # novo. Semear é o que faz o `history.json` continuar em vez de recomeçar.
        if self._curva_retomada or self._wall_retomado:
            rec.semear(self._curva_retomada, self._wall_retomado)
            if verbose:
                print(f"[registro] continuando a curva: {len(rec.record.curve)} pontos "
                      f"e {self._wall_retomado / 60:.0f} min já gravados")
        self._rec = rec
        self.piso()

        while self.global_step < alvo:
            stats = self.iterate()
            self.iteration += 1
            # Quantos passos de gradiente o orçamento de ambiente comprou. Fica no
            # metadado porque é o eixo do §2.1 da revisão e não dá para reconstruir do
            # `config` — o early-stop por KL corta épocas. Zero significa "o agente não
            # reporta", não "não atualizou".
            self._atualizacoes += int(stats.get("atualizacoes", 0) or 0)

            # Quem chama `registra_fim` no laço de coleta já contabilizou os episódios com
            # a causa da morte junto; registrar de novo aqui contaria cada um duas vezes e
            # a média móvel ficaria certa por acidente, mas as frações, não.
            m, k = stats.get("train_score_mean"), stats.get("n_episodes") or 0
            if not self._registrou_causas and m is not None and k:
                self._registra_episodios(m, k)

            if self.global_step >= self._proximo_log:
                self._proximo_log = self.global_step + self.cfg.log_every_steps
                # a curva registra a **média móvel**, não a iteração isolada: é o número
                # que responde "o treino está andando?" sem depender de quantos episódios
                # acabaram no exato momento do log
                bloco = self.resumo_bloco()
                ponto = {"episodes": self.episodes,
                         "train_score_mean": self.media_movel(),
                         "train_score_iter": stats.get("train_score_mean"),
                         **{k: v for k, v in stats.items() if k != "train_score_mean"},
                         **self.resumo_janela(), **bloco}
                self.history.append({"global_step": self.global_step, **ponto})
                rec.log(self.global_step, **ponto)
                if verbose:
                    self._imprimir(stats, bloco)
                self._marcar_bloco()

            if self.global_step >= self._proximo_eval:
                self._proximo_eval = proximo_multiplo(self.global_step,
                                                      self.cfg.eval_every_steps)
                av = self.avaliar()
                av["global_step"] = self.global_step
                av["episodes"] = self.episodes
                self.evals.append(av)
                rec.log(self.global_step, eval_score_mean=av["score_mean"],
                        eval_score_p95=av["score_p95"], episodes=self.episodes)
                if verbose:
                    print(f"  [eval] passo {self.global_step:,} · "
                          f"score {av['score_mean']:.2f} "
                          f"(piso {self.baseline:.2f})")
                if av["score_mean"] > self.melhor:
                    self.melhor = av["score_mean"]
                    self.salvar("best")
                self.salvar("last")
                # E o registro vai junto, na cadência da avaliação. Antes, `runs/` só
                # nascia no fim: uma sessão derrubada às 4 h de treino não deixava curva
                # nenhuma lá — só o checkpoint, que ninguém lê como registro. O
                # `skip_validation` é obrigatório aqui: `final` só existe depois do último
                # passo, então o parcial **não** passa no contrato, e é assim que se
                # distingue um arquivo em andamento de uma execução terminada.
                try:
                    rec.save(skip_validation=True)
                except Exception as e:                  # nunca derrubar o treino por isso
                    if verbose:
                        print(f"  [registro] parcial não gravado: {e!r}")

        final = self.avaliar()
        rec.log(self.global_step, eval_score_mean=final["score_mean"],
                eval_score_p95=final["score_p95"], episodes=self.episodes)

        # O melhor checkpoint é medido com o **mesmo** protocolo, e não reaproveita o
        # número da avaliação periódica: aquele veio de outra amostra, e comparar duas
        # medições ruidosas favorece sistematicamente quem foi medido mais vezes.
        melhor = self.avaliar_melhor(verbose=verbose)
        rec.finish(final, melhor_stats=melhor,
                   comparable=getattr(self.cfg, "comparable", True),
                   caveat=getattr(self.cfg, "caveat", ""))
        rec.record.meta["baseline"] = self.baseline
        # Onde este número foi produzido. Uma curva do Kaggle e outra do Colab são
        # comparáveis — o contrato garante isso — mas o **tempo de parede** não é, e
        # `wall_s_total` é lido com frequência como se fosse.
        rec.record.meta.update(resumo_plataforma())
        # Quantos canais a rede realmente viu. Fica no metadado porque é a diferença que
        # torna uma curva incomparável com outra, e "comparable=False + caveat em prosa"
        # não é conferível por máquina — este número é.
        if getattr(self, "env", None) is not None:
            rec.record.meta["obs_channels"] = int(
                getattr(self.env, "n_channels", CONTRATO["n_channels"]))
        if self._atualizacoes:
            rec.record.meta["atualizacoes"] = int(self._atualizacoes)
        self.salvar("last")

        # O registro é gravado SEMPRE. Estourar no fim de um treino de horas e perder a
        # curva seria o pior desfecho possível; o portão do contrato age na hora de
        # montar a arena, não na hora de escrever. As violações ficam no metadado e
        # `RunRecord.oficial` passa a ser False.
        problemas = validate(rec.record)
        if problemas:
            rec.record.meta["contract_violations"] = problemas
            if verbose:
                print("\n[contrato] esta execução NÃO entra na arena:")
                for p in problemas:
                    print(f"  - {p}")
        caminho = rec.save(skip_validation=True)
        if verbose:
            print(f"[registro] {caminho}")

        self.artefatos(rec, verbose=verbose)
        return rec

    # ---------------------------------------------------------------- artefatos
    def modelo_melhor(self):
        """O modelo do checkpoint `best`, ou `None` se ele não existe.

        Carrega numa instância separada de propósito: `self.model` continua sendo o do
        último passo, porque é ele que define a curva e o número oficial. Trocar em
        silêncio faria a última avaliação medir uma coisa e a curva outra.
        """
        import keras

        caminho = self._caminho("best", "keras")
        if not os.path.exists(caminho):
            return None
        return keras.models.load_model(caminho)

    @contextlib.contextmanager
    def politica_de_checkpoint(self, tag="best"):
        """Uma política que joga pelo checkpoint `tag`, válida dentro do bloco.

        `None` quando o checkpoint não existe. É um gerenciador de contexto porque há
        agentes — o DreamerV3 — que só conseguem jogar um checkpoint **trocando os pesos
        dos próprios submodelos**, e nesse caso a restauração precisa acontecer mesmo se a
        avaliação levantar.
        """
        m = self.modelo_melhor() if tag == "best" else None
        yield None if m is None else self.politica_do_modelo(m)

    def avaliar_melhor(self, verbose=True):
        """Roda o protocolo oficial sobre o melhor checkpoint. `{}` se não houver.

        Avalia **pelo modelo carregado**, sem tocar em `self.model`: a troca de atributo
        era silenciosamente ineficaz em dois agentes (ver `politica_do_modelo`).
        """
        try:
            with self.politica_de_checkpoint("best") as pol:
                if pol is None:
                    return {}
                stats = self.avaliar(politica=pol)
        except NotImplementedError as e:
            if verbose:
                print(f"  [melhor] não avaliado: {e}")
            return {"indisponivel": str(e),
                    "global_step": int(self._passo_do_melhor())}
        stats["global_step"] = int(self._passo_do_melhor())
        if verbose:
            print(f"  [melhor] checkpoint do passo {stats['global_step']:,} · "
                  f"score {stats['score_mean']:.2f} "
                  f"(último: {self.evals[-1]['score_mean']:.2f})"
                  if self.evals else
                  f"  [melhor] score {stats['score_mean']:.2f}")
        return stats

    def _passo_do_melhor(self):
        caminho = self._caminho("best", "json")
        if os.path.exists(caminho):
            with open(caminho, encoding="utf-8") as f:
                return json.load(f).get("global_step", 0)
        return 0

    def copiar_modelos(self, destino, verbose=True):
        """Leva `last.keras` e `best.keras` para dentro da pasta da execução.

        Os checkpoints vivem em `ckpt_dir`, que é compartilhado e sobrescrito pela
        execução seguinte. Sem esta cópia, o `history.json` afirma um score que ninguém
        consegue reproduzir nem inspecionar depois — e o GIF vira a única evidência de
        como o agente jogava.

        Os dois, e não só o melhor: `last` é o modelo que produziu o número **oficial**,
        então é ele que permite reconferir a curva; `best` é o que se leva para o jogo.
        """
        import shutil

        pasta = os.path.join(destino, "modelos")
        os.makedirs(pasta, exist_ok=True)
        copiados = {}
        for tag in ("last", "best"):
            # o `.npz` acompanha o `.keras`: para o DreamerV3 é ele que carrega o modelo
            # do mundo, e sem ele a pasta guarda um ator que não joga (§1.4 da revisão)
            for ext in ("keras", "npz"):
                origem = self._caminho(tag, ext)
                if os.path.exists(origem):
                    alvo = os.path.join(pasta, f"{tag}.{ext}")
                    shutil.copyfile(origem, alvo)
                    copiados[tag if ext == "keras" else f"{tag}+pesos"] = alvo
        if verbose and copiados:
            mb = sum(os.path.getsize(c) for c in copiados.values()) / 1e6
            print(f"  [modelos] {', '.join(sorted(copiados))} em {pasta} ({mb:.1f} MB)")
        return copiados

    def artefatos(self, rec, verbose=True):
        """Gráfico, GIFs e os modelos — tudo ao lado do `history.json`.

        A pasta da execução tem que ser autossuficiente: quem a recebe consegue ver a
        curva, ver o agente jogando e **rodar o modelo**, sem depender de nenhum estado
        que ficou na máquina de quem treinou.
        """
        import os

        destino = os.path.dirname(rec.save(skip_validation=True))
        saida = {}
        saida["modelos"] = self.copiar_modelos(destino, verbose=verbose)

        if self.cfg.salvar_grafico:
            try:
                import matplotlib
                matplotlib.use("Agg")
                from ..plot import plot_run

                fig, _ = plot_run(rec.record)
                caminho = os.path.join(destino, "curva.png")
                fig.savefig(caminho, dpi=150, facecolor=fig.get_facecolor())
                matplotlib.pyplot.close(fig)
                saida["grafico"] = caminho
            except Exception as e:                      # nunca derrubar o treino por isso
                saida["grafico_erro"] = repr(e)

        if self.cfg.salvar_gif:
            from ..env.render import render_episode

            politica = self.politica()
            for seed in self.cfg.gif_seeds:
                try:
                    caminho, score, motivo = render_episode(
                        politica, caminho=os.path.join(destino, f"episodio_s{seed}.gif"),
                        board_size=self.cfg.board_size, seed=seed,
                        canal_fome=getattr(self.env, "canal_fome", False),
                    )
                    saida[f"gif_s{seed}"] = {"caminho": caminho, "score": score,
                                             "fim": motivo}
                    if verbose:
                        print(f"[gif] seed {seed}: score {score}, terminou por {motivo}")
                except Exception as e:
                    saida[f"gif_s{seed}_erro"] = repr(e)

        rec.record.meta["artefatos"] = saida
        rec.save(skip_validation=True)
        return saida

    #: Legenda impressa uma vez, antes da primeira linha de log.
    LEGENDA = ("[log] cada métrica sai como  janela | bloco  — a média móvel dos últimos "
               "~{janela} episódios\n"
               "      à esquerda, e só os episódios encerrados desde o log anterior à "
               "direita.\n"
               "      Elas discordam por muitos logs antes de a móvel virar: a da "
               "esquerda diz onde o\n"
               "      agente está, a da direita diz para onde ele está indo.")

    @staticmethod
    def _par(janela, bloco, fmt, largura):
        """`janela | bloco` no mesmo formato, com `—` quando o bloco está vazio."""
        esq = f"{janela:{fmt}}" if janela is not None else "—"
        dir_ = f"{bloco:{fmt}}" if bloco is not None else "—"
        return f"{esq:>{largura}}|{dir_:<{largura}}"

    def _imprimir(self, stats, bloco=None):
        """Uma linha por log: a janela móvel e o bloco novo, lado a lado.

        O score sozinho é ambíguo: 1,2 pontos pode ser "bate em tudo" ou "anda em círculo
        até morrer de fome", e a curva fica igual nos dois casos. Por isso a linha traz a
        **repartição das causas de fim**, que separa os dois de imediato, mais o
        comprimento médio do episódio, que é o sinal mais precoce de todos — uma cobra que
        aprende a sobreviver alonga os episódios antes de o score subir.

        E cada uma dessas medidas aparece **duas vezes**: sobre a janela de
        `JANELA_EPISODIOS` e sobre os episódios encerrados desde o log anterior. A média
        móvel existe para o número não pular com uma amostra de 3 episódios, mas o preço é
        atraso — com ~80 episódios por log ela carrega seis logs de passado. Numa
        degradação em curso as duas colunas discordam bem antes de a curva virar, e é
        exatamente essa discordância que se quer ver.
        """
        if not self._legenda_impressa:
            print(self.LEGENDA.format(janela=self.JANELA_EPISODIOS))
            self._legenda_impressa = True
        r = self.resumo_janela()
        b = bloco if bloco is not None else self.resumo_bloco()
        n_bloco = b.get("bloco_episodios", 0)
        partes = [
            f"passo {self.global_step:>10,}",
            f"ep {self.episodes:>7,} +{n_bloco:<4}",
            "score " + self._par(self.media_movel(), b.get("bloco_train_score_mean"),
                                 ".2f", 6),
        ]
        if "win_rate" in r:
            partes += [
                "fome " + self._par(r["frac_fome"], b.get("bloco_frac_fome"), ".1%", 6),
                "colisão " + self._par(r["frac_colisao"], b.get("bloco_frac_colisao"),
                                       ".1%", 6),
                "vit " + self._par(r["win_rate"], b.get("bloco_win_rate"), ".1%", 6),
                self._par(r["passos_por_episodio"], b.get("bloco_passos_por_episodio"),
                          ".0f", 4) + " passos/ep",
            ]
        partes.append(f"janela {self.episodios_na_janela()}")
        print(" · ".join(partes))
