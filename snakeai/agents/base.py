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
        if getattr(cfg, "canal_fome", False) and not variant.endswith(self.SUFIXO_FOME):
            variant += self.SUFIXO_FOME
        self.variant = variant
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
        self._registrou_causas = False
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
        }
        with open(self._caminho(tag, "json"), "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False)

    def retomar(self, tag="last"):
        """Retoma do checkpoint. O Colab derruba a sessão — é questão de quando."""
        import keras

        m, s = self._caminho(tag, "keras"), self._caminho(tag, "json")
        if not (os.path.exists(m) and os.path.exists(s)):
            return False
        self.model = keras.models.load_model(m)
        self._carregar_extra(tag)
        self.on_model_reloaded()
        with open(s, encoding="utf-8") as f:
            estado = json.load(f)
        self.global_step = estado["global_step"]
        self.episodes = estado["episodes"]
        self.iteration = estado["iteration"]
        self.history = estado["history"]
        self.evals = estado.get("evals", [])
        self.baseline = estado.get("baseline")
        self.melhor = estado.get("melhor", -np.inf)
        self._proximo_eval = self.global_step + self.cfg.eval_every_steps
        self._proximo_log = self.global_step
        return True

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
                ponto = {"episodes": self.episodes,
                         "train_score_mean": self.media_movel(),
                         "train_score_iter": stats.get("train_score_mean"),
                         **{k: v for k, v in stats.items() if k != "train_score_mean"},
                         **self.resumo_janela()}
                self.history.append({"global_step": self.global_step, **ponto})
                rec.log(self.global_step, **ponto)
                if verbose:
                    self._imprimir(stats)

            if self.global_step >= self._proximo_eval:
                self._proximo_eval = self.global_step + self.cfg.eval_every_steps
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

    def _imprimir(self, stats):
        """Uma linha por log, sobre a janela de episódios — ver `self._janela`.

        O score sozinho é ambíguo: 1,2 pontos pode ser "bate em tudo" ou "anda em círculo
        até morrer de fome", e a curva fica igual nos dois casos. Por isso a linha traz a
        **repartição das causas de fim**, que separa os dois de imediato, mais o
        comprimento médio do episódio, que é o sinal mais precoce de todos — uma cobra que
        aprende a sobreviver alonga os episódios antes de o score subir.
        """
        r = self.resumo_janela()
        m = self.media_movel()
        partes = [f"passo {self.global_step:>10,}",
                  f"ep {self.episodes:>8,}",
                  f"treino {(f'{m:.2f}' if m is not None else '—'):>6}"]
        if "win_rate" in r:
            partes.append(f"vit {r['win_rate']:6.1%}")
            partes.append(f"fome {r['frac_fome']:5.1%}")
            partes.append(f"colisão {r['frac_colisao']:5.1%}")
            partes.append(f"{r['passos_por_episodio']:5.0f} passos/ep")
        partes.append(f"(janela de {self.episodios_na_janela()} episódios)")
        print(" · ".join(partes))
