"""Andaime comum a todos os agentes.

O que fica aqui é o que **precisa** ser idêntico entre algoritmos para que a comparação
valha: a cadência da avaliação, o formato do registro, o critério de "melhor checkpoint",
e os agendamentos lineares. O que varia — como o agente aprende — fica em cada módulo.

Foi essa separação que faltou no repositório antigo: cada notebook tinha o próprio laço de
treino, a própria noção de época e o próprio jeito de avaliar, e por isso as curvas nunca
puderam ser sobrepostas.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..eval import evaluate, keras_policy, random_baseline
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

    def __post_init__(self):
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

    def __init__(self, cfg, variant="default"):
        self.cfg = cfg
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
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------- agendamentos
    def frac(self):
        """Fração do orçamento já gasta, em [0, 1]. Base de todo agendamento linear."""
        return min(1.0, self.global_step / max(1, self.cfg.total_steps))

    def linear(self, inicio, fim):
        return inicio + self.frac() * (fim - inicio)

    # -------------------------------------------------------------- avaliação
    def politica(self):
        """A função de política que `snakeai.eval` consome. Sobrescreva se precisar."""
        return keras_policy(self.model)

    def avaliar(self, episodes=None, safety=False):
        """Roda o protocolo oficial. **Nunca** com exploração — é o número honesto."""
        stats, _ = evaluate(
            self.politica(),
            board_size=self.cfg.board_size,
            episodes=episodes or self.cfg.eval_episodes,
            num_envs=self.cfg.eval_envs,
            greedy=CONTRATO["eval_greedy"],
            safety=safety,
            seed=CONTRATO["eval_seed"],
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

    def salvar(self, tag="last"):
        self.model.save(self._caminho(tag, "keras"))
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
        rec = Recorder(self.algo, variant=self.variant, seed=self.cfg.seed,
                       net=self.cfg.net,
                       params=self.model.count_params() if self.model else 0,
                       config=asdict(self.cfg), root=self.cfg.runs_dir)
        self.piso()

        while self.global_step < alvo:
            stats = self.iterate()
            self.iteration += 1

            if self.global_step >= self._proximo_log:
                self._proximo_log = self.global_step + self.cfg.log_every_steps
                ponto = {"episodes": self.episodes,
                         "train_score_mean": stats.get("train_score_mean"),
                         **{k: v for k, v in stats.items() if k != "train_score_mean"}}
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
        rec.finish(final)
        rec.record.meta["baseline"] = self.baseline
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
    def artefatos(self, rec, verbose=True):
        """Gráfico de diagnóstico e GIFs do agente jogando.

        Ficam ao lado do `history.json`, na pasta da execução — assim um checkpoint
        antigo nunca fica órfão da imagem que o explicava.
        """
        import os

        destino = os.path.dirname(rec.save(skip_validation=True))
        saida = {}

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
        m = stats.get("train_score_mean")
        m = f"{m:.2f}" if m is not None else "—"
        print(f"passo {self.global_step:>10,} · ep {self.episodes:>8,} · "
              f"treino {m:>6}")
