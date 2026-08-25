"""A exportação — e a conferência de paridade, que é o que a torna uma afirmação.

O defeito que motivou estes testes quebrou uma execução do Rainbow **na penúltima
célula**, depois de 19 mil segundos de GPU::

    ValueError: operands could not be broadcast together with shapes (200,) (200,121)

A causa: a redução "distribuição de átomos → escore por ação" era aplicada só no lado
Keras. O lado TFLite continuava `(lote, ações, átomos)`, e os dois `argmax` comparavam
eixos diferentes. O mesmo descuido, num formato diferente, atingia o LBC — cuja saída é
`(lote, políticas, ações)`.

Os testes rápidos (NumPy puro) fixam a redução e a escolha da saída; os de ponta a ponta
convertem de verdade, porque a forma da saída do `Interpreter` é justamente o que ninguém
consegue prever de cabeça.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pytest

from snakeai.env.vec_snake import N_ACTIONS
from snakeai.export import (
    _escores_por_acao,
    _indice_da_politica,
    _q_de_logits_c51,
    export_model,
)
from snakeai.nets import build_actor_critic_populacao, build_policy_q, build_q_network
from snakeai.nets.heads import q_de_distribuicao, suporte_c51

N_ATOMOS = 121


# ------------------------------------------------------ a redução para escore por ação
def test_c51_collapse_picks_the_same_action_as_the_real_q():
    """`_q_de_logits_c51` não calcula `Q` — calcula algo com o **mesmo `argmax`**.

    O suporte é afim e crescente (`z_i = v_min + i·Δz`), então a esperança do índice do
    átomo ordena as ações exatamente como `Σ p·z`. É o que permite ao exportador conferir
    a escolha sem conhecer `v_min`/`v_max`, que moram no agente.
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, N_ACTIONS, N_ATOMOS)).astype(np.float32)

    for v_min, v_max in ((-24.0, 24.0), (-2.0, 60.0), (0.0, 1.0)):
        q = np.asarray(q_de_distribuicao(logits, suporte_c51(v_min, v_max, N_ATOMOS)))
        assert (q.argmax(-1) == _q_de_logits_c51(logits).argmax(-1)).all(), \
            f"a escolha mudou com o suporte [{v_min}, {v_max}]"


def test_c51_collapse_is_not_the_mean_of_the_logits():
    """A média dos logits — o que estava aqui antes — ignora a softmax e troca a ação.

    Não é um detalhe estético: era o número que o relatório publicava como
    `acoes_iguais`.
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(512, N_ACTIONS, N_ATOMOS)).astype(np.float32) * 3.0
    q = np.asarray(q_de_distribuicao(logits, suporte_c51(-24.0, 24.0, N_ATOMOS)))
    assert (logits.mean(-1).argmax(-1) != q.argmax(-1)).any()


@pytest.mark.parametrize("forma, esperada", [
    ((7, N_ACTIONS), (7,)),                        # actor-critic, DQN sem C51
    ((7, N_ACTIONS, N_ATOMOS), (7,)),              # C51 — o caso do Rainbow
    ((7, 3, N_ACTIONS), (7, 3)),                   # população do LBC
])
def test_action_axis_is_found_in_every_output_format(forma, esperada):
    """A ação sai do último eixo depois da redução — em todos os formatos do repositório.

    O `(7, 3, 3)` do LBC é o caso ambíguo de propósito: `N_ACTIONS = 3` e a população
    padrão tem 3 políticas. Ele tem de ser lido como população, não como um C51 de três
    átomos.
    """
    rng = np.random.default_rng(2)
    t = rng.normal(size=forma).astype(np.float32)
    assert _escores_por_acao(t).argmax(-1).shape == esperada


def test_unknown_output_format_says_so_instead_of_broadcasting():
    with pytest.raises(ValueError, match="ações"):
        _escores_por_acao(np.zeros((4, 7, 9), dtype=np.float32))


# --------------------------------------------------- qual saída do .tflite é a política
def test_policy_output_is_chosen_by_shape_not_by_column_count():
    """A regra antiga — "a que tem `N_ACTIONS` colunas" — não achava a saída do C51.

    `(1, 3, 121)` tem 121 colunas, não 3; a busca falhava e o código caía na primeira
    saída sem avisar.
    """
    ref = np.zeros((1, N_ACTIONS, N_ATOMOS), dtype=np.float32)
    cands = [np.zeros((1, 1), dtype=np.float32), ref.copy()]
    assert _indice_da_politica(cands, ref) == 1


def test_ties_are_broken_by_value_because_output_order_is_not_guaranteed():
    """No ACER as duas saídas (`logits` e `Q(s,·)`) têm a mesma forma.

    A ordem das saídas do `Interpreter` não é a ordem do `keras.Model`, então pegar a
    primeira é sortear entre comparar a política com a política e a política com o
    crítico. O valor desempata.
    """
    ref = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    outra = np.array([[9.0, -4.0, 0.0]], dtype=np.float32)
    assert _indice_da_politica([outra, ref + 1e-3], ref) == 1
    assert _indice_da_politica([ref + 1e-3, outra], ref) == 0


def test_no_matching_output_is_an_error_with_the_shapes_in_it():
    with pytest.raises(ValueError, match="nenhuma saída"):
        _indice_da_politica([np.zeros((1, 8), dtype=np.float32)],
                            np.zeros((1, N_ACTIONS), dtype=np.float32))


# ------------------------------------------------------------------- de ponta a ponta
@pytest.mark.parametrize("nome, constroi", [
    ("c51", lambda: build_q_network(net="resnet_tiny", dueling=True, noisy=True,
                                    n_atoms=N_ATOMOS)),
    ("lbc", lambda: build_actor_critic_populacao(net="resnet_tiny", n_politicas=3)),
    ("acer", lambda: build_policy_q(net="resnet_tiny")),
])
def test_export_checks_parity_for_every_output_format(tmp_path, nome, constroi):
    """Sem quantização, o `.tflite` tem de escolher **a mesma** ação que o `.keras`.

    Os três formatos que não eram `(lote, ações)` estão aqui: o C51 do Rainbow quebrava
    com `ValueError`, o do LBC também, e o do ACER passava — comparando, dependendo da
    ordem em que o conversor listou as saídas, a política contra o crítico.
    """
    rel = export_model(constroi(), out_dir=str(tmp_path / nome), formatos=("fp32",))
    paridade = rel["fp32_paridade"]
    assert "erro" not in paridade, paridade["erro"]
    assert paridade["acoes_iguais"] >= 0.99
    assert paridade["erro_max_logits"] < 1e-2


def test_a_failed_parity_check_does_not_throw_away_the_export(tmp_path, monkeypatch):
    """A conferência roda **depois** de os arquivos estarem em disco, no fim de um treino.

    Deixá-la levar a execução junto foi o que custou 19 mil segundos de GPU. O relatório
    registra a falha — e o `.keras` e o `.tflite` continuam lá.
    """
    import snakeai.export as ex

    monkeypatch.setattr(ex, "conferir_paridade",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    rel = ex.export_model(build_q_network(net="resnet_tiny"),
                          out_dir=str(tmp_path / "falha"), formatos=("fp32",))

    assert rel["fp32_paridade"] == {"erro": "ValueError: boom"}
    assert (tmp_path / "falha" / "modelo.keras").exists()
    assert (tmp_path / "falha" / "modelo_fp32.tflite").exists()
