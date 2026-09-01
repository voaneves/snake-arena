"""O meta-controlador do LBC, isolado do treino.

O bandit é a peça que decide *como explorar*, e ele erra em silêncio: uma seleção que
nunca sai do uniforme e uma que trava no primeiro braço produzem curvas de treino
plausíveis e diferentes, sem exceção nenhuma no caminho. Estes testes fixam os três
comportamentos que separam um UCB de um sorteio caro — varrer no começo, decidir quando há
evidência, e **esquecer** quando a evidência envelhece.
"""

import numpy as np
import pytest

from snakeai.bandit import BanditUCB


def bandit(n=4, **kw):
    kw.setdefault("c", 1.0)
    kw.setdefault("janela", 16)
    return BanditUCB(n, rng=np.random.default_rng(0), **kw)


# ------------------------------------------------------------------- o básico
def test_a_bandit_needs_arms():
    with pytest.raises(ValueError):
        BanditUCB(0)


def test_an_arm_with_no_data_has_no_value():
    """`NaN`, não zero. Zero afirmaria que o braço é ruim, e ninguém mediu isso."""
    b = bandit()
    assert np.isnan(b.valores()).all()
    assert b.visitas().sum() == 0


def test_the_distribution_is_uniform_before_any_evidence():
    p = bandit().distribuicao()
    assert p == pytest.approx(np.full(4, 0.25))


# ------------------------------------------------------------------ exploração
def test_an_unvisited_arm_is_optimistic_not_pessimistic():
    """Um braço nunca puxado tem que competir com o melhor, não com o pior.

    Sem o otimismo, um braço que ninguém tocou entra na softmax com valor indefinido e
    some — e o bandit passa o treino inteiro convencido de que a primeira coisa que tentou
    era a melhor, sem nunca ter olhado as outras.
    """
    b = bandit()
    for _ in range(10):
        b.registrar(0, 50.0)
    p = b.distribuicao()
    assert p[1] > p[0], "o braço virgem tem que ser preferido ao já explorado"


def test_c_zero_turns_the_bandit_greedy():
    b = bandit(c=0.0)
    for _ in range(10):
        b.registrar(0, 1.0)
        b.registrar(1, 10.0)
    assert int(np.argmax(b.distribuicao())) == 1


def test_c_buys_probability_for_the_under_visited_arm():
    """O que `c` compra, e onde: num braço **pouco visitado**, não em todos.

    Com todos os braços puxados igualmente o bônus é o mesmo para todos e some da
    softmax — o que está certo, e é fácil de testar errado. O `c` só aparece quando as
    contagens diferem, que é exatamente a situação em que a incerteza difere.
    """
    def p_do_pior_pouco_visto(c):
        b = bandit(n=2, c=c)
        for _ in range(30):
            b.registrar(0, 10.0)
        for _ in range(3):
            b.registrar(1, 1.0)
        return b.distribuicao()[1]

    assert p_do_pior_pouco_visto(0.0) < p_do_pior_pouco_visto(1.0)
    assert p_do_pior_pouco_visto(1.0) < p_do_pior_pouco_visto(5.0)


def test_the_temperature_is_what_lets_the_bandit_decide():
    """Com os valores normalizados em `[0, 1]`, uma softmax de temperatura 1 nunca dá a um
    braço mais que ~2,7× a probabilidade de outro. Sem este parâmetro, "aprender a
    selecionar" seria impossível por construção — e a curva do LBC seria a da ablação de
    seleção aleatória, sem que nada quebrasse."""
    def p_top(temperatura):
        b = bandit(n=4, c=0.0, temperatura=temperatura)
        for _ in range(10):
            for k, v in enumerate((0.0, 1.0, 2.0, 10.0)):
                b.registrar(k, v)
        return b.distribuicao().max()

    assert p_top(1.0) < 0.6
    assert p_top(0.1) > 0.9


# -------------------------------------------------------------- não estacionário
def test_the_window_forgets_an_arm_that_stopped_paying():
    """O motivo de existir da janela deslizante.

    A política muda embaixo do bandit: um comportamento muito exploratório é ótimo no
    início e péssimo no fim. Com média desde sempre, o braço bom de antes continuaria
    sendo escolhido por centenas de milhares de passos depois de deixar de ser bom.
    """
    b = bandit(n=2, janela=8, c=0.0)
    for _ in range(50):          # o braço 0 foi excelente por muito tempo
        b.registrar(0, 100.0)
        b.registrar(1, 10.0)
    assert int(np.argmax(b.distribuicao())) == 0

    for _ in range(8):           # e agora não é mais — uma janela basta para virar
        b.registrar(0, 0.0)
        b.registrar(1, 10.0)
    assert int(np.argmax(b.distribuicao())) == 1
    assert b.visitas()[0] == 8, "a janela não pode guardar mais que o próprio tamanho"


def test_total_pulls_survive_the_window():
    """A contagem da janela governa o bônus; a contagem total é relatório.

    São números diferentes de propósito, e trocá-los faz o bônus parar de decair — o
    bandit continuaria explorando como se estivesse no primeiro episódio.
    """
    b = bandit(n=2, janela=4)
    for _ in range(10):
        b.registrar(0, 1.0)
    assert b.visitas()[0] == 4
    assert b.puxadas_totais[0] == 10


# ------------------------------------------------------------------ normalização
def _entropia_da_selecao(escala, **kw):
    """Entropia de `P_Ψ` quando os três braços rendem 1, 2 e 3 vezes `escala`."""
    b = bandit(n=3, c=1.0, **kw)
    for _ in range(10):
        for k, v in enumerate((1.0, 2.0, 3.0)):
            b.registrar(k, v * escala)
    p = b.distribuicao()
    return float(-(p * np.log(p + 1e-12)).sum())


def test_normalization_makes_the_selection_independent_of_the_reward_scale():
    """A escala do retorno muda por duas ordens de grandeza durante o treino — no Snake,
    de ~1 ponto para ~80. Sem normalizar, a mesma configuração de bandit é quase uniforme
    no começo e quase gulosa no fim, **sem que ninguém tenha mexido em nada**: a exploração
    do meta-controlador viraria função do quanto o agente já sabe jogar.

    Normalizado, o que importa é a ordem e a distância relativa entre os braços, e as duas
    sobrevivem à mudança de escala.
    """
    assert _entropia_da_selecao(0.001) == pytest.approx(_entropia_da_selecao(1.0),
                                                        abs=1e-9)


def test_without_normalization_the_scale_decides_how_greedy_the_bandit_is():
    """O controle do teste anterior: com a fórmula crua do paper, mudar só a escala do
    retorno leva a seleção de praticamente uniforme a praticamente determinística."""
    # `piso_uniforme=0` porque o que se mede aqui é a fórmula crua do paper, e o piso é
    # um acréscimo deste repositório (ver `docs/LBC.md` §2.9).
    quase_uniforme = _entropia_da_selecao(0.001, normalizar=False, piso_uniforme=0.0)
    quase_guloso = _entropia_da_selecao(1.0, normalizar=False, piso_uniforme=0.0)
    assert quase_uniforme > 1.0          # log 3 = 1,0986
    assert quase_guloso < 0.1


def test_identical_arms_do_not_blow_up_the_normalization():
    """Todo braço com o mesmo valor é o caso do começo do treino, quando ninguém pontua."""
    b = bandit(n=3)
    for _ in range(5):
        for k in range(3):
            b.registrar(k, 0.0)
    p = b.distribuicao()
    assert np.isfinite(p).all() and p == pytest.approx(np.full(3, 1 / 3))


# ---------------------------------------------------------------------- relato
def test_the_summary_says_whether_selection_learned_anything():
    """`mab_entropia` perto de `log K` é um bandit que não decidiu nada — e a curva
    deveria então coincidir com a da ablação de seleção aleatória."""
    b = bandit(n=4, c=0.0)
    assert b.resumo()["mab_entropia"] == pytest.approx(np.log(4))
    for _ in range(20):
        b.registrar(2, 100.0)
        b.registrar(0, 0.0)
        b.registrar(1, 0.0)
        b.registrar(3, 0.0)
    r = b.resumo()
    assert r["mab_braco_top"] == 2
    assert r["mab_entropia"] < r["mab_entropia_max"]
    assert r["mab_bracos_visitados"] == 4


def test_sampling_respects_the_distribution():
    b = bandit(n=2, c=0.0)
    for _ in range(20):
        b.registrar(0, 0.0)
        b.registrar(1, 100.0)
    amostras = b.amostrar(2000)
    assert (amostras == 1).mean() > 0.9
