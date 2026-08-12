"""Os notebooks do Colab.

O acordo: **o único arquivo que vai para o Colab é o `.ipynb`**, e ele roda do zero, sem
clonar nada. Ao mesmo tempo, o ambiente e a régua de avaliação precisam ser idênticos em
todos — senão as curvas voltam a ser incomparáveis, que foi como os treze notebooks do
`colab-rl` acabaram.

A solução não escolhe entre as duas coisas: o pacote é a fonte, o notebook é gerado. Estes
testes são o que torna isso uma garantia em vez de uma intenção — se alguém editar a cópia
dentro de um notebook, `test_notebooks_are_in_sync_with_the_package` acusa e diz qual.
"""

import json
import os
import subprocess
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "tools"))

from gerar_notebooks import (  # noqa: E402
    MARCA_FIM,
    MARCA_INICIO,
    NOTEBOOKS,
    NUCLEO,
    fonte_combinada,
    monta_notebook,
)

CAMINHOS = [os.path.join(RAIZ, "notebooks", s["arquivo"]) for s in NOTEBOOKS]


def carrega(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def codigo_de(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def bloco_gerado(nb):
    for fonte in codigo_de(nb):
        if MARCA_INICIO in fonte:
            return fonte
    raise AssertionError("nenhuma célula gerada encontrada")


# --------------------------------------------------------------- existem e são válidos
@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_exists_and_is_valid_json(caminho):
    assert os.path.exists(caminho), "rode `python tools/gerar_notebooks.py`"
    nb = carrega(caminho)
    assert nb["nbformat"] == 4
    assert nb["cells"]


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_generated_code_compiles(caminho):
    """A prova de que o notebook não vai morrer na segunda célula no Colab."""
    fonte = bloco_gerado(carrega(caminho))
    compile(fonte, os.path.basename(caminho), "exec")


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_every_code_cell_compiles(caminho):
    for i, fonte in enumerate(codigo_de(carrega(caminho))):
        compile(fonte, f"{os.path.basename(caminho)}[{i}]", "exec")


# ---------------------------------------------------------------------- sincronia
def test_notebooks_are_in_sync_with_the_package():
    """Editar a cópia dentro do notebook não muda o pacote — e este teste acusa.

    É o que transforma "cópia idêntica" de intenção em garantia.
    """
    r = subprocess.run(
        [sys.executable, os.path.join(RAIZ, "tools", "gerar_notebooks.py"), "--check"],
        capture_output=True, text=True, cwd=RAIZ,
    )
    assert r.returncode == 0, (
        "notebooks fora de sincronia com o pacote:\n" + r.stdout + r.stderr
        + "\nRode `python tools/gerar_notebooks.py`."
    )


def test_every_notebook_embeds_the_same_core():
    """O núcleo — ambiente, avaliação, registro — tem que ser byte a byte igual.

    Se dois notebooks rodarem jogos ligeiramente diferentes, o gráfico da arena mente. É
    exatamente o defeito que este repositório existe para consertar.
    """
    nucleo = fonte_combinada(NUCLEO)
    for caminho in CAMINHOS:
        assert nucleo in bloco_gerado(carrega(caminho)), \
            f"{os.path.basename(caminho)} não contém o núcleo intacto"


def test_signature_is_recorded_in_metadata():
    for spec, caminho in zip(NOTEBOOKS, CAMINHOS):
        nb = carrega(caminho)
        meta = nb["metadata"]["snake_arena"]
        assert meta["assinatura"] == monta_notebook(spec)["metadata"]["snake_arena"]["assinatura"]
        assert set(NUCLEO).issubset(set(meta["gerado_de"]))


# ------------------------------------------------------------------- feitos p/ Colab
@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_is_configured_for_colab_gpu(caminho):
    nb = carrega(caminho)
    assert nb["metadata"]["accelerator"] == "GPU"
    assert "colab" in nb["metadata"]


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_has_no_repo_dependency(caminho):
    """Autocontido de verdade: nada de `git clone`, `pip install -e`, `import snakeai`."""
    fontes = codigo_de(carrega(caminho))
    junto = "\n".join(fontes)
    for proibido in ("git clone", "pip install -e", "import snakeai", "from snakeai"):
        assert proibido not in junto, f"{os.path.basename(caminho)} depende de {proibido!r}"


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_pins_the_keras_backend(caminho):
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert 'KERAS_BACKEND' in junto and 'tensorflow' in junto


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_offers_drive_and_resume(caminho):
    """A sessão do Colab cai. Sem retomada e sem Drive, o treino longo é inviável."""
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "USAR_DRIVE" in junto and "drive.mount" in junto
    assert 'retomar("last")' in junto


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_checks_the_contract_at_the_end(caminho):
    """O usuário tem que saber, sem sair do notebook, se a execução entra na arena."""
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "validate(" in junto and "arena" in junto.lower()


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_notebook_produces_gif_and_export(caminho):
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "render_episode" in junto
    assert "export_model" in junto


def test_the_generated_core_defines_what_the_notebook_uses():
    """Sanidade estática: os nomes que as células chamam existem no bloco gerado."""
    fonte = bloco_gerado(carrega(CAMINHOS[0]))
    for nome in ("class VecSnake", "def evaluate", "def verdict", "def format_verdict",
                 "def render_episode", "def export_model", "def validate",
                 "def plot_run", "class Recorder", "class AgentBase"):
        assert nome in fonte, f"o núcleo gerado não define {nome!r}"
