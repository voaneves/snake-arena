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


def test_multiline_relative_imports_are_stripped_whole():
    """O import relativo pode ocupar várias linhas sem o parêntese sozinho no fim.

    A versão anterior do `_limpa` olhava se a linha terminava em `(`; com este formato ela
    apagava só a primeira linha e deixava a segunda órfã, e o notebook nascia com
    `IndentationError`. É o formato que `snakeai/kfac.py` e `snakeai/agents/dreamerv3.py`
    usam, então isto não é hipotético.
    """
    from gerar_notebooks import _limpa  # noqa: PLC0415

    fonte = (
        "from ..kfac import (KFac, captura_kfac,\n"
        "                    perda_fisher_gaussiana)\n"
        "from .base import AgentBase\n"
        "\n"
        "X = 1\n"
    )
    limpo = _limpa(fonte, "fake.py")
    assert "perda_fisher_gaussiana" not in limpo
    assert "AgentBase" not in limpo
    assert "X = 1" in limpo
    compile(limpo, "fake.py", "exec")


def test_absolute_imports_survive():
    """Só os relativos saem — `import numpy as np` tem que continuar lá."""
    from gerar_notebooks import _limpa  # noqa: PLC0415

    limpo = _limpa("import numpy as np\nfrom dataclasses import (dataclass,\n    field)\n",
                   "fake.py")
    assert "numpy" in limpo and "dataclass" in limpo and "field" in limpo


# --------------------------------------------------- o README contra a pasta
def _readme():
    with open(os.path.join(RAIZ, "README.md"), encoding="utf-8") as f:
        return f.read()


def test_readme_never_links_a_notebook_that_does_not_exist():
    """Um badge do Colab apontando para um arquivo inexistente abre uma página de erro.

    Isto não é hipotético: a tabela de badges apontava para `00_arena.ipynb` e
    `99_ablation_redes.ipynb`, dois arquivos que nunca existiram neste repositório, e
    ninguém percebeu porque badge quebrado só quebra quando alguém clica.
    """
    import re  # noqa: PLC0415

    citados = set(re.findall(r"notebooks/([\w.]+\.ipynb)", _readme()))
    assert citados, "o README não cita nenhum notebook"
    faltando = sorted(n for n in citados
                      if not os.path.exists(os.path.join(RAIZ, "notebooks", n)))
    assert not faltando, f"o README aponta para notebooks que não existem: {faltando}"


@pytest.mark.parametrize("spec", NOTEBOOKS, ids=lambda s: s["arquivo"])
def test_every_notebook_has_a_colab_badge(spec):
    """Todo notebook gerado tem que estar na tabela de badges — senão ele existe no
    repositório e não existe para quem lê o README, que é o mesmo que não existir."""
    readme = _readme()
    alvo = f"blob/main/notebooks/{spec['arquivo']}"
    assert alvo in readme, f"{spec['arquivo']} não tem badge do Colab no README"


@pytest.mark.parametrize("spec", NOTEBOOKS, ids=lambda s: s["arquivo"])
def test_every_notebook_is_in_the_algorithm_table(spec):
    readme = _readme()
    assert f"`{spec['arquivo']}`" in readme, \
        f"{spec['arquivo']} não aparece na tabela de algoritmos do README"


def test_the_notebooks_folder_has_exactly_what_the_generator_declares():
    """Nada de notebook órfão na pasta: se está lá e o gerador não o conhece, ele não é
    verificado por nenhum teste de sincronia e vai apodrecer em silêncio."""
    na_pasta = {f for f in os.listdir(os.path.join(RAIZ, "notebooks"))
                if f.endswith(".ipynb")}
    declarados = {s["arquivo"] for s in NOTEBOOKS}
    assert na_pasta == declarados, (
        f"órfãos na pasta: {sorted(na_pasta - declarados)}; "
        f"declarados e ausentes: {sorted(declarados - na_pasta)}")


# ------------------------------------------------------- o download no fim
def _celula_download(caminho):
    for c in carrega(caminho)["cells"]:
        if c["cell_type"] == "code" and "Baixar tudo" in "".join(c["source"]):
            return "".join(c["source"])
    return None


@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_every_notebook_ends_by_packing_the_run(caminho):
    """Sem esta célula, o Colab termina o treino e o resultado morre com a sessão.

    Era o que acontecia: a pasta da execução ficava no `/content`, e quem não lembrasse de
    baixar à mão perdia horas de GPU quando a máquina caísse.
    """
    src = _celula_download(caminho)
    assert src is not None, "notebook sem a célula de download"
    assert "make_archive" in src and "entregar_arquivo" in src


@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_the_download_cell_survives_outside_colab(caminho, tmp_path):
    """Ela roda de verdade aqui, onde `google.colab` não existe.

    Este é o caso que mais importa: sessão desconectada, aba fechada, ou execução fora do
    Colab. O `.zip` tem que existir de qualquer forma — o download automático é conveniência,
    o arquivo é o resultado.
    """
    import types  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    pasta = tmp_path / "snake-arena"
    execucao = pasta / "runs" / "dqn" / "base" / "seed0"
    execucao.mkdir(parents=True)
    (pasta / "export").mkdir()
    for nome in ("history.json", "curva.png", "episodio_s7.gif"):
        (execucao / nome).write_text("x")
    (pasta / "export" / "modelo.tflite").write_text("y")

    from snakeai.plataforma import entregar_arquivo  # noqa: PLC0415

    registro = types.SimpleNamespace(
        record=types.SimpleNamespace(algo="dqn", variant="base", seed=0))
    escopo = {"os": os, "PASTA": str(pasta), "registro": registro,
              "entregar_arquivo": entregar_arquivo,
              "CAMINHO_REGISTRO": str(execucao / "history.json")}
    exec(compile(_celula_download(caminho), "celula", "exec"), escopo)

    dentro = set(zipfile.ZipFile(escopo["ZIP"]).namelist())
    assert {"history.json", "curva.png", "episodio_s7.gif"} <= dentro
    assert "export/modelo.tflite" in dentro, \
        "o modelo exportado mora fora da pasta da execução e tem que ser copiado para dentro"


@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_the_run_path_is_captured_before_it_is_zipped(caminho):
    """A célula de contrato tem que guardar o caminho em vez de só imprimi-lo — senão a
    célula seguinte não tem como saber que pasta compactar."""
    codigo = "\n".join(codigo_de(carrega(caminho)))
    assert "CAMINHO_REGISTRO = registro.save" in codigo


@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_drive_is_on_by_default(caminho):
    """`USAR_DRIVE = True` é o padrão, e isso não é preferência de estilo.

    A sessão do Colab cai — é questão de quando, não de se. Sem o Drive ela leva junto os
    checkpoints, e um treino de 5 M passos que caiu na terceira hora recomeça do zero em
    vez de retomar. O custo de ligar é uma tela de autorização; o de não ligar é a
    execução inteira.
    """
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "USAR_DRIVE = True" in junto


# ------------------------------------------- last e best, os dois lados do resultado
@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_notebook_reports_both_the_last_and_the_best_model(caminho):
    """RL profundo não melhora monotonicamente: a execução pode terminar pior do que já
    esteve. Reportar só o último esconde isso; reportar só o melhor premia o ruído."""
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "modelo_melhor()" in junto, "o notebook nunca carrega o melhor checkpoint"
    assert 'registro.record.melhor' in junto


@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_notebook_exports_both_models_to_separate_folders(caminho):
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert '"export", "last"' in junto and '"export", "best"' in junto


def test_the_calibrated_acktr_notebook_is_the_same_agent_with_one_flag():
    """O `98` não pode ser um agente novo — se fosse, a diferença entre as duas curvas
    incluiria tudo o que divergiu entre as duas implementações."""
    base = next(s for s in NOTEBOOKS if s["arquivo"] == "08_acktr.ipynb")
    cal = next(s for s in NOTEBOOKS
               if s["arquivo"] == "98_acktr_kl_max_corrigido.ipynb")
    assert cal["agente"] == base["agente"] == "ACKTR"
    assert cal["modulos"] == base["modulos"]
    assert cal["extra_cfg"].strip() == "kl_calibrado=True,"

    codigo = "\n".join(codigo_de(carrega(
        os.path.join(RAIZ, "notebooks", cal["arquivo"]))))
    assert "kl_calibrado=True" in codigo
    base_codigo = "\n".join(codigo_de(carrega(
        os.path.join(RAIZ, "notebooks", base["arquivo"]))))
    assert "kl_calibrado=True" not in base_codigo, \
        "o 08 é o controle: tem que continuar sem a calibração"


def test_the_two_acktr_notebooks_embed_byte_identical_code():
    """A única diferença permitida entre os dois é a linha de configuração."""
    a = bloco_gerado(carrega(os.path.join(RAIZ, "notebooks", "08_acktr.ipynb")))
    b = bloco_gerado(carrega(os.path.join(
        RAIZ, "notebooks", "98_acktr_kl_max_corrigido.ipynb")))
    assert a == b


# --------------------------------------------------- Colab e Kaggle, o mesmo arquivo
@pytest.mark.parametrize("caminho", CAMINHOS, ids=lambda c: os.path.basename(c))
def test_notebook_runs_on_colab_and_on_kaggle(caminho):
    """Um `.ipynb` só para os dois serviços.

    A quota gratuita do Colab não sustenta 27 execuções de 5 M passos; a do Kaggle é maior
    e tem execução headless. Manter dois arquivos diferentes traria de volta exatamente o
    problema que este repositório existe para consertar — duas cópias que divergem.
    """
    nb = carrega(caminho)
    junto = "\n".join(codigo_de(nb))
    assert "pasta_de_trabalho(" in junto, "a pasta tem que ser escolhida por detecção"
    assert "semear_checkpoints(" in junto, "sem isto o Kaggle não retoma"
    assert "entregar_arquivo(" in junto

    # Nas células escritas à mão, nada de caminho ou import de uma plataforma só. O bloco
    # gerado fica de fora porque é lá que `plataforma.py` mora, e é justamente o módulo
    # cujo trabalho é conhecer os dois casos.
    # Comentários fora: eles *explicam* as duas plataformas, e explicar é o objetivo.
    # O que não pode é código com caminho fixo.
    mao = "\n".join(
        l for f in codigo_de(nb) if MARCA_INICIO not in f
        for l in f.splitlines() if not l.lstrip().startswith("#"))
    for proibido in ("/content/drive", "/kaggle/working", "from google.colab import"):
        assert proibido not in mao, (
            f"{os.path.basename(caminho)} tem {proibido!r} em código: a escolha da "
            "plataforma é do `snakeai/plataforma.py`, não da célula")
    assert "PASTA = pasta_de_trabalho(" in mao


def test_the_platform_module_is_part_of_the_shared_core():
    """Se cada notebook trouxesse a própria detecção, elas divergiriam."""
    assert "snakeai/plataforma.py" in NUCLEO
