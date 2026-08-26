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
    """A sessão cai. Sem retomada e sem armazenamento que persiste, treino longo é inviável.

    E **sem knob**: a escolha do armazenamento é automática. Um parâmetro a mais na célula
    é uma coisa a mais para procurar e esquecer, e o valor certo dele é sempre o mesmo.
    """
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "drive.mount" in junto and "PASTA = pasta_de_trabalho()" in junto
    assert "USAR_DRIVE" not in junto, "o armazenamento não é mais configurado à mão"
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


# ------------------------------------------------- capacidade declarada
def test_the_capacity_table_matches_what_the_builders_produce():
    """A tabela do `COMPARABILITY.md` é computada, não escrita à mão.

    A arena iguala passos de ambiente e **não** iguala capacidade — 22× entre o menor e o
    maior. Uma tabela que envelhece em silêncio seria pior que nenhuma: ela afirmaria
    capacidade igualada onde não está. Se alguém mexer numa cabeça de rede, este teste
    falha e diz para rodar `python tools/tabela_parametros.py`.
    """
    from tabela_parametros import coleta, markdown            # noqa: PLC0415

    with open(os.path.join(RAIZ, "docs", "COMPARABILITY.md"), encoding="utf-8") as f:
        doc = f.read()
    assert markdown(coleta()) in doc, (
        "a tabela de capacidade divergiu — rode `python tools/tabela_parametros.py` e "
        "cole a saída no `docs/COMPARABILITY.md`"
    )


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
def test_storage_persists_by_default_without_being_asked(caminho):
    """O armazenamento que sobrevive à queda é o padrão, e não uma opção a marcar.

    A sessão cai — é questão de quando, não de se. Sem armazenamento persistente ela leva
    junto os checkpoints, e um treino de 5 M passos que caiu na terceira hora recomeça do
    zero. Deixar isso numa chave que o usuário tem que achar e ligar transforma um
    requisito em pegadinha.
    """
    junto = "\n".join(codigo_de(carrega(caminho)))
    assert "PASTA = pasta_de_trabalho()" in junto
    assert "/content/drive/MyDrive" in junto, "o caminho do Drive tem que estar no núcleo"


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


def test_the_acktr_ablation_is_the_same_agent_with_one_flag():
    """O `98` não pode ser um agente novo — se fosse, a diferença entre as duas curvas
    incluiria tudo o que divergiu entre as duas implementações.

    Os papéis se inverteram depois da medição: o `08` é o oficial, com a região de
    confiança calibrada por padrão, e o `98` é o braço de controle que volta ao alvo
    nominal."""
    oficial = next(s for s in NOTEBOOKS if s["arquivo"] == "08_acktr.ipynb")
    ablacao = next(s for s in NOTEBOOKS
                   if s["arquivo"] == "98_acktr_kl_nominal.ipynb")
    assert ablacao["agente"] == oficial["agente"] == "ACKTR"
    assert ablacao["modulos"] == oficial["modulos"]
    assert ablacao["extra_cfg"].strip().startswith("kl_calibrado=False,")

    codigo = "\n".join(codigo_de(carrega(
        os.path.join(RAIZ, "notebooks", ablacao["arquivo"]))))
    assert "kl_calibrado=False" in codigo
    oficial_codigo = "\n".join(codigo_de(carrega(
        os.path.join(RAIZ, "notebooks", oficial["arquivo"]))))
    assert "kl_calibrado" not in oficial_codigo.split("ASSINATURA_PACOTE")[-1], \
        "o 08 usa o padrão do pacote; nada de repetir a configuração na célula"


def test_the_two_acktr_notebooks_embed_byte_identical_code():
    """A única diferença permitida entre os dois é a linha de configuração."""
    a = bloco_gerado(carrega(os.path.join(RAIZ, "notebooks", "08_acktr.ipynb")))
    b = bloco_gerado(carrega(os.path.join(
        RAIZ, "notebooks", "98_acktr_kl_nominal.ipynb")))
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


# ------------------------------------------- o espaço de nomes achatado do notebook
#: `__all__` é declaração por módulo e some no achatamento sem consequência.
COLISOES_TOLERADAS = {"__all__"}


def _nomes_de_topo(caminho):
    import ast  # noqa: PLC0415

    with open(os.path.join(RAIZ, caminho), encoding="utf-8") as f:
        arv = ast.parse(f.read())
    nomes = []
    for no in arv.body:
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nomes.append((no.name, no))
        elif isinstance(no, ast.Assign):
            nomes += [(a.id, no) for a in no.targets if isinstance(a, ast.Name)]
    return nomes


def test_no_two_inlined_modules_define_the_same_name():
    """No notebook, os módulos viram **um espaço de nomes só** — e o último vence.

    Duas funções `resumo()` em módulos diferentes convivem em paz no pacote e viram um
    apagamento silencioso no notebook: a que for inlinada depois substitui a outra, sem
    erro, e quem chamava a primeira passa a chamar a segunda. Foi assim que
    `plataforma.resumo` e `nets/registry.resumo` colidiram.

    Constantes repetidas com o **mesmo valor** são toleradas — são duplicação, não
    ambiguidade. Com valores diferentes seria pior ainda, e o teste falha.
    """
    import ast  # noqa: PLC0415

    modulos = list(NUCLEO) + sorted({m for n in NOTEBOOKS for m in n["modulos"]})
    visto = {}
    conflitos = []
    for m in modulos:
        for nome, no in _nomes_de_topo(m):
            if nome in COLISOES_TOLERADAS:
                continue
            if nome in visto and visto[nome][0] != m:
                antes_m, antes_no = visto[nome]
                iguais = (isinstance(no, ast.Assign) and isinstance(antes_no, ast.Assign)
                          and ast.dump(no.value) == ast.dump(antes_no.value))
                if not iguais:
                    conflitos.append(f"{nome!r}: {antes_m} e {m}")
            visto.setdefault(nome, (m, no))
    assert not conflitos, (
        "nomes que colidem no notebook (o último inlinado apaga o primeiro):\n  "
        + "\n  ".join(conflitos))


def test_the_generator_refuses_a_renaming_relative_import():
    """`from ..x import y as z` some inteiro no achatamento, e `z` fica indefinido.

    Sem apelido não há problema: o nome importado é o mesmo que o módulo inlinado define.
    Com apelido, a ligação existia **só** no import — e o import é justamente o que sai.
    Deu `NameError` no fim de um treino de 5 M passos.
    """
    from gerar_notebooks import _limpa  # noqa: PLC0415

    with pytest.raises(ValueError, match="apelido"):
        _limpa("from ..plataforma import resumo as _r\nx = _r()\n", "fake.py")

    # sem apelido, segue sendo removido em silêncio, que é o certo
    assert "plataforma" not in _limpa("from ..plataforma import resumo\n", "fake.py")


# ------------------------------------------------------ os papers na tabela
#: Notebook → identificador arXiv do paper que **define** o algoritmo. Ablações deste
#: repositório (`97`, `98`, `99`) não têm paper e por isso não entram aqui.
PAPERS = {
    "01_ppo.ipynb": "1707.06347",
    "02_dqn.ipynb": "1312.5602",
    "03_rainbow.ipynb": "1710.02298",
    "04_a2c.ipynb": "1602.01783",
    "05_acer.ipynb": "1611.01224",
    "06_alphazero.ipynb": "1712.01815",
    "07_muzero.ipynb": "1911.08265",
    "08_acktr.ipynb": "1708.05144",
    "09_dreamerv3.ipynb": "2301.04104",
    "10_lbc.ipynb": "2305.05239",
    "11_soap.ipynb": "2407.18913",
    "12_acektr.ipynb": "1806.03884",
}

#: Ablações deste repositório: variam **um** parâmetro de um algoritmo já implementado.
#: Dar a elas o paper do algoritmo base sugeriria que a variação é do paper, e não é.
SEM_PAPER = {"94_rainbow_nstep3.ipynb",
             "95_a2c_orcamento_esparso.ipynb", "96_ppo_orcamento_esparso.ipynb",
             "97_ppo_canal_de_fome.ipynb", "98_acktr_kl_nominal.ipynb",
             "99_ablacoes.ipynb"}


def test_every_algorithm_notebook_is_classified_as_paper_or_ablation():
    """Nenhum notebook pode ficar de fora das duas listas sem alguém decidir em qual entra.

    Sem isto, um notebook novo entraria na tabela sem paper e sem ser ablação, e a coluna
    passaria a significar "às vezes tem link" — que não significa nada.
    """
    declarados = {s["arquivo"] for s in NOTEBOOKS}
    classificados = set(PAPERS) | SEM_PAPER
    assert declarados == classificados, (
        f"não classificados: {sorted(declarados - classificados)} · "
        f"classificados que não existem: {sorted(classificados - declarados)}")


def test_every_arxiv_link_in_the_readme_is_in_the_bibliography():
    """A tabela do README e `docs/REFERENCIAS.md` não podem divergir.

    São duas listas da mesma coisa, escritas em lugares diferentes — o arranjo que produz
    divergência silenciosa. Aqui a bibliografia é a superset: tudo que o README cita tem que
    estar lá, com o arquivo que implementa e o teste que prova. O contrário é permitido, e é
    o ponto: `docs/REFERENCIAS.md` cobre peças que não têm linha na tabela de algoritmos.
    """
    import re

    referencias = open(os.path.join(RAIZ, "docs", "REFERENCIAS.md"),
                       encoding="utf-8").read()
    no_readme = set(re.findall(r"arxiv\.org/abs/([\d.]+)", _readme()))
    na_biblio = set(re.findall(r"arxiv\.org/abs/([\d.]+)", referencias))
    faltando = sorted(no_readme - na_biblio)
    assert not faltando, f"citados no README e ausentes da bibliografia: {faltando}"


def test_the_bibliography_covers_every_notebook_paper():
    referencias = open(os.path.join(RAIZ, "docs", "REFERENCIAS.md"),
                       encoding="utf-8").read()
    for arquivo, arxiv in sorted(PAPERS.items()):
        assert arxiv in referencias, f"{arquivo}: {arxiv} não está em docs/REFERENCIAS.md"


@pytest.mark.parametrize("arquivo,arxiv", sorted(PAPERS.items()))
def test_the_algorithm_table_links_the_defining_paper(arquivo, arxiv):
    """Cada algoritmo aponta para o trabalho que o define, na mesma linha da tabela.

    Os identificadores foram conferidos um a um contra o abstract no arXiv — um ID trocado
    leva a um paper existente e plausível, que é o pior tipo de erro de citação: não quebra
    nada e ninguém confere.
    """
    readme = _readme()
    url = f"https://arxiv.org/abs/{arxiv}"
    assert url in readme, f"{arquivo}: falta o link para {url}"

    linha = next((l for l in readme.splitlines() if f"`{arquivo}`" in l), None)
    assert linha is not None
    assert url in linha, (
        f"{arquivo}: o paper existe no README mas não na linha da tabela — a coluna 📎 "
        "dessa linha está vazia ou aponta para outro trabalho")


@pytest.mark.parametrize("arquivo", sorted(SEM_PAPER))
def test_ablations_carry_no_paper(arquivo):
    """Uma ablação daqui com um link de paper leria como se a variação fosse do paper."""
    readme = _readme()
    linha = next((l for l in readme.splitlines() if f"`{arquivo}`" in l), None)
    assert linha is not None
    assert "arxiv.org" not in linha, f"{arquivo} é ablação deste repositório, não tem paper"


def test_every_arxiv_link_is_well_formed():
    """`abs`, nunca `pdf`: a página do abstract tem BibTeX, versões e a lista de citações.
    E `https`, porque `http://arxiv.org` redireciona e alguns leitores de markdown não
    seguem o redirecionamento."""
    import re  # noqa: PLC0415

    links = re.findall(r"https?://(?:www\.)?arxiv\.org/\S*?(?=[)\s])", _readme())
    assert links, "nenhum link do arXiv no README"
    for u in links:
        assert u.startswith("https://arxiv.org/abs/"), f"link mal formado: {u}"
        assert re.fullmatch(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}", u), \
            f"identificador do arXiv fora do formato: {u}"


@pytest.mark.parametrize("caminho", CAMINHOS, ids=[s["arquivo"] for s in NOTEBOOKS])
def test_the_notebook_carries_the_package_signature_as_a_constant(caminho):
    """A assinatura precisa ser legível **em tempo de execução**, não só num comentário:
    é ela que vira `meta["assinatura_pacote"]` no registro, no lugar do `commit` que o
    Kaggle não tem. Ver `docs/ANTES_DO_ARTIGO.md`."""
    nb = carrega(caminho)
    marca = nb["metadata"]["snake_arena"]["assinatura"]
    assert f'ASSINATURA_PACOTE = "{marca}"' in bloco_gerado(nb)
