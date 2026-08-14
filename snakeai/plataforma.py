"""Onde o notebook está rodando — Colab, Kaggle ou máquina local.

Por que isto existe
-------------------
O mesmo `.ipynb` precisa rodar nos dois serviços gratuitos, e eles diferem exatamente nos
três pontos que decidem se um treino de horas sobrevive:

======================  ==============================  ==============================
                        Colab                           Kaggle
======================  ==============================  ==============================
pasta que persiste      Google Drive, montado à mão     ``/kaggle/working``, automático
retomar depois da queda  o Drive continua lá             anexar a saída da execução
                                                        anterior em ``/kaggle/input``
baixar o resultado      ``google.colab.files.download``  painel *Output*, sem código
======================  ==============================  ==============================

A detecção é por **capacidade observada**, não por variável de ambiente decorada: `colab`
só se o módulo `google.colab` existir de fato, `kaggle` só se `/kaggle/working` for
gravável. Um notebook rodando em qualquer outro lugar cai no caso `local` e continua
funcionando — o que também é o que faz a suíte de testes conseguir exercitar isto aqui.

O problema que o Kaggle resolve
-------------------------------
No Colab a sessão cai por inatividade e o teto de uso é opaco. O Kaggle tem cota semanal
de GPU declarada e um caminho **headless**: *Save Version → Save & Run All* roda o notebook
inteiro sem aba aberta, e a saída vira um artefato versionado. Para um treino de 5 M passos
que leva ~40 minutos, isso é a diferença entre "torcer para não cair" e "enfileirar e
buscar depois".

A contrapartida é que `/kaggle/working` **não** volta sozinho na sessão seguinte: ele vira
a *saída* daquela versão. Para continuar de onde parou, anexe a saída anterior como
entrada (*Add Input → Your Work → Notebook Output*) e `semear_checkpoints` faz o resto.
"""

from __future__ import annotations

import os
import shutil

__all__ = ["detecta", "pasta_de_trabalho", "semear_checkpoints", "entregar_arquivo",
           "resumo_plataforma", "COLAB", "KAGGLE", "LOCAL"]

COLAB, KAGGLE, LOCAL = "colab", "kaggle", "local"


def detecta():
    """`"colab"`, `"kaggle"` ou `"local"`, por capacidade observada."""
    try:
        import google.colab  # noqa: F401,PLC0415
        return COLAB
    except Exception:
        pass
    if os.path.isdir("/kaggle/working") and os.access("/kaggle/working", os.W_OK):
        return KAGGLE
    return LOCAL


def pasta_de_trabalho(usar_drive="auto", nome="snake-arena", verbose=True):
    """A pasta onde checkpoints, `runs/` e export vão viver. **Sem nada para configurar.**

    `usar_drive="auto"` é o padrão e resolve tudo sozinho: no Colab tenta montar o Drive,
    no Kaggle usa `/kaggle/working`, no local usa o diretório atual. Passar `True` ou
    `False` força o comportamento no Colab e não faz diferença nos outros dois — o mesmo
    notebook roda nos três sem editar célula, que é o ponto.

    Se a montagem do Drive falhar (o usuário recusa a autorização, ou a sessão não tem
    navegador), cai para `/content` **avisando alto**: ali o treino roda, mas a queda da
    sessão leva os checkpoints junto, e descobrir isso depois de três horas é pior do que
    ler um aviso agora.
    """
    onde = detecta()

    if onde == COLAB and usar_drive is not False:
        try:
            from google.colab import drive  # noqa: PLC0415

            drive.mount("/content/drive")
            raiz = os.path.join("/content/drive/MyDrive", nome)
        except Exception as e:
            if usar_drive is True:
                raise
            print(f"AVISO: não consegui montar o Drive ({type(e).__name__}: {e}).")
            print("       Usando /content, que NÃO sobrevive à queda da sessão —")
            print("       se o treino cair, ele recomeça do zero.")
            raiz = os.path.join("/content", nome)
    elif onde == COLAB:
        raiz = os.path.join("/content", nome)
    elif onde == KAGGLE:
        raiz = os.path.join("/kaggle/working", nome)
    else:
        raiz = os.path.abspath(nome)

    os.makedirs(raiz, exist_ok=True)
    if verbose:
        print(f"plataforma: {onde} · pasta: {raiz}")
        if onde == KAGGLE:
            print("  lembre: /kaggle/working vira a SAÍDA desta versão. Para continuar "
                  "depois,\n  anexe esta saída como entrada da próxima execução.")
    return raiz


def semear_checkpoints(ckpt_dir, verbose=True):
    """Traz checkpoints de execuções anteriores anexadas em `/kaggle/input`.

    É isto que faz "retomar" funcionar no Kaggle. A sessão nova nasce com
    `/kaggle/working` vazio; o que sobreviveu está montado **somente leitura** em
    `/kaggle/input/<algum-nome>/`. Copiamos para `ckpt_dir` só o que ainda não existe lá —
    um checkpoint desta sessão sempre vence o de uma anterior, senão retomar andaria para
    trás.

    Devolve a lista do que foi copiado. Fora do Kaggle, lista vazia e nenhum efeito.
    """
    if detecta() != KAGGLE or not os.path.isdir("/kaggle/input"):
        return []

    os.makedirs(ckpt_dir, exist_ok=True)
    copiados = []
    for raiz, _, arquivos in os.walk("/kaggle/input"):
        if os.path.basename(raiz) != "checkpoints":
            continue
        for nome in arquivos:
            if not nome.endswith((".keras", ".json")):
                continue
            destino = os.path.join(ckpt_dir, nome)
            if os.path.exists(destino):
                continue
            shutil.copyfile(os.path.join(raiz, nome), destino)
            copiados.append(destino)

    if verbose and copiados:
        print(f"  [retomada] {len(copiados)} arquivo(s) de checkpoint vieram de "
              f"/kaggle/input")
    return copiados


def entregar_arquivo(caminho, verbose=True):
    """Entrega o arquivo ao usuário, do jeito que a plataforma permite.

    No Colab dispara o download pelo navegador — que só funciona com a aba aberta. No
    Kaggle não há o que disparar: o que está em `/kaggle/working` aparece sozinho no painel
    *Output*, e é justamente por isso que o Kaggle aguenta execução headless. No local, o
    arquivo já está no disco.

    Devolve `True` só quando um download foi realmente disparado.
    """
    onde = detecta()
    if onde == COLAB:
        try:
            from google.colab import files  # noqa: PLC0415

            files.download(caminho)
            return True
        except Exception as e:                       # aba fechada, sessão sem navegador
            if verbose:
                print(f"download automático não rolou ({type(e).__name__}: {e})")
    elif onde == KAGGLE and verbose:
        print("no Kaggle não há download automático: o arquivo já está no painel "
              "**Output**,\nà direita, e é baixável de lá mesmo com a aba fechada.")
    if verbose:
        print(f"arquivo: {caminho}")
    return False


def resumo_plataforma():
    """Dicionário com plataforma e aceleradores visíveis — vai para o `meta` do registro.

    O nome é longo de propósito. No notebook gerado todos os módulos viram **um espaço de
    nomes só**, e um `resumo()` aqui colidiria com o `resumo()` de `snakeai/nets/registry.py`
    — o último inlinado venceria e o outro sumiria sem erro nenhum.
    `tests/test_notebooks.py::test_no_two_inlined_modules_define_the_same_name` tranca isso.
    """
    info = {"plataforma": detecta()}
    try:
        import tensorflow as tf  # noqa: PLC0415

        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]
        info["n_gpus"] = len(gpus)
    except Exception:
        info["gpus"], info["n_gpus"] = [], 0
    return info
