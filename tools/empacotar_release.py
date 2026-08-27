"""Monta os `.zip` que vão como anexo do Release.

Por que isto é um script e não uma linha de shell
-------------------------------------------------
Porque a linha de shell erra de dois jeitos silenciosos. `Compress-Archive` alimentado por
pipeline **achata a estrutura de pastas** — os 58 modelos viram 58 arquivos soltos com
nomes repetidos (`last.keras` vinte e nove vezes), e quem baixar não tem como saber de que
execução veio qual. E qualquer variação de `zip -r runs/` leva junto o que o `.gitignore`
manda ficar de fora e o que ele manda entrar, sem distinguir: o anexo passa a duplicar o
que já está no clone.

Aqui a divisão é a mesma do repositório, e ela é a razão de existirem **dois** arquivos:

* `modelos-<tag>.zip` — só os pesos (`.keras` e `.npz`). É o que **não** está no git, e é
  o único motivo pelo qual tirá-los de lá não perde nada.
* `runs-<tag>.zip` — a pasta de execução inteira, registro e GIFs inclusive. Redundante com
  o clone de propósito: é o instantâneo autossuficiente daquela tag, para quem quer os
  números de v0.1.0 depois que o `main` já andou.

Uso::

    python tools/empacotar_release.py v0.1.0-alpha
    python tools/empacotar_release.py v0.1.0-alpha --listar   # confere sem escrever nada
"""

from __future__ import annotations

import argparse
import os
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Os pesos: o que sai do git e entra no anexo. Ver o `.gitignore` e o §"O que entra no
#: git" do README — as duas listas têm de dizer a mesma coisa.
PESOS = (".keras", ".npz")

#: O registro: o que fica no git **e** vai no instantâneo.
REGISTRO = (".json", ".png", ".gif")


def arquivos(runs="runs", extensoes=PESOS):
    """Caminhos relativos à raiz, ordenados — a ordem estável mantém o zip reproduzível."""
    achados = []
    for base, _, nomes in os.walk(os.path.join(RAIZ, runs)):
        for nome in nomes:
            if nome.endswith(tuple(extensoes)):
                caminho = os.path.join(base, nome)
                achados.append(os.path.relpath(caminho, RAIZ).replace(os.sep, "/"))
    return sorted(achados)


def empacotar(destino, caminhos, verbose=True):
    """Escreve o `.zip` preservando `runs/<algo>/<variante>/seed<N>/...` dentro dele."""
    total = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in caminhos:
            z.write(os.path.join(RAIZ, rel), arcname=rel)
            total += os.path.getsize(os.path.join(RAIZ, rel))
    if verbose:
        mb = os.path.getsize(destino) / 1e6
        print(f"  {os.path.basename(destino)}: {len(caminhos)} arquivos, "
              f"{total / 1e6:.1f} MB → {mb:.1f} MB compactados")
    return destino


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", help="a tag do release, ex.: v0.1.0-alpha")
    ap.add_argument("--saida", default=RAIZ, help="onde escrever os .zip")
    ap.add_argument("--listar", action="store_true",
                    help="mostra o que entraria em cada arquivo, sem escrever")
    args = ap.parse_args(argv)

    pesos = arquivos(extensoes=PESOS)
    tudo = arquivos(extensoes=PESOS + REGISTRO)

    if not pesos:
        print("nenhum peso em runs/ — nada a anexar. Os modelos são gerados pelo treino, "
              "e uma pasta de execução sem eles é registro, não modelo.")
    if args.listar:
        print(f"modelos-{args.tag}.zip  ({len(pesos)} arquivos)")
        for c in pesos:
            print("   ", c)
        print(f"\nruns-{args.tag}.zip  ({len(tudo)} arquivos)")
        for c in tudo[:8]:
            print("   ", c)
        if len(tudo) > 8:
            print(f"    … mais {len(tudo) - 8}")
        return 0

    print("empacotando:")
    if pesos:
        empacotar(os.path.join(args.saida, f"modelos-{args.tag}.zip"), pesos)
    empacotar(os.path.join(args.saida, f"runs-{args.tag}.zip"), tudo)
    print("\nagora:")
    print(f"  gh release create {args.tag} --prerelease \\")
    print(f"    --title \"{args.tag} — a plataforma completa, a arena pela metade\" \\")
    print("    --notes-file docs/RELEASE_v0.1.0.md")
    print(f"  gh release upload {args.tag} modelos-{args.tag}.zip runs-{args.tag}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
