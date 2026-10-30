"""
paso1_dividir_dataset.py
========================
Divide las imágenes originales en train / val / test (estratificado por clase):

    100 % de las imágenes
      ├── 15 % ............................................. TEST (nunca se toca al entrenar)
      └── 85 % restante
            ├── 15 % de ese 85 % ........................... VALIDACIÓN
            └── 85 % de ese 85 % ........................... ENTRENAMIENTO

Con 80 imágenes por clase:  entrenamiento 58 | validación 10 | prueba 12.

Uso:
    python paso1_dividir_dataset.py                       # usa MODO_SPLIT de config.py
    python paso1_dividir_dataset.py --modo secuencial

Si existe la carpeta dataset_test_externo/ (fotos con otra iluminación, altura, pieza, etc.),
se copia completa a dataset_split/test_externo/ para evaluar generalización real.
"""
import csv
import shutil
import random
import argparse

from config import *  # [IMPORT_LOCAL]
from utils_comunes import *  # [IMPORT_LOCAL]


def dividir_lista(archivos, modo, rng):
    n = len(archivos)
    n_test = int(round(n * FRAC_TEST))
    n_val = int(round((n - n_test) * FRAC_VAL))
    if modo == "secuencial":
        orden = sorted(archivos, key=clave_natural)
        n_train = n - n_test - n_val
        return orden[:n_train], orden[n_train:n_train + n_val], orden[n_train + n_val:]
    mezclado = list(archivos)
    rng.shuffle(mezclado)
    test = mezclado[:n_test]
    val = mezclado[n_test:n_test + n_val]
    train = mezclado[n_test + n_val:]
    return train, val, test


def main_dividir(modo=None):
    modo = modo or MODO_SPLIT
    if not DIR_ORIGINAL.exists():
        raise SystemExit(f"No existe la carpeta {DIR_ORIGINAL}. Crea 'dataset_original' junto a este script "
                         f"y copia ahí tus imágenes (normal_1.jpg, rayadura_1.jpg, ...).")

    por_clase, ignoradas = listar_imagenes_por_clase(DIR_ORIGINAL)
    for p in ignoradas:
        print(f"[AVISO] No se reconoce la clase de '{p.name}' (revisa ALIAS_CLASES en config.py). Se ignora.")
    for c in CLASES:
        if not por_clase[c]:
            raise SystemExit(f"No se encontraron imágenes de la clase '{c}'.")

    if DIR_SPLIT.exists():
        shutil.rmtree(DIR_SPLIT)       # carpeta derivada: se regenera siempre para no mezclar divisiones
    rng = random.Random(SEMILLA)
    filas = []
    print(f"\nModo de división: {modo}")
    print(f"{'clase':<12}{'total':>7}{'train':>7}{'val':>6}{'test':>6}")
    for c in CLASES:
        train, val, test = dividir_lista(por_clase[c], modo, rng)
        for nombre, lista in [("train", train), ("val", val), ("test", test)]:
            for p in lista:
                destino = DIR_SPLIT / nombre / c / p.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, destino)
                filas.append((nombre, c, p.name))
        print(f"{c:<12}{len(por_clase[c]):>7}{len(train):>7}{len(val):>6}{len(test):>6}")

    # Conjunto externo opcional
    if DIR_EXTERNO.exists():
        ext, _ = listar_imagenes_por_clase(DIR_EXTERNO)
        n_ext = 0
        for c in CLASES:
            for p in ext[c]:
                destino = DIR_SPLIT / "test_externo" / c / p.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, destino)
                filas.append(("test_externo", c, p.name))
                n_ext += 1
        print(f"\nConjunto externo detectado: {n_ext} imágenes -> dataset_split/test_externo/")

    DIR_RES.mkdir(parents=True, exist_ok=True)
    with open(DIR_RES / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "clase", "archivo"])
        w.writerows(filas)
    print(f"\nDivisión guardada en {DIR_SPLIT}\nLista de archivos: {DIR_RES / 'split_manifest.csv'}")


if __name__ == "__main__":  # [QUITAR]
    ap = argparse.ArgumentParser()  # [QUITAR]
    ap.add_argument("--modo", choices=["aleatorio", "secuencial"], default=None)  # [QUITAR]
    main_dividir(ap.parse_args().modo)  # [QUITAR]
