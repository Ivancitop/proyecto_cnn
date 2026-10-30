"""
construir_pipeline_completo.py
==============================
Genera 'pipeline_completo.py': un ÚNICO archivo con todo el código (config + utilidades +
pasos 1 a 5), útil para entregar o ejecutar de una sola vez.
Si cambias algún archivo individual, vuelve a ejecutar:  python construir_pipeline_completo.py
"""
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MODULOS = ["config", "utils_comunes", "paso1_dividir_dataset", "paso2_preprocesamiento",
           "paso3_linea_base", "paso4_entrenar_cnn", "paso5_evaluar_errores"]

CABECERA = '''"""
pipeline_completo.py  (archivo generado automáticamente por construir_pipeline_completo.py)
=====================================================================================
Inspección visual con CNN - todo el flujo en un solo archivo.

Estructura de carpetas esperada junto a este archivo:
    dataset_original/   <- tus imágenes: normal_1.jpg, rayadura_1.jpg, defectuosa_1.jpg ...
    dataset_test_externo/  (opcional) imágenes con otras condiciones, solo para prueba

Uso:
    python pipeline_completo.py                    # pasos 1 a 5
    python pipeline_completo.py --estudio          # además, estudio de hiperparámetros
    python pipeline_completo.py --desde 4          # reanuda desde el paso 4 (CNN)
    python pipeline_completo.py --seleccionar-roi  # solo elegir la ROI con el ratón
"""
import argparse
'''

RUNNER = '''

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline completo de inspección con CNN")
    ap.add_argument("--desde", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="1 dividir, 2 preprocesar, 3 línea base, 4 CNN, 5 evaluación de errores")
    ap.add_argument("--estudio", action="store_true", help="ejecuta además el estudio de hiperparámetros")
    ap.add_argument("--modo-split", choices=["aleatorio", "secuencial"], default=None)
    ap.add_argument("--seleccionar-roi", action="store_true")
    args = ap.parse_args()

    if args.seleccionar_roi:
        main_preprocesar(seleccionar_roi=True)
        raise SystemExit
    pasos = [(1, "DIVIDIR DATASET", lambda: main_dividir(args.modo_split)),
             (2, "PREPROCESAMIENTO", main_preprocesar),
             (3, "LÍNEA BASE CONVENCIONAL", main_linea_base),
             (4, "ENTRENAR CNN", main_entrenar),
             (5, "EVALUAR TIPOS DE FALLO", main_evaluar)]
    for n, nombre, fn in pasos:
        if n < args.desde:
            continue
        print("\\n" + "=" * 78 + f"\\n PASO {n}: {nombre}\\n" + "=" * 78)
        fn()
        if n == 4 and args.estudio:
            print("\\n" + "=" * 78 + "\\n ESTUDIO DE HIPERPARÁMETROS\\n" + "=" * 78)
            main_estudio()
'''


def construir():
    partes = [CABECERA]
    for nombre in MODULOS:
        lineas = (AQUI / f"{nombre}.py").read_text(encoding="utf-8").splitlines()
        limpias = [l for l in lineas if "[IMPORT_LOCAL]" not in l and "[QUITAR]" not in l]
        partes.append(f"\n\n# {'=' * 77}\n# >>> {nombre}.py\n# {'=' * 77}\n" + "\n".join(limpias))
    partes.append(RUNNER)
    destino = AQUI / "pipeline_completo.py"
    destino.write_text("\n".join(partes), encoding="utf-8")
    print(f"Generado {destino}")


if __name__ == "__main__":
    construir()
