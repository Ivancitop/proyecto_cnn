"""
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



# =============================================================================
# >>> config.py
# =============================================================================
"""
config.py
=========
Configuración central del proyecto de inspección visual con CNN.
Todos los parámetros que probablemente quieras cambiar están aquí.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SEMILLA = 42

# ----------------------------------------------------------------------------
# CARPETAS
# ----------------------------------------------------------------------------
DIR_ORIGINAL = RAIZ / "dataset_original"       # <- aquí van tus imágenes: normal_1.jpg, rayadura_1.jpg ...
DIR_EXTERNO = RAIZ / "dataset_test_externo"    # <- (opcional) imágenes con otras condiciones, solo para prueba
DIR_SPLIT = RAIZ / "dataset_split"             # se genera: train / val / test (imágenes originales)
DIR_PROC = RAIZ / "dataset_proc"               # se genera: imágenes ya recortadas y redimensionadas
DIR_RES = RAIZ / "resultados"                  # se genera: gráficas, métricas, modelos

# ----------------------------------------------------------------------------
# CLASES
# El nombre del archivo define la clase: "normal_12.jpg" -> normal.
# Si tus archivos usan otro prefijo, agrégalo en ALIAS_CLASES (sin importar acentos ni mayúsculas).
# ----------------------------------------------------------------------------
CLASES = ["normal", "rayadura", "defectuosa"]
ALIAS_CLASES = {
    "normal": ["normal"],
    "rayadura": ["rayadura", "rayaduras", "rayon", "rayones", "raya", "rayado"],
    "defectuosa": ["defectuosa", "defectuoso", "defecto", "defectos", "rota", "roto",
                   "rotura", "deformacion", "deformada"],
}
NOMBRES_MOSTRAR = {"normal": "Normal", "rayadura": "Rayadura", "defectuosa": "Defectuosa (rota)"}

# ----------------------------------------------------------------------------
# DIVISIÓN DEL DATASET
#   1) 15 % del total -> test
#   2) del 85 % restante, 15 % -> validación y el resto -> entrenamiento
# MODO_SPLIT:
#   "aleatorio"  : mezcla al azar (por clase)
#   "secuencial" : respeta el orden de captura (train = primeras, val = intermedias, test = últimas).
#                  Útil si tomaste las fotos en ráfaga: evita que fotos casi idénticas
#                  caigan en train y en test a la vez.
# ----------------------------------------------------------------------------
FRAC_TEST = 0.15
FRAC_VAL = 0.15
MODO_SPLIT = "aleatorio"

# ----------------------------------------------------------------------------
# PREPROCESAMIENTO
# ROI_MODO: "auto"    -> detecta la pieza por umbralización (revisa resultados/preprocesamiento/ejemplos_roi.png)
#           "manual"  -> usa ROI_MANUAL = (x, y, ancho, alto) en píxeles de la imagen original
#                        (obtenlo con:  python paso2_preprocesamiento.py --seleccionar-roi)
#           "ninguno" -> usa la imagen completa
# ----------------------------------------------------------------------------
ROI_MODO = "auto"
ROI_MANUAL = None
MARGEN_ROI = 0.08          # margen extra alrededor de la pieza (fracción)
IMG_BASE = 128             # tamaño (lado) con el que se guardan las imágenes procesadas
USAR_GRIS = False          # True -> escala de grises (1 canal); False -> color (3 canales)

# Criterios para descartar imágenes inválidas
MIN_LADO = 200             # px mínimos de ancho/alto
MIN_DESV = 5.0             # desviación estándar mínima (descarta imágenes casi uniformes)
MIN_MEDIA, MAX_MEDIA = 8, 247   # brillo medio mínimo/máximo (sub/sobreexposición)
UMBRAL_NITIDEZ = 3.0       # varianza del Laplaciano mínima (descarta imágenes muy desenfocadas)

# Data augmentation (solo en entrenamiento)
AUG = dict(rotacion=10, traslacion=0.05, zoom=(0.9, 1.1),
           brillo=0.25, contraste=0.25, ruido_std=0.02)

# ----------------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------------
CNN_BASE = dict(
    img_size=128,      # tamaño de entrada (<= IMG_BASE recomendado)
    n_filtros=16,      # filtros de la 1.ª capa (se duplican en cada capa)
    n_capas=3,         # bloques convolucionales
    lr=1e-3,
    batch_size=16,
    epocas=40,
    dropout=0.3,
    paciencia=12,      # early stopping sobre val_loss (None = desactivado)
    weight_decay=1e-4,
)

# Estudio de hiperparámetros: se varía UN factor a la vez manteniendo el resto en CNN_BASE
GRID = {
    "img_size": [64, 96, 128],
    "n_filtros": [8, 16, 32],
    "n_capas": [2, 3, 4],
    "lr": [1e-2, 1e-3, 1e-4],
    "batch_size": [8, 16, 32],
    "epocas": [10, 20, 40],
}
N_SEMILLAS_ESTUDIO = 2     # repeticiones por configuración (más = más fiable, más lento)


# =============================================================================
# >>> utils_comunes.py
# =============================================================================
"""
utils_comunes.py
================
Funciones compartidas: lectura de imágenes, ROI, preprocesamiento, perturbaciones,
métricas y gráficas.
"""
import re
import json
import time
import random
import unicodedata
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


EXT_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Generales
# ----------------------------------------------------------------------------
def fijar_semilla(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch
        torch.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
    except ImportError:
        pass


def etiquetas_mostrar():
    return [NOMBRES_MOSTRAR.get(c, c) for c in CLASES]


def clave_natural(p):
    """Ordena normal_2 antes que normal_10."""
    nums = re.findall(r"\d+", Path(p).stem)
    return int(nums[-1]) if nums else 0


def normalizar_texto(txt):
    nfkd = unicodedata.normalize("NFKD", txt.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detectar_clase(nombre_archivo):
    """Devuelve la clase según el prefijo del nombre (normal_1.jpg -> 'normal') o None."""
    base = normalizar_texto(Path(nombre_archivo).stem)
    m = re.match(r"^([a-z]+)", base)
    if not m:
        return None
    prefijo = m.group(1)
    for clase, alias in ALIAS_CLASES.items():
        if prefijo in [normalizar_texto(a) for a in alias]:
            return clase
    return None


def listar_imagenes_por_clase(carpeta):
    por_clase = {c: [] for c in CLASES}
    ignoradas = []
    for p in sorted(Path(carpeta).rglob("*"), key=lambda q: (q.parent.as_posix(), clave_natural(q))):
        if p.suffix.lower() not in EXT_VALIDAS:
            continue
        clase = detectar_clase(p.name)
        if clase is None:
            ignoradas.append(p)
        else:
            por_clase[clase].append(p)
    return por_clase, ignoradas


def leer_imagen(ruta, color=True):
    """Lectura robusta a rutas con acentos/espacios (Windows)."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    if datos.size == 0:
        return None
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(datos, flag)


def guardar_imagen(ruta, img):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ruta.suffix or ".png", img)
    if not ok:
        raise IOError(f"No se pudo codificar {ruta}")
    buf.tofile(str(ruta))


def medir_tiempo_ms(fn, n=50, calentamiento=5):
    for _ in range(calentamiento):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(np.std(t))


# ----------------------------------------------------------------------------
# Validación, ROI y preprocesamiento
# ----------------------------------------------------------------------------
def es_imagen_valida(img):
    """(True, 'ok') o (False, motivo)."""
    if img is None or img.size == 0:
        return False, "ilegible o corrupta"
    h, w = img.shape[:2]
    if min(h, w) < MIN_LADO:
        return False, f"muy pequeña ({w}x{h})"
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    esc = 480 / max(h, w)
    if esc < 1:
        gris = cv2.resize(gris, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    if gris.std() < MIN_DESV:
        return False, "sin contenido (casi uniforme)"
    media = gris.mean()
    if media < MIN_MEDIA:
        return False, "subexpuesta"
    if media > MAX_MEDIA:
        return False, "sobreexpuesta"
    if cv2.Laplacian(gris, cv2.CV_64F).var() < UMBRAL_NITIDEZ:
        return False, "desenfocada"
    return True, "ok"


def mascara_pieza(gris):
    """Umbralización de Otsu; el fondo se asume como lo que domina en los bordes de la imagen."""
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    if borde.mean() > 127:
        m = 255 - m
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def hacer_cuadrado(caja, ancho, alto, margen):
    """Convierte (x, y, w, h) en un cuadrado centrado con margen, dentro de la imagen."""
    x, y, w, h = caja
    cx, cy = x + w / 2, y + h / 2
    lado = max(w, h) * (1 + 2 * margen)
    lado = min(lado, min(ancho, alto))
    x0 = int(round(min(max(cx - lado / 2, 0), ancho - lado)))
    y0 = int(round(min(max(cy - lado / 2, 0), alto - lado)))
    lado = int(round(lado))
    return (x0, y0, lado, lado)


def calcular_roi(img):
    """Devuelve la región de interés cuadrada (x, y, lado, lado) en píxeles de la imagen original."""
    alto, ancho = img.shape[:2]
    if ROI_MODO == "ninguno":
        return hacer_cuadrado((0, 0, ancho, alto), ancho, alto, 0)
    if ROI_MODO == "manual" and ROI_MANUAL is not None:
        return hacer_cuadrado(ROI_MANUAL, ancho, alto, 0)

    esc = min(1.0, 480 / max(alto, ancho))
    peq = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else img
    gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(mascara_pieza(gris), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caja = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 0.05 * peq.shape[0] * peq.shape[1]:
            caja = tuple(int(round(v / esc)) for v in (x, y, w, h))
    if caja is None:   # respaldo: recorte central del 60 %
        lado = int(0.6 * min(alto, ancho))
        caja = ((ancho - lado) // 2, (alto - lado) // 2, lado, lado)
    return hacer_cuadrado(caja, ancho, alto, MARGEN_ROI)


def preprocesar_imagen(img, roi=None, tam=None):
    """Recorta ROI -> redimensiona -> (opcional) escala de grises. Devuelve uint8."""
    tam = tam or IMG_BASE
    x, y, w, h = roi if roi is not None else calcular_roi(img)
    recorte = img[y:y + h, x:x + w]
    recorte = cv2.resize(recorte, (tam, tam), interpolation=cv2.INTER_AREA)
    if USAR_GRIS:
        recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    return recorte


def cargar_procesada(ruta):
    """Carga una imagen ya procesada (uint8; BGR o gris según USAR_GRIS)."""
    return leer_imagen(ruta, color=not USAR_GRIS)


def listar_split(split, base=None):
    """[(ruta, indice_clase), ...] de dataset_proc/<split>/<clase>/*.png"""
    base = Path(base) if base else DIR_PROC
    items = []
    for i, c in enumerate(CLASES):
        carpeta = base / split / c
        if carpeta.exists():
            for p in sorted(carpeta.glob("*.png"), key=clave_natural):
                items.append((p, i))
    return items


def cargar_normalizacion():
    with open(DIR_PROC / "normalizacion.json", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Perturbaciones para probar robustez (se aplican sobre imágenes procesadas uint8)
# ----------------------------------------------------------------------------
PERTURBACIONES = ["original", "brillo_bajo", "brillo_alto", "contraste_bajo",
                  "ruido", "rotacion_15", "desenfoque", "zoom_in"]


def perturbar(img, tipo, semilla=0):
    if tipo == "original":
        return img.copy()
    f = img.astype(np.float32)
    h, w = img.shape[:2]
    if tipo == "brillo_bajo":
        f = f * 0.6
    elif tipo == "brillo_alto":
        f = f * 1.4
    elif tipo == "contraste_bajo":
        f = (f - f.mean()) * 0.5 + f.mean()
    elif tipo == "ruido":
        f = f + np.random.RandomState(semilla).normal(0, 15, f.shape)
    elif tipo == "rotacion_15":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif tipo == "desenfoque":
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    elif tipo == "zoom_in":
        m = int(0.12 * min(h, w))
        return cv2.resize(img[m:h - m, m:w - m], (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(tipo)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Métricas y gráficas
# ----------------------------------------------------------------------------
def calcular_metricas(y_true, y_pred):
    labels = list(range(len(CLASES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "n": int(len(y_true)),
        "por_clase": {CLASES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "soporte": int(s[i])} for i in labels},
    }


def graficar_matriz_confusion(y_true, y_pred, titulo, ruta):
    """Matriz de confusión: conteos (izq.) y porcentaje por clase real (der.). Devuelve la matriz."""
    labels = list(range(len(CLASES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    suma = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, suma, out=np.zeros_like(cm, dtype=float), where=suma > 0) * 100
    nombres = etiquetas_mostrar()

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, fmt, sub in [(ejes[0], cm, "d", "Conteo"), (ejes[1], cmn, ".0f", "% por clase real")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks(labels); ax.set_yticks(labels)
        ax.set_xticklabels(nombres, rotation=20); ax.set_yticklabels(nombres)
        ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real"); ax.set_title(sub)
        for i in labels:
            for j in labels:
                v = mat[i, j]
                ax.text(j, i, format(v, fmt) + ("%" if sub != "Conteo" else ""), ha="center", va="center",
                        color="white" if v > mat.max() * 0.55 else "black", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(titulo)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return cm


def guardar_json(obj, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# >>> paso1_dividir_dataset.py
# =============================================================================
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




# =============================================================================
# >>> paso2_preprocesamiento.py
# =============================================================================
"""
paso2_preprocesamiento.py
=========================
Para cada imagen de dataset_split/:
  1. Descarta imágenes inválidas (corruptas, muy pequeñas, uniformes, sub/sobreexpuestas, desenfocadas).
  2. Recorta la región de interés (ROI) alrededor de la pieza.
  3. Redimensiona a IMG_BASE x IMG_BASE (y a grises si USAR_GRIS=True).
  4. Guarda en dataset_proc/ y calcula media/desviación del conjunto de entrenamiento (normalización).
También define la data augmentation que usará la CNN (solo en entrenamiento).

Uso:
    python paso2_preprocesamiento.py                       # procesa todo
    python paso2_preprocesamiento.py --seleccionar-roi     # elegir ROI manual con el ratón
"""
import csv
import shutil
import argparse



# ----------------------------------------------------------------------------
# Selección interactiva de ROI
# ----------------------------------------------------------------------------
def seleccionar_roi_interactivo():
    carpeta = DIR_ORIGINAL if DIR_ORIGINAL.exists() else DIR_SPLIT
    por_clase, _ = listar_imagenes_por_clase(carpeta)
    ruta = next((v[0] for v in por_clase.values() if v), None)
    if ruta is None:
        raise SystemExit("No hay imágenes para seleccionar la ROI.")
    img = leer_imagen(ruta)
    esc = min(1.0, 1000 / img.shape[1])
    vista = cv2.resize(img, None, fx=esc, fy=esc)
    print(f"Imagen: {ruta.name}. Dibuja el rectángulo sobre la pieza y pulsa ENTER (c = cancelar).")
    x, y, w, h = cv2.selectROI("Selecciona la ROI", vista, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("Cancelado.")
        return
    x, y, w, h = [int(round(v / esc)) for v in (x, y, w, h)]
    print("\nCopia esto en config.py:\n")
    print(f'ROI_MODO = "manual"\nROI_MANUAL = ({x}, {y}, {w}, {h})\n')
    print("(Todas las imágenes deben tener la misma resolución y la pieza en una zona similar.)")


# ----------------------------------------------------------------------------
# Data augmentation (se usa desde paso4)
# ----------------------------------------------------------------------------
def crear_augmentacion():
    """Transformaciones sobre tensores float [0,1] de forma (C, H, W)."""
    import torch
    from torchvision.transforms import v2

    def ruido(x):
        return (x + torch.randn_like(x) * AUG["ruido_std"]).clamp(0, 1)

    return v2.Compose([
        v2.RandomAffine(degrees=AUG["rotacion"], translate=(AUG["traslacion"], AUG["traslacion"]),
                        scale=AUG["zoom"]),
        v2.ColorJitter(brightness=AUG["brillo"], contrast=AUG["contraste"]),
        v2.Lambda(ruido),
    ])


def guardar_ejemplos_augmentacion(n_ejemplos=4):
    """Figura con una imagen de cada clase y varias versiones aumentadas."""
    import torch
    items = listar_split("train")
    if not items:
        return
    aug = crear_augmentacion()
    fig, ejes = plt.subplots(len(CLASES), n_ejemplos + 1, figsize=(2.2 * (n_ejemplos + 1), 2.3 * len(CLASES)))
    for fila, c in enumerate(CLASES):
        ruta = next(p for p, i in items if i == CLASES.index(c))
        img = cargar_procesada(ruta)
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        t = t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
        for col in range(n_ejemplos + 1):
            x = t if col == 0 else aug(t)
            arr = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            arr = arr[..., 0] if arr.shape[2] == 1 else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            ejes[fila, col].imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
            ejes[fila, col].axis("off")
            ejes[fila, col].set_title(f"{NOMBRES_MOSTRAR[c]}\n(original)" if col == 0 else "aumentada", fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_RES / "preprocesamiento" / "ejemplos_augmentacion.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Proceso principal
# ----------------------------------------------------------------------------
def main_preprocesar(seleccionar_roi=False):
    if seleccionar_roi:
        seleccionar_roi_interactivo()
        return
    if not DIR_SPLIT.exists():
        raise SystemExit("Primero ejecuta paso1_dividir_dataset.py")
    if DIR_PROC.exists():
        shutil.rmtree(DIR_PROC)
    (DIR_RES / "preprocesamiento").mkdir(parents=True, exist_ok=True)

    invalidas, conteo, ejemplos = [], {}, {c: [] for c in CLASES}
    splits = [d.name for d in sorted(DIR_SPLIT.iterdir()) if d.is_dir()]
    for split in splits:
        for clase in CLASES:
            carpeta = DIR_SPLIT / split / clase
            if not carpeta.exists():
                continue
            for p in sorted(carpeta.iterdir(), key=clave_natural):
                if p.suffix.lower() not in EXT_VALIDAS:
                    continue
                try:
                    img = leer_imagen(p)
                except Exception:
                    img = None
                ok, motivo = es_imagen_valida(img)
                if not ok:
                    invalidas.append((split, clase, p.name, motivo))
                    continue
                roi = calcular_roi(img)
                proc = preprocesar_imagen(img, roi)
                guardar_imagen(DIR_PROC / split / clase / (p.stem + ".png"), proc)
                conteo[(split, clase)] = conteo.get((split, clase), 0) + 1
                if split == "train" and len(ejemplos[clase]) < 2:
                    ejemplos[clase].append((img, roi, proc))

    # Reporte de imágenes descartadas
    with open(DIR_RES / "preprocesamiento" / "imagenes_invalidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "clase", "archivo", "motivo"])
        w.writerows(invalidas)
    print(f"Imágenes descartadas: {len(invalidas)}")
    for fila in invalidas:
        print("   ", *fila)

    print("\nImágenes procesadas por conjunto:")
    print(f"{'split':<14}" + "".join(f"{c:>12}" for c in CLASES))
    for split in splits:
        print(f"{split:<14}" + "".join(f"{conteo.get((split, c), 0):>12}" for c in CLASES))

    # Estadísticas de normalización (solo con entrenamiento, para no filtrar información de val/test)
    items = listar_split("train")
    if not items:
        raise SystemExit("No hay imágenes de entrenamiento procesadas.")
    n_canales = 1 if USAR_GRIS else 3
    suma, suma2, n_pix = np.zeros(n_canales), np.zeros(n_canales), 0
    for p, _ in items:
        a = cargar_procesada(p).astype(np.float64) / 255.0
        a = a.reshape(-1, n_canales)
        suma += a.sum(0)
        suma2 += (a ** 2).sum(0)
        n_pix += a.shape[0]
    media = suma / n_pix
    desv = np.sqrt(np.maximum(suma2 / n_pix - media ** 2, 1e-12))
    guardar_json({"mean": media.tolist(), "std": desv.tolist(), "gris": bool(USAR_GRIS),
                  "img_base": IMG_BASE}, DIR_PROC / "normalizacion.json")
    print(f"\nNormalización (train): media={np.round(media, 4).tolist()}  std={np.round(desv, 4).tolist()}")

    # Figura de verificación de la ROI
    filas = [(c, e) for c in CLASES for e in ejemplos[c]]
    if filas:
        fig, ejes = plt.subplots(len(filas), 2, figsize=(6, 3 * len(filas)))
        ejes = np.atleast_2d(ejes)
        for k, (c, (img, roi, proc)) in enumerate(filas):
            vista = img.copy()
            x, y, w, h = roi
            grosor = max(3, img.shape[1] // 200)
            cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 255, 0), grosor)
            ejes[k, 0].imshow(cv2.cvtColor(vista, cv2.COLOR_BGR2RGB)); ejes[k, 0].set_title(f"{c}: ROI detectada", fontsize=9)
            if proc.ndim == 2:
                ejes[k, 1].imshow(proc, cmap="gray", vmin=0, vmax=255)
            else:
                ejes[k, 1].imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
            ejes[k, 1].set_title(f"procesada {IMG_BASE}x{IMG_BASE}", fontsize=9)
            ejes[k, 0].axis("off"); ejes[k, 1].axis("off")
        fig.tight_layout()
        fig.savefig(DIR_RES / "preprocesamiento" / "ejemplos_roi.png", dpi=120)
        plt.close(fig)
        print(f"Revisa visualmente la ROI en: {DIR_RES / 'preprocesamiento' / 'ejemplos_roi.png'}")

    try:
        guardar_ejemplos_augmentacion()
        print("Ejemplos de augmentation: resultados/preprocesamiento/ejemplos_augmentacion.png")
    except ImportError:
        print("(PyTorch no instalado: se omite la figura de augmentation)")




# =============================================================================
# >>> paso3_linea_base.py
# =============================================================================
"""
paso3_linea_base.py
===================
Línea base CONVENCIONAL (sin redes neuronales):

  Imagen -> escala de grises -> filtrado (Gauss) -> umbralización (Otsu) -> bordes (Canny)
         -> área / perímetro / solidez / textura -> árbol de decisión pequeño (regla de decisión)

El árbol es poco profundo a propósito: es una regla de decisión legible (se guarda en reglas_arbol.txt).
La profundidad se elige con el conjunto de validación; test solo se mide al final.

Uso:
    python paso3_linea_base.py
"""
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text


NOMBRES_CARACT = ["media", "desviacion", "entropia", "lap_var", "grad_medio", "dens_bordes",
                  "n_comp_bordes", "frac_pieza", "area_rel", "perim_rel", "solidez",
                  "circularidad", "frac_oscuras"]


def a_gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def extraer_caracteristicas(img):
    """Vector de 13 características clásicas a partir de una imagen uint8 (gris o BGR)."""
    gris = a_gris(img)
    h, w = gris.shape
    filtrada = cv2.GaussianBlur(gris, (5, 5), 0)                       # filtrado

    # --- Intensidad / textura
    media, desv = float(gris.mean()), float(gris.std())
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    entropia = float(-(p * np.log2(p)).sum())
    lap_var = float(cv2.Laplacian(filtrada, cv2.CV_64F).var())
    gx = cv2.Sobel(filtrada, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtrada, cv2.CV_32F, 0, 1, ksize=3)
    grad_medio = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    # --- Bordes (Canny con umbrales automáticos a partir de la mediana)
    mediana = float(np.median(filtrada))
    bordes = cv2.Canny(filtrada, int(max(0, 0.67 * mediana)), int(min(255, 1.33 * mediana)))
    dens_bordes = float((bordes > 0).mean())
    n_comp_bordes = float(cv2.connectedComponents(bordes)[0] - 1)

    # --- Umbralización -> forma (área, perímetro, solidez, circularidad)
    mascara = mascara_pieza(gris)
    frac_pieza = float((mascara > 0).mean())
    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_rel = perim_rel = solidez = circ = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        area_hull = cv2.contourArea(cv2.convexHull(c))
        area_rel = area / (h * w)
        perim_rel = per / (2 * (h + w))
        solidez = area / area_hull if area_hull > 0 else 0.0
        circ = 4 * np.pi * area / (per ** 2) if per > 0 else 0.0

    # --- Zonas anormalmente oscuras (huecos, roturas, sombras)
    frac_oscuras = float((gris < media - 2 * desv).mean())

    valores = [media, desv, entropia, lap_var, grad_medio, dens_bordes, n_comp_bordes,
               frac_pieza, area_rel, perim_rel, float(solidez), float(circ), frac_oscuras]
    return dict(zip(NOMBRES_CARACT, valores))


def matriz_caracteristicas(split):
    items = listar_split(split)
    X = np.array([list(extraer_caracteristicas(cargar_procesada(p)).values()) for p, _ in items])
    y = np.array([c for _, c in items])
    return X, y, items


def predecir_baseline(modelo, imagenes):
    X = np.array([list(extraer_caracteristicas(im).values()) for im in imagenes])
    return modelo.predict(X)


def main_linea_base():
    salida = DIR_RES / "baseline"
    salida.mkdir(parents=True, exist_ok=True)
    fijar_semilla()

    X_tr, y_tr, _ = matriz_caracteristicas("train")
    X_va, y_va, _ = matriz_caracteristicas("val")
    if len(y_tr) == 0:
        raise SystemExit("No hay datos procesados. Ejecuta paso1 y paso2 primero.")

    # Elegir profundidad con validación (en empate gana el árbol más simple)
    mejor_d, mejor_acc = 1, -1
    for d in range(1, 8):
        acc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=3, random_state=SEMILLA).fit(X_tr, y_tr).score(X_va, y_va)
        if acc > mejor_acc + 1e-9:
            mejor_d, mejor_acc = d, acc
    clf = DecisionTreeClassifier(max_depth=mejor_d, min_samples_leaf=3, random_state=SEMILLA).fit(X_tr, y_tr)
    print(f"Profundidad elegida (por validación): {mejor_d}  (acc val = {mejor_acc:.3f})")

    resultados = {"profundidad": mejor_d, "n_caracteristicas": len(NOMBRES_CARACT),
                  "n_nodos": int(clf.tree_.node_count)}
    for split in ["val", "test", "test_externo"]:
        X, y, items = matriz_caracteristicas(split)
        if len(y) == 0:
            continue
        pred = clf.predict(X)
        m = calcular_metricas(y, pred)
        resultados[split] = m
        graficar_matriz_confusion(y, pred, f"Línea base (árbol) - {split}", salida / f"matriz_confusion_{split}.png")
        print(f"[{split:>12}] accuracy={m['accuracy']:.3f}  F1 macro={m['f1_macro']:.3f}  (n={m['n']})")

    # Latencia (extracción de características + predicción, 1 imagen)
    prueba = cargar_procesada(listar_split("test")[0][0])
    ms, sd = medir_tiempo_ms(lambda: clf.predict(np.array([list(extraer_caracteristicas(prueba).values())])), n=100)
    resultados["latencia_ms_cpu"] = ms
    print(f"Latencia: {ms:.2f} ± {sd:.2f} ms/imagen (CPU)")

    # Reglas legibles + importancia
    with open(salida / "reglas_arbol.txt", "w", encoding="utf-8") as f:
        f.write("Clases: " + ", ".join(f"{i}={c}" for i, c in enumerate(CLASES)) + "\n\n")
        f.write(export_text(clf, feature_names=NOMBRES_CARACT))
        f.write("\nImportancia de características:\n")
        for i in np.argsort(clf.feature_importances_)[::-1]:
            if clf.feature_importances_[i] > 0:
                f.write(f"  {NOMBRES_CARACT[i]:<16}{clf.feature_importances_[i]:.3f}\n")

    # Boxplots de las características más importantes por clase
    top = [i for i in np.argsort(clf.feature_importances_)[::-1] if clf.feature_importances_[i] > 0][:4]
    if top:
        fig, ejes = plt.subplots(1, len(top), figsize=(3.4 * len(top), 3.6))
        ejes = np.atleast_1d(ejes)
        for ax, i in zip(ejes, top):
            ax.boxplot([X_tr[y_tr == k, i] for k in range(len(CLASES))])
            ax.set_xticklabels(etiquetas_mostrar(), rotation=25)
            ax.set_title(NOMBRES_CARACT[i])
        fig.tight_layout()
        fig.savefig(salida / "caracteristicas_por_clase.png", dpi=130)
        plt.close(fig)

    joblib.dump({"modelo": clf, "columnas": NOMBRES_CARACT}, salida / "baseline_modelo.joblib")
    guardar_json(resultados, salida / "metricas.json")
    print(f"Resultados en {salida}")




# =============================================================================
# >>> paso4_entrenar_cnn.py
# =============================================================================
"""
paso4_entrenar_cnn.py
=====================
Entrena una CNN (PyTorch) sobre dataset_proc/ y registra:
  accuracy / loss de entrenamiento y validación, nº de parámetros, MACs, tiempo de entrenamiento,
  curvas de loss y accuracy, matriz de confusión, precision / recall / F1.

Uso:
    python paso4_entrenar_cnn.py               # entrena la configuración CNN_BASE y evalúa en test
    python paso4_entrenar_cnn.py --estudio     # estudio de hiperparámetros (usa SOLO validación)

El estudio varía un factor a la vez (GRID en config.py): tamaño de imagen, nº de filtros,
nº de capas, learning rate, batch size y nº de épocas.
"""
import time
import copy
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def imagen_a_tensor(img_uint8, img_size):
    """uint8 (H,W) o (H,W,3) -> tensor float [0,1] de forma (C, img_size, img_size)."""
    if img_uint8.shape[0] != img_size:
        img_uint8 = cv2.resize(img_uint8, (img_size, img_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img_uint8.astype(np.float32) / 255.0)
    return t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1).contiguous()


class DatasetImagenes(Dataset):
    """Carga todo el split en memoria (son pocas imágenes pequeñas)."""

    def __init__(self, split, img_size, aumentar=False, base=None):
        norm = cargar_normalizacion()
        n_c = len(norm["mean"])
        self.media = torch.tensor(norm["mean"], dtype=torch.float32).view(n_c, 1, 1)
        self.std = torch.tensor(norm["std"], dtype=torch.float32).view(n_c, 1, 1)
        self.items = listar_split(split, base)
        self.rutas = [p for p, _ in self.items]
        self.y = torch.tensor([c for _, c in self.items], dtype=torch.long)
        self.x = [imagen_a_tensor(cargar_procesada(p), img_size) for p in self.rutas]
        self.aug = crear_augmentacion() if aumentar else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.x[i]
        if self.aug is not None:
            x = self.aug(x)
        return (x - self.media) / self.std, self.y[i]


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
class CNNSimple(nn.Module):
    """n_capas bloques [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2]; filtros: f, 2f, 4f ...
    Después: promedio global -> Dropout -> Linear."""

    def __init__(self, in_ch, n_clases, n_filtros=16, n_capas=3, dropout=0.3):
        super().__init__()
        bloques, c_in = [], in_ch
        for i in range(n_capas):
            c_out = n_filtros * (2 ** i)
            bloques += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*bloques)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_clases))

    def forward(self, x):
        return self.clasificador(self.pool(self.features(x)))


def construir_modelo(cfg):
    n_c = len(cargar_normalizacion()["mean"])
    return CNNSimple(n_c, len(CLASES), cfg["n_filtros"], cfg["n_capas"], cfg["dropout"])


def contar_parametros(modelo):
    return int(sum(p.numel() for p in modelo.parameters() if p.requires_grad))


def contar_macs(modelo, in_ch, img_size):
    """Multiplicaciones-acumulaciones aproximadas de capas Conv2d y Linear (complejidad)."""
    total, hooks = [0], []

    def hook(m, entrada, salida):
        if isinstance(m, nn.Conv2d):
            total[0] += salida.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features

    for m in modelo.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    estado = modelo.training
    modelo.eval()
    with torch.no_grad():
        modelo(torch.zeros(1, in_ch, img_size, img_size, device=next(modelo.parameters()).device))
    modelo.train(estado)
    for h in hooks:
        h.remove()
    return int(total[0])


# ----------------------------------------------------------------------------
# Entrenamiento / evaluación
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluar_modelo(modelo, loader):
    modelo.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    perdida, ys, preds, probs = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(DISPOSITIVO), y.to(DISPOSITIVO)
        salida = modelo(x)
        perdida += crit(salida, y).item()
        probs.append(torch.softmax(salida, 1).cpu().numpy())
        preds.append(salida.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    ys, preds, probs = np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)
    return perdida / len(ys), float((ys == preds).mean()), ys, preds, probs


def entrenar(cfg, semilla=SEMILLA, verbose=True):
    fijar_semilla(semilla)
    ds_tr = DatasetImagenes("train", cfg["img_size"], aumentar=True)
    ds_va = DatasetImagenes("val", cfg["img_size"])
    bs = cfg["batch_size"]
    drop_last = len(ds_tr) % bs == 1          # evita un lote de 1 imagen con BatchNorm
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=drop_last, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0)

    modelo = construir_modelo(cfg).to(DISPOSITIVO)
    n_params = contar_parametros(modelo)
    n_c = ds_tr.x[0].shape[0]
    macs = contar_macs(modelo, n_c, cfg["img_size"])
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.CrossEntropyLoss()

    hist, mejor = [], {"loss": float("inf"), "estado": None, "epoca": 0}
    t0 = time.perf_counter()
    for ep in range(1, cfg["epocas"] + 1):
        modelo.train()
        s_loss, s_ok, n = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(DISPOSITIVO), y.to(DISPOSITIVO)
            opt.zero_grad()
            salida = modelo(x)
            loss = crit(salida, y)
            loss.backward()
            opt.step()
            s_loss += loss.item() * len(y)
            s_ok += (salida.argmax(1) == y).sum().item()
            n += len(y)
        l_va, a_va, _, _, _ = evaluar_modelo(modelo, dl_va)
        hist.append(dict(epoca=ep, loss_train=s_loss / n, acc_train=s_ok / n, loss_val=l_va, acc_val=a_va,
                         tiempo_s=time.perf_counter() - t0))
        if l_va < mejor["loss"] - 1e-6:
            mejor = {"loss": l_va, "estado": copy.deepcopy(modelo.state_dict()), "epoca": ep}
        if verbose and (ep == 1 or ep % 5 == 0 or ep == cfg["epocas"]):
            h = hist[-1]
            print(f"  ép {ep:3d} | loss {h['loss_train']:.4f} / {h['loss_val']:.4f} | "
                  f"acc {h['acc_train']:.3f} / {h['acc_val']:.3f}  (train / val)")
        if cfg.get("paciencia") and ep - mejor["epoca"] >= cfg["paciencia"]:
            if verbose:
                print(f"  Early stopping en la época {ep} (mejor: {mejor['epoca']})")
            break
    tiempo = time.perf_counter() - t0

    modelo.load_state_dict(mejor["estado"])
    l_va, a_va, y_va, p_va, _ = evaluar_modelo(modelo, dl_va)
    m_va = calcular_metricas(y_va, p_va)
    return dict(modelo=modelo, historial=pd.DataFrame(hist), mejor_epoca=mejor["epoca"],
                val_loss=l_va, val_acc=a_va, val_f1=m_va["f1_macro"], params=n_params, macs=macs,
                tiempo_entrenamiento_s=tiempo, epocas_ejecutadas=len(hist))


def graficar_curvas(hist, mejor_epoca, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(hist["epoca"], hist["loss_train"], label="entrenamiento")
    ejes[0].plot(hist["epoca"], hist["loss_val"], label="validación")
    ejes[0].set_title("Loss"); ejes[0].set_xlabel("Época")
    ejes[1].plot(hist["epoca"], hist["acc_train"], label="entrenamiento")
    ejes[1].plot(hist["epoca"], hist["acc_val"], label="validación")
    ejes[1].set_title("Accuracy"); ejes[1].set_xlabel("Época"); ejes[1].set_ylim(0, 1.02)
    for ax in ejes:
        ax.axvline(mejor_epoca, color="gray", ls="--", lw=1, label=f"mejor época ({mejor_epoca})")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def guardar_checkpoint(res, cfg, ruta):
    torch.save({"state_dict": res["modelo"].state_dict(), "cfg": cfg, "clases": CLASES}, ruta)


def cargar_checkpoint(ruta=None):
    ruta = ruta or (DIR_RES / "cnn" / "modelo_mejor.pt")
    ck = torch.load(ruta, map_location=DISPOSITIVO)
    modelo = construir_modelo(ck["cfg"]).to(DISPOSITIVO)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    return modelo, ck["cfg"]


# ----------------------------------------------------------------------------
# Programas principales
# ----------------------------------------------------------------------------
def main_entrenar():
    salida = DIR_RES / "cnn"
    salida.mkdir(parents=True, exist_ok=True)
    cfg = dict(CNN_BASE)
    print(f"Dispositivo: {DISPOSITIVO} | configuración: {cfg}")
    res = entrenar(cfg, SEMILLA, verbose=True)

    res["historial"].to_csv(salida / "historial_entrenamiento.csv", index=False)
    graficar_curvas(res["historial"], res["mejor_epoca"], salida / "curvas_loss_accuracy.png")
    guardar_checkpoint(res, cfg, salida / "modelo_mejor.pt")

    # Evaluación final en test (y en test externo si existe) con el mejor modelo
    resumen = {"cfg": cfg, "params": res["params"], "macs": res["macs"],
               "tiempo_entrenamiento_s": res["tiempo_entrenamiento_s"],
               "mejor_epoca": res["mejor_epoca"], "epocas_ejecutadas": res["epocas_ejecutadas"],
               "train_acc_final": float(res["historial"].iloc[res["mejor_epoca"] - 1]["acc_train"]),
               "train_loss_final": float(res["historial"].iloc[res["mejor_epoca"] - 1]["loss_train"]),
               "val": {"loss": res["val_loss"], "acc": res["val_acc"], "f1_macro": res["val_f1"]}}
    for split in ["test", "test_externo"]:
        if not listar_split(split):
            continue
        ds = DatasetImagenes(split, cfg["img_size"])
        loss, acc, y, pred, _ = evaluar_modelo(res["modelo"], DataLoader(ds, batch_size=64))
        m = calcular_metricas(y, pred)
        m["loss"] = loss
        resumen[split] = m
        graficar_matriz_confusion(y, pred, f"CNN - {split}", salida / f"matriz_confusion_{split}.png")
        print(f"\n[{split}] accuracy={m['accuracy']:.3f} | precision={m['precision_macro']:.3f} | "
              f"recall={m['recall_macro']:.3f} | F1={m['f1_macro']:.3f}")
        for c in CLASES:
            pc = m["por_clase"][c]
            print(f"    {c:<12} P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  (n={pc['soporte']})")
    guardar_json(resumen, salida / "metricas.json")

    print(f"\nParámetros: {res['params']:,} | MACs: {res['macs']:,} | "
          f"tiempo de entrenamiento: {res['tiempo_entrenamiento_s']:.1f} s | mejor época: {res['mejor_epoca']}")
    print(f"Resultados en {salida}")


def main_estudio():
    salida = DIR_RES / "estudio_hiperparametros"
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {DISPOSITIVO}. Estudio con {N_SEMILLAS_ESTUDIO} semilla(s) por configuración.")
    cache, filas = {}, []

    # Configuraciones únicas: la base + una variación de cada factor
    configs = {}
    for factor, valores in GRID.items():
        for v in valores:
            cfg = dict(CNN_BASE, **{factor: v}, paciencia=None)      # sin early stopping para comparar épocas
            configs[tuple(sorted(cfg.items()))] = cfg
    print(f"{len(configs)} configuraciones x {N_SEMILLAS_ESTUDIO} semillas")

    for k, (clave, cfg) in enumerate(configs.items(), 1):
        for s in range(N_SEMILLAS_ESTUDIO):
            r = entrenar(cfg, SEMILLA + s, verbose=False)
            cache.setdefault(clave, []).append(r)
        r0 = cache[clave]
        print(f"[{k}/{len(configs)}] val_acc={np.mean([x['val_acc'] for x in r0]):.3f} "
              f"val_f1={np.mean([x['val_f1'] for x in r0]):.3f} params={r0[0]['params']:,}")

    for factor, valores in GRID.items():
        for v in valores:
            cfg = dict(CNN_BASE, **{factor: v}, paciencia=None)
            rs = cache[tuple(sorted(cfg.items()))]
            filas.append(dict(
                factor=factor, valor=v, es_base=(v == CNN_BASE[factor]),
                val_acc_media=np.mean([r["val_acc"] for r in rs]), val_acc_std=np.std([r["val_acc"] for r in rs]),
                val_f1_media=np.mean([r["val_f1"] for r in rs]), val_loss_media=np.mean([r["val_loss"] for r in rs]),
                params=rs[0]["params"], macs=rs[0]["macs"],
                tiempo_s=np.mean([r["tiempo_entrenamiento_s"] for r in rs])))
    df = pd.DataFrame(filas)
    df.to_csv(salida / "resultados_estudio.csv", index=False)

    fig, ejes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (factor, g) in zip(ejes.ravel(), df.groupby("factor", sort=False)):
        x = np.arange(len(g))
        ax.bar(x, g["val_acc_media"], yerr=g["val_acc_std"], capsize=4,
               color=["tab:orange" if b else "tab:blue" for b in g["es_base"]])
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in g["valor"]])
        ax.set_ylim(max(0, g["val_acc_media"].min() - 0.15), 1.02)
        ax.set_title(factor); ax.set_ylabel("accuracy validación"); ax.grid(axis="y", alpha=0.3)
        for xi, (_, fila) in zip(x, g.iterrows()):
            ax.text(xi, ax.get_ylim()[0] + 0.01, f"{fila['tiempo_s']:.0f}s\n{fila['params'] / 1000:.0f}k",
                    ha="center", va="bottom", fontsize=7, color="white")
    fig.suptitle("Estudio de hiperparámetros (naranja = configuración base; texto: tiempo y nº de parámetros)")
    fig.tight_layout()
    fig.savefig(salida / "estudio_hiperparametros.png", dpi=140)
    plt.close(fig)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nResultados en {salida}")
    print("Nota: con ~10 imágenes de validación por clase las diferencias pequeñas NO son significativas.")




# =============================================================================
# >>> paso5_evaluar_errores.py
# =============================================================================
"""
paso5_evaluar_errores.py
========================
Análisis de tipos de fallo del modelo entrenado (paso4) y comparación con la línea base (paso3):

  * Matriz de confusión (clase real vs predicha) para CNN y línea base.
  * Defecto más fácil / más difícil, confusiones entre clases, errores sistemáticos
    (falsos negativos: defecto -> "normal", falsas alarmas, imágenes mal clasificadas con su confianza).
  * Grad-CAM: dónde "mira" la red (¿al defecto o al fondo/bordes?).
  * Robustez ante cambios de brillo, contraste, ruido, rotación, desenfoque y zoom.
  * Latencia y complejidad: CNN vs método convencional.

Uso:
    python paso5_evaluar_errores.py

Si existe dataset_split/test_externo/ (otras condiciones de captura) también se evalúa.
"""
import copy
import joblib



# ----------------------------------------------------------------------------
# Predicción con CNN sobre imágenes uint8 (permite probar perturbaciones)
# ----------------------------------------------------------------------------
def predecir_cnn(modelo, cfg, imagenes):
    norm = cargar_normalizacion()
    n_c = len(norm["mean"])
    media = torch.tensor(norm["mean"], dtype=torch.float32).view(n_c, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32).view(n_c, 1, 1)
    modelo.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(imagenes), 64):
            lote = torch.stack([(imagen_a_tensor(im, cfg["img_size"]) - media) / std
                                for im in imagenes[i:i + 64]]).to(DISPOSITIVO)
            p = torch.softmax(modelo(lote), 1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
    return np.concatenate(preds), np.concatenate(probs)


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def grad_cam(modelo, x, clase=None):
    """x: tensor normalizado (1, C, H, W). Devuelve (mapa [0,1] HxW, clase usada)."""
    modelo.eval()
    x = x.to(DISPOSITIVO)
    with torch.enable_grad():
        a = modelo.features[:-1](x)               # mapa de activación antes del último MaxPool
        a.retain_grad()
        salida = modelo.clasificador(modelo.pool(modelo.features[-1](a)))
        if clase is None:
            clase = int(salida.argmax(1))
        modelo.zero_grad()
        salida[0, clase].backward()
    pesos = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((pesos * a).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8), clase


def superponer_cam(img_uint8, cam):
    h, w = cam.shape
    base = cv2.resize(img_uint8, (w, h), interpolation=cv2.INTER_AREA)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    calor = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(base, 0.55, calor, 0.45, 0), cv2.COLOR_BGR2RGB)


def figura_gradcam(modelo, cfg, ds, indices, y, pred, probs, ruta, titulo, columnas=4):
    if len(indices) == 0:
        return
    filas = int(np.ceil(len(indices) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3 * columnas, 3.3 * filas))
    ejes = np.atleast_1d(ejes).ravel()
    for ax in ejes:
        ax.axis("off")
    for ax, i in zip(ejes, indices):
        x, _ = ds[i]
        cam, _ = grad_cam(modelo, x.unsqueeze(0), clase=int(pred[i]))
        img = cargar_procesada(ds.rutas[i])
        ax.imshow(superponer_cam(img, cam))
        color = "green" if y[i] == pred[i] else "red"
        ax.set_title(f"{ds.rutas[i].stem}\nreal: {CLASES[y[i]]} -> pred: {CLASES[pred[i]]} ({probs[i, pred[i]]:.2f})",
                     fontsize=8, color=color)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Análisis textual de errores
# ----------------------------------------------------------------------------
def analisis_texto(m, cm, y, pred, probs, rutas, split):
    L = [f"=== Análisis de tipos de fallo - conjunto '{split}' (n={len(y)}) ===", ""]
    L.append(f"Accuracy: {m['accuracy']:.3f} | F1 macro: {m['f1_macro']:.3f} | "
             f"errores: {int((y != pred).sum())} de {len(y)}")
    L.append("")
    L.append("Rendimiento por clase:")
    for c in CLASES:
        pc = m["por_clase"][c]
        L.append(f"  {c:<12} precision={pc['precision']:.3f}  recall={pc['recall']:.3f}  f1={pc['f1']:.3f}  (n={pc['soporte']})")

    # Defecto más fácil / más difícil
    L.append("")
    defectos = [c for c in CLASES if c != "normal"]
    orden = sorted(defectos, key=lambda c: (m["por_clase"][c]["recall"], m["por_clase"][c]["f1"]), reverse=True)
    rec = {c: m["por_clase"][c]["recall"] for c in defectos}
    if len(defectos) > 1 and abs(rec[orden[0]] - rec[orden[-1]]) < 1e-9:
        L.append(f"Defecto más fácil/difícil: sin diferencias apreciables (recall = {rec[orden[0]]:.2f} en todos).")
    elif defectos:
        L.append(f"Defecto MÁS FÁCIL de detectar: {orden[0]} (recall {rec[orden[0]]:.2f})")
        L.append(f"Defecto MÁS DIFÍCIL de detectar: {orden[-1]} (recall {rec[orden[-1]]:.2f})")

    # Confusiones
    L.append("")
    L.append("Confusiones entre clases (real -> predicha):")
    pares = [(cm[i, j], i, j) for i in range(len(CLASES)) for j in range(len(CLASES)) if i != j and cm[i, j] > 0]
    if not pares:
        L.append("  Ninguna: todas las imágenes se clasificaron correctamente.")
    for n, i, j in sorted(pares, reverse=True):
        L.append(f"  {CLASES[i]} -> {CLASES[j]}: {n} caso(s) ({100 * n / max(cm[i].sum(), 1):.1f} % de '{CLASES[i]}')")
    asim = [(i, j) for _, i, j in pares if i < j and abs(int(cm[i, j]) - int(cm[j, i])) >= 2]
    for i, j in asim:
        L.append(f"  * Confusión asimétrica: {CLASES[i]}->{CLASES[j]} = {cm[i, j]} vs {CLASES[j]}->{CLASES[i]} = {cm[j, i]} "
                 f"(posible sesgo sistemático del modelo hacia una de las dos clases).")

    # Errores críticos en inspección
    if "normal" in CLASES:
        k = CLASES.index("normal")
        fn = int(sum(cm[i, k] for i in range(len(CLASES)) if i != k))
        fp = int(sum(cm[k, j] for j in range(len(CLASES)) if j != k))
        n_def = int(sum(cm[i].sum() for i in range(len(CLASES)) if i != k))
        L.append("")
        L.append(f"Falsos negativos críticos (defecto clasificado como normal): {fn} de {n_def} piezas defectuosas")
        L.append(f"Falsas alarmas (normal clasificada como defecto): {fp} de {int(cm[k].sum())} piezas normales")

    # Imágenes mal clasificadas
    mal = np.where(y != pred)[0]
    L.append("")
    L.append(f"Imágenes mal clasificadas ({len(mal)}):")
    for i in mal:
        L.append(f"  {rutas[i].name:<28} real={CLASES[y[i]]:<11} pred={CLASES[pred[i]]:<11} confianza={probs[i, pred[i]]:.2f}")
    dudosas = int(((probs.max(1) < 0.6) & (y == pred)).sum())
    L.append(f"Aciertos con confianza baja (<0.60): {dudosas}")
    if len(mal):
        L.append("")
        L.append("Qué revisar para detectar errores sistemáticos: ¿las imágenes falladas comparten posición, cara del cubo,")
        L.append("iluminación o altura? ¿están consecutivas (mismo momento de captura)? Mira el Grad-CAM de errores:")
        L.append("si la red mira bordes, sombras o fondo en lugar del defecto, está aprendiendo un atajo.")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Robustez y latencia
# ----------------------------------------------------------------------------
def prueba_robustez(modelo, cfg, clf, split="test"):
    items = listar_split(split)
    imgs = [cargar_procesada(p) for p, _ in items]
    y = np.array([c for _, c in items])
    filas = []
    for tipo in PERTURBACIONES:
        pert = [perturbar(im, tipo, semilla=i) for i, im in enumerate(imgs)]
        p_cnn, _ = predecir_cnn(modelo, cfg, pert)
        p_base = predecir_baseline(clf, pert)
        mc, mb = calcular_metricas(y, p_cnn), calcular_metricas(y, p_base)
        filas.append(dict(perturbacion=tipo, acc_cnn=mc["accuracy"], f1_cnn=mc["f1_macro"],
                          acc_baseline=mb["accuracy"], f1_baseline=mb["f1_macro"]))
    return pd.DataFrame(filas)


def medir_latencias(modelo, cfg, clf, img):
    norm = cargar_normalizacion()
    n_c = len(norm["mean"])
    media = torch.tensor(norm["mean"], dtype=torch.float32).view(n_c, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32).view(n_c, 1, 1)
    m_cpu = copy.deepcopy(modelo).cpu().eval()

    def f_cnn():
        x = ((imagen_a_tensor(img, cfg["img_size"]) - media) / std).unsqueeze(0)
        with torch.no_grad():
            m_cpu(x).argmax(1)

    def f_base():
        clf.predict(np.array([list(extraer_caracteristicas(img).values())]))

    return medir_tiempo_ms(f_cnn, n=100)[0], medir_tiempo_ms(f_base, n=100)[0]


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main_evaluar():
    salida = DIR_RES / "evaluacion"
    salida.mkdir(parents=True, exist_ok=True)
    ruta_cnn = DIR_RES / "cnn" / "modelo_mejor.pt"
    ruta_base = DIR_RES / "baseline" / "baseline_modelo.joblib"
    if not ruta_cnn.exists():
        raise SystemExit("Falta el modelo CNN: ejecuta paso4_entrenar_cnn.py")
    if not ruta_base.exists():
        raise SystemExit("Falta la línea base: ejecuta paso3_linea_base.py")
    modelo, cfg = cargar_checkpoint(ruta_cnn)
    clf = joblib.load(ruta_base)["modelo"]
    resumen = {}

    for split in ["test", "test_externo"]:
        items = listar_split(split)
        if not items:
            continue
        print(f"\n################ {split} ({len(items)} imágenes) ################")
        ds = DatasetImagenes(split, cfg["img_size"])
        imgs = [cargar_procesada(p) for p in ds.rutas]
        y = ds.y.numpy()
        pred, probs = predecir_cnn(modelo, cfg, imgs)
        pred_b = predecir_baseline(clf, imgs)

        m, mb = calcular_metricas(y, pred), calcular_metricas(y, pred_b)
        resumen[split] = (m, mb)
        cm = graficar_matriz_confusion(y, pred, f"CNN - matriz de confusión ({split})", salida / f"matriz_cnn_{split}.png")
        graficar_matriz_confusion(y, pred_b, f"Línea base - matriz de confusión ({split})", salida / f"matriz_baseline_{split}.png")

        pd.DataFrame({"archivo": [p.name for p in ds.rutas], "real": [CLASES[i] for i in y],
                      "pred_cnn": [CLASES[i] for i in pred], "confianza_cnn": probs.max(1).round(4),
                      "ok_cnn": y == pred, "pred_baseline": [CLASES[i] for i in pred_b],
                      "ok_baseline": y == pred_b}).to_csv(salida / f"predicciones_{split}.csv", index=False)

        texto = analisis_texto(m, cm, y, pred, probs, ds.rutas, split)
        (salida / f"analisis_errores_{split}.txt").write_text(texto, encoding="utf-8")
        print(texto)

        mal = list(np.where(y != pred)[0][:12])
        figura_gradcam(modelo, cfg, ds, mal, y, pred, probs, salida / f"gradcam_errores_{split}.png",
                       f"Grad-CAM de imágenes mal clasificadas ({split})")
        ok = [int(np.where((y == k) & (pred == k))[0][j]) for k in range(len(CLASES))
              for j in range(min(2, int(((y == k) & (pred == k)).sum())))]
        figura_gradcam(modelo, cfg, ds, ok, y, pred, probs, salida / f"gradcam_aciertos_{split}.png",
                       f"Grad-CAM de aciertos ({split}): ¿mira el defecto?")

    # ---------------- Robustez y latencia (sobre test) ----------------
    if "test" not in resumen:
        raise SystemExit("No hay conjunto de test procesado.")
    rob = prueba_robustez(modelo, cfg, clf, "test")
    rob.to_csv(salida / "robustez.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(rob))
    ax.bar(x - 0.2, rob["acc_cnn"], 0.4, label="CNN")
    ax.bar(x + 0.2, rob["acc_baseline"], 0.4, label="Línea base")
    ax.set_xticks(x); ax.set_xticklabels(rob["perturbacion"], rotation=25)
    ax.set_ylabel("Accuracy (test)"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Robustez ante perturbaciones")
    fig.tight_layout()
    fig.savefig(salida / "robustez.png", dpi=140)
    plt.close(fig)
    print("\nRobustez (accuracy en test):")
    print(rob.round(3).to_string(index=False))

    img_prueba = cargar_procesada(listar_split("test")[0][0])
    lat_cnn, lat_base = medir_latencias(modelo, cfg, clf, img_prueba)

    # ---------------- Tabla comparativa ----------------
    m, mb = resumen["test"]
    pert = rob[rob["perturbacion"] != "original"]
    meta_cnn, meta_base = {}, {}
    try:
        meta_cnn = json.load(open(DIR_RES / "cnn" / "metricas.json", encoding="utf-8"))
        meta_base = json.load(open(DIR_RES / "baseline" / "metricas.json", encoding="utf-8"))
    except FileNotFoundError:
        pass
    params = meta_cnn.get("params", contar_parametros(modelo))
    macs = meta_cnn.get("macs", 0)
    comp = pd.DataFrame([
        dict(metodo="Convencional (árbol de decisión)", accuracy=mb["accuracy"], f1_macro=mb["f1_macro"],
             robustez_acc_media=pert["acc_baseline"].mean(), latencia_ms_cpu=lat_base,
             complejidad=f"{meta_base.get('n_caracteristicas', 13)} características, "
                         f"{meta_base.get('n_nodos', '?')} nodos"),
        dict(metodo="CNN", accuracy=m["accuracy"], f1_macro=m["f1_macro"],
             robustez_acc_media=pert["acc_cnn"].mean(), latencia_ms_cpu=lat_cnn,
             complejidad=f"{params:,} parámetros, {macs / 1e6:.1f} M MACs"),
    ])
    comp.to_csv(salida / "comparacion_cnn_vs_baseline.csv", index=False)
    por_clase = pd.DataFrame({"clase": CLASES,
                              "recall_cnn": [m["por_clase"][c]["recall"] for c in CLASES],
                              "f1_cnn": [m["por_clase"][c]["f1"] for c in CLASES],
                              "recall_baseline": [mb["por_clase"][c]["recall"] for c in CLASES],
                              "f1_baseline": [mb["por_clase"][c]["f1"] for c in CLASES]})
    por_clase.to_csv(salida / "comparacion_por_clase.csv", index=False)
    print("\n=== Comparación método convencional vs CNN (test) ===")
    print(comp.round(3).to_string(index=False))
    print("\n=== Por clase ===")
    print(por_clase.round(3).to_string(index=False))
    print(f"\nTodo guardado en {salida}")
    print("Recuerda: con ~12 imágenes de test por clase, un solo error cambia el accuracy ~3 puntos.")




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
        print("\n" + "=" * 78 + f"\n PASO {n}: {nombre}\n" + "=" * 78)
        fn()
        if n == 4 and args.estudio:
            print("\n" + "=" * 78 + "\n ESTUDIO DE HIPERPARÁMETROS\n" + "=" * 78)
            main_estudio()
