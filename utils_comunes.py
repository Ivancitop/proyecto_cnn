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

from config import *  # [IMPORT_LOCAL]

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
