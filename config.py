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
USAR_GRIS = True         # True -> escala de grises (1 canal); False -> color (3 canales)

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
    img_size=IMG_BASE,      # tamaño de entrada (<= IMG_BASE recomendado)
    n_filtros=16,      # filtros de la 1.ª capa (se duplican en cada capa)
    n_capas=6,         # bloques convolucionales
    lr=5e-3,
    batch_size=16,
    epocas=25,
    dropout=0.3,
    paciencia=None,      # early stopping sobre val_loss (None = desactivado)
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
