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

from config import *  # [IMPORT_LOCAL]
from utils_comunes import *  # [IMPORT_LOCAL]


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


if __name__ == "__main__":  # [QUITAR]
    ap = argparse.ArgumentParser()  # [QUITAR]
    ap.add_argument("--seleccionar-roi", action="store_true")  # [QUITAR]
    main_preprocesar(ap.parse_args().seleccionar_roi)  # [QUITAR]
