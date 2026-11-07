"""Paso 6: evaluación de domain shift en condiciones A, B y C.

Estructura esperada:
    pruebas/
      condicionA/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...
      condicionB/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...
      condicionC/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...

Las etiquetas se obtienen del prefijo de cada archivo mediante utils_comunes.py.
Se reutilizan ROI, tamaño, escala de grises y normalización del entrenamiento.
Para diagnosticar fallos del ROI automático, --roi-manual permite dibujar un ROI
individual por imagen y guardar sus coordenadas en un JSON reutilizable.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import CLASES, DIR_PROC, DIR_RES, IMG_BASE, MARGEN_ROI
from utils_comunes import (
    calcular_roi,
    detectar_clase,
    etiquetas_mostrar,
    hacer_cuadrado,
    leer_imagen,
    listar_imagenes_por_clase,
    preprocesar_imagen,
)
from paso4_entrenar_cnn import construir_modelo, imagen_a_tensor


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONDICIONES = ("condicionA", "condicionB", "condicionC")


def cargar_checkpoint(ruta: Path, device: torch.device):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {ruta}\n"
            "Primero ejecuta: python paso4_entrenar_cnn.py"
        )
    try:
        ck = torch.load(ruta, map_location=device, weights_only=False)
    except TypeError:  # Compatibilidad con versiones antiguas de PyTorch.
        ck = torch.load(ruta, map_location=device)
    if not isinstance(ck, dict) or "state_dict" not in ck or "cfg" not in ck:
        raise ValueError("El checkpoint no tiene el formato esperado: state_dict + cfg.")
    clases_ck = ck.get("clases", CLASES)
    if list(clases_ck) != list(CLASES):
        raise ValueError(f"Las clases del checkpoint {clases_ck} no coinciden con config.py {CLASES}.")
    modelo = construir_modelo(ck["cfg"]).to(device)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    return modelo, ck["cfg"]


def cargar_normalizacion():
    ruta = DIR_PROC / "normalizacion.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero paso2_preprocesamiento.py y paso4_entrenar_cnn.py."
        )
    with open(ruta, "r", encoding="utf-8") as f:
        norm = json.load(f)
    media = torch.tensor(norm["mean"], dtype=torch.float32).view(-1, 1, 1)
    desv = torch.tensor(norm["std"], dtype=torch.float32).view(-1, 1, 1).clamp_min(1e-8)
    return media, desv


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(CLASES)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    por_clase: dict[str, dict[str, float | int]] = {}
    fprs, fnrs = [], []
    total = int(cm.sum())
    for i, clase in enumerate(CLASES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)
        por_clase[clase] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
            "soporte": int(support[i]), "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        }

    # Métricas binarias de inspección: normal = pieza buena; las otras dos = defecto.
    real_defecto = np.asarray(y_true) != 0
    pred_defecto = np.asarray(y_pred) != 0
    tp = int(np.sum(real_defecto & pred_defecto))       # defecto detectado y rechazado
    tn = int(np.sum(~real_defecto & ~pred_defecto))     # pieza buena aceptada
    fp = int(np.sum(~real_defecto & pred_defecto))      # falsa alarma: pieza buena rechazada
    fn = int(np.sum(real_defecto & ~pred_defecto))      # falso aceptado: pieza defectuosa aceptada
    binary_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    binary_fnr = fn / (fn + tp) if (fn + tp) else 0.0
    binary_precision = tp / (tp + fp) if (tp + fp) else 0.0
    binary_recall = tp / (tp + fn) if (tp + fn) else 0.0
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if (binary_precision + binary_recall) else 0.0
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f1)),
        "fpr_macro_ovr": float(np.mean(fprs)),
        "fnr_macro_ovr": float(np.mean(fnrs)),
        "por_clase": por_clase,
        "inspeccion_binaria": {
            "verdaderos_positivos_defecto": tp,
            "verdaderos_negativos_normal": tn,
            "falsos_positivos_falsa_alarma": fp,
            "falsos_negativos_falso_aceptado": fn,
            "false_positive_rate": float(binary_fpr),
            "false_negative_rate": float(binary_fnr),
            "precision_rechazo": float(binary_precision),
            "recall_defectos": float(binary_recall),
            "f1_defectos": float(binary_f1),
        },
        "matriz_confusion": cm.tolist(),
    }


def guardar_matriz(cm: np.ndarray, titulo: str, destino: Path):
    nombres = etiquetas_mostrar()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASES)), nombres, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASES)), nombres)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def seleccionar_roi_manual(img_bgr: np.ndarray, etiqueta: str) -> tuple[int, int, int, int]:
    """Permite dibujar un ROI con el mouse y devuelve un ROI cuadrado en pixeles originales."""
    alto, ancho = img_bgr.shape[:2]
    max_ancho, max_alto = 1200, 800
    escala = min(1.0, max_ancho / ancho, max_alto / alto)
    if escala < 1.0:
        vista = cv2.resize(
            img_bgr,
            (int(round(ancho * escala)), int(round(alto * escala))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        vista = img_bgr

    nombre_ventana = f"ROI manual - {etiqueta}"
    print(f"\nSelecciona toda la pieza en: {etiqueta}")
    print("Arrastra con el mouse y confirma con ENTER o ESPACIO. C cancela la selección.")
    x, y, w, h = cv2.selectROI(
        nombre_ventana, vista, showCrosshair=True, fromCenter=False
    )
    cv2.destroyWindow(nombre_ventana)

    if w <= 0 or h <= 0:
        raise RuntimeError("Selección manual cancelada o vacía.")

    # cv2.selectROI trabaja sobre la vista escalada; regresamos a coordenadas originales.
    if escala < 1.0:
        x = int(round(x / escala))
        y = int(round(y / escala))
        w = int(round(w / escala))
        h = int(round(h / escala))

    x = max(0, min(int(x), ancho - 1))
    y = max(0, min(int(y), alto - 1))
    w = max(1, min(int(w), ancho - x))
    h = max(1, min(int(h), alto - y))

    # Convierte el rectángulo a cuadrado y agrega el mismo margen definido en config.py.
    return tuple(int(v) for v in hacer_cuadrado((x, y, w, h), ancho, alto, MARGEN_ROI))


def cargar_rois_manuales(ruta: Path) -> dict[str, dict[str, Any]]:
    if not ruta.exists():
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    if not isinstance(datos, dict):
        raise ValueError(f"El archivo de ROIs manuales no contiene un diccionario: {ruta}")
    return datos


def guardar_rois_manuales(rois: dict[str, dict[str, Any]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)
    temporal.replace(ruta)


def obtener_roi_para_imagen(
    img_bgr: np.ndarray,
    ruta_imagen: Path,
    carpeta_pruebas: Path,
    condicion: str,
    condiciones_manuales: set[str],
    rois_guardados: dict[str, dict[str, Any]],
    ruta_rois: Path,
    forzar_seleccion: bool = False,
) -> tuple[int, int, int, int] | None:
    """ROI individual. Devuelve None para conservar la detección automática."""
    if condicion not in condiciones_manuales:
        return None

    clave = ruta_imagen.relative_to(carpeta_pruebas).as_posix()
    alto, ancho = img_bgr.shape[:2]
    registro = rois_guardados.get(clave)

    if registro and not forzar_seleccion:
        forma_guardada = registro.get("shape")
        roi_guardado = registro.get("roi")
        if forma_guardada == [alto, ancho] and isinstance(roi_guardado, list) and len(roi_guardado) == 4:
            x, y, w, h = [int(v) for v in roi_guardado]
            if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= ancho and y + h <= alto:
                return (x, y, w, h)
        print(f"[ROI] Se volverá a seleccionar {clave}: cambió el tamaño o el registro no es válido.")

    roi = seleccionar_roi_manual(img_bgr, ruta_imagen.name)
    rois_guardados[clave] = {"roi": list(roi), "shape": [alto, ancho]}
    guardar_rois_manuales(rois_guardados, ruta_rois)
    return roi


def inferir_una(
    img_bgr: np.ndarray,
    modelo: torch.nn.Module,
    cfg: dict,
    media: torch.Tensor,
    desv: torch.Tensor,
    device: torch.device,
    roi_override: tuple[int, int, int, int] | None = None,
) -> tuple[int, np.ndarray, float, float]:
    t0 = time.perf_counter()
    roi = roi_override if roi_override is not None else calcular_roi(img_bgr)
    procesada = preprocesar_imagen(img_bgr, roi=roi, tam=int(cfg["img_size"]))
    x = imagen_a_tensor(procesada, int(cfg["img_size"]))
    x = ((x - media) / desv).unsqueeze(0).to(device)
    t_pre = (time.perf_counter() - t0) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = (time.perf_counter() - t1) * 1000.0
    return int(np.argmax(probs)), probs, t_pre, t_inf


def main():
    parser = argparse.ArgumentParser(description="Evalúa domain shift en condiciones A, B y C.")
    parser.add_argument("--pruebas", type=Path, default=Path("pruebas"),
                        help="Carpeta que contiene condicionA, condicionB y condicionC.")
    parser.add_argument("--modelo", type=Path, default=DIR_RES / "cnn" / "modelo_mejor.pt")
    parser.add_argument("--salida", type=Path, default=DIR_RES / "paso6_domain_shift")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--roi-manual",
        choices=("condicionA", "condicionB", "condicionC", "todas"),
        default=None,
        help="Selecciona manualmente un ROI distinto por imagen para la condición indicada.",
    )
    parser.add_argument(
        "--roi-json", type=Path, default=None,
        help="JSON donde se guardan/reutilizan las coordenadas de los ROIs manuales.",
    )
    parser.add_argument(
        "--forzar-roi", action="store_true",
        help="Vuelve a pedir el ROI aunque ya exista una selección guardada.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    modelo, cfg = cargar_checkpoint(args.modelo, device)
    media, desv = cargar_normalizacion()
    args.salida.mkdir(parents=True, exist_ok=True)

    if args.roi_manual == "todas":
        condiciones_manuales = set(CONDICIONES)
    elif args.roi_manual:
        condiciones_manuales = {args.roi_manual}
    else:
        condiciones_manuales = set()

    ruta_rois = args.roi_json or (args.salida / "rois_manuales.json")
    rois_guardados = cargar_rois_manuales(ruta_rois) if condiciones_manuales else {}

    resumen: dict[str, Any] = {
        "modelo": str(args.modelo), "device": str(device), "condiciones": {},
        "roi_manual_condiciones": sorted(condiciones_manuales),
        "roi_manual_json": str(ruta_rois) if condiciones_manuales else None,
        "nota_roi": (
            "Las condiciones indicadas usan ROI manual individual por imagen; "
            "las demás usan ROI automático. Esta ejecución sirve para diagnóstico, "
            "no para reportar el rendimiento autónomo del detector de ROI."
        ) if condiciones_manuales else "Todas las condiciones usan ROI automático.",
    }
    filas_resumen: list[dict[str, Any]] = []
    filas_imagenes: list[dict[str, Any]] = []
    ignoradas: list[dict[str, str]] = []

    for condicion in CONDICIONES:
        carpeta = args.pruebas / condicion
        if not carpeta.is_dir():
            print(f"[AVISO] No existe {carpeta}; se omite.")
            continue
        por_clase, no_clasificadas = listar_imagenes_por_clase(carpeta)
        for p in no_clasificadas:
            ignoradas.append({"condicion": condicion, "archivo": str(p), "motivo": "prefijo de clase no reconocido"})
        items = [(p, CLASES.index(clase)) for clase in CLASES for p in por_clase[clase]]
        if not items:
            print(f"[AVISO] No se encontraron imágenes etiquetadas en {carpeta}.")
            continue

        y_true, y_pred = [], []
        tiempos_pre, tiempos_inf = [], []
        for ruta, real in items:
            try:
                img = leer_imagen(ruta)
                if img is None:
                    raise ValueError("imagen ilegible")
                roi_override = obtener_roi_para_imagen(
                    img, ruta, args.pruebas, condicion, condiciones_manuales,
                    rois_guardados, ruta_rois, args.forzar_roi,
                )
                pred, probs, t_pre, t_inf = inferir_una(
                    img, modelo, cfg, media, desv, device, roi_override=roi_override
                )
                y_true.append(real)
                y_pred.append(pred)
                tiempos_pre.append(t_pre)
                tiempos_inf.append(t_inf)
                fila = {
                    "condicion": condicion, "archivo": str(ruta), "clase_real": CLASES[real],
                    "clase_predicha": CLASES[pred], "correcta": bool(real == pred),
                    "confianza": float(np.max(probs)),
                    "tiempo_preprocesamiento_ms": t_pre, "tiempo_inferencia_ms": t_inf,
                    "roi_modo": "manual" if condicion in condiciones_manuales else "automatico",
                }
                for i, clase in enumerate(CLASES):
                    fila[f"prob_{clase}"] = float(probs[i])
                filas_imagenes.append(fila)
            except Exception as exc:
                ignoradas.append({"condicion": condicion, "archivo": str(ruta), "motivo": str(exc)})
                print(f"[AVISO] {ruta.name}: {exc}")

        if not y_true:
            continue
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)
        m = calcular_metricas(y_t, y_p)
        m["tiempo_preprocesamiento_ms_media"] = float(np.mean(tiempos_pre))
        m["tiempo_inferencia_ms_media"] = float(np.mean(tiempos_inf))
        m["tiempo_total_preproceso_inferencia_ms_media"] = float(np.mean(np.asarray(tiempos_pre) + np.asarray(tiempos_inf)))
        resumen["condiciones"][condicion] = m
        filas_resumen.append({
            "condicion": condicion, "n": m["n"], "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"], "fpr_macro_ovr": m["fpr_macro_ovr"],
            "fnr_macro_ovr": m["fnr_macro_ovr"],
            "fpr_binario": m["inspeccion_binaria"]["false_positive_rate"],
            "fnr_binario": m["inspeccion_binaria"]["false_negative_rate"],
            "preprocesamiento_ms": m["tiempo_preprocesamiento_ms_media"],
            "inferencia_ms": m["tiempo_inferencia_ms_media"],
        })
        guardar_matriz(np.asarray(m["matriz_confusion"]), f"Matriz de confusión - {condicion}",
                       args.salida / f"matriz_confusion_{condicion}.png")
        print(
            f"[{condicion}] n={m['n']} | acc={m['accuracy']:.3f} | "
            f"P={m['precision_macro']:.3f} | R={m['recall_macro']:.3f} | "
            f"F1={m['f1_macro']:.3f} | FPR={m['fpr_macro_ovr']:.3f} | FNR={m['fnr_macro_ovr']:.3f}"
        )

    if not resumen["condiciones"]:
        raise SystemExit("No se evaluó ninguna imagen. Revisa las carpetas y los prefijos normal/rayadura/defectuosa.")

    # Degradación respecto de A. Los valores negativos representan una mejora.
    if "condicionA" in resumen["condiciones"]:
        base = resumen["condiciones"]["condicionA"]
        for fila in filas_resumen:
            if fila["condicion"] == "condicionA":
                fila["degradacion_accuracy_pp_vs_A"] = 0.0
                fila["degradacion_accuracy_relativa_pct_vs_A"] = 0.0
                fila["degradacion_f1_pp_vs_A"] = 0.0
            else:
                actual = resumen["condiciones"][fila["condicion"]]
                delta_acc = base["accuracy"] - actual["accuracy"]
                fila["degradacion_accuracy_pp_vs_A"] = 100.0 * delta_acc
                fila["degradacion_accuracy_relativa_pct_vs_A"] = (
                    100.0 * delta_acc / base["accuracy"] if base["accuracy"] else None
                )
                fila["degradacion_f1_pp_vs_A"] = 100.0 * (base["f1_macro"] - actual["f1_macro"])
        resumen["referencia_degradacion"] = "condicionA"
    else:
        resumen["nota_degradacion"] = "No se calculó degradación: condicionA no estuvo disponible."

    with open(args.salida / "metricas_por_condicion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    if filas_resumen:
        campos = list(dict.fromkeys(k for fila in filas_resumen for k in fila))
        with open(args.salida / "comparacion_condiciones.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas_resumen)
    if filas_imagenes:
        campos = list(dict.fromkeys(k for fila in filas_imagenes for k in fila))
        with open(args.salida / "predicciones_por_imagen.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas_imagenes)
    if ignoradas:
        with open(args.salida / "imagenes_omitidas.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["condicion", "archivo", "motivo"])
            w.writeheader()
            w.writerows(ignoradas)

    # Gráfica de accuracy y F1 para comparar el cambio de dominio.
    if filas_resumen:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = np.arange(len(filas_resumen))
        ancho = 0.36
        ax.bar(xs - ancho / 2, [f["accuracy"] for f in filas_resumen], ancho, label="Accuracy")
        ax.bar(xs + ancho / 2, [f["f1_macro"] for f in filas_resumen], ancho, label="F1 macro")
        ax.set_xticks(xs, [f["condicion"] for f in filas_resumen])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Métrica")
        ax.set_title("Rendimiento ante cambios de dominio")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.salida / "comparacion_accuracy_f1.png", dpi=150)
        plt.close(fig)

    print(f"\nResultados guardados en: {args.salida.resolve()}")
    print("FPR/FNR OVR son promedios one-vs-rest de las tres clases.")
    print("En inspección binaria, FP = pieza normal rechazada y FN = pieza defectuosa aceptada.")


if __name__ == "__main__":
    main()
