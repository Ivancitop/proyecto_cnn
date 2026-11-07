"""Paso 6: evaluación de domain shift en condiciones A, B y C.

Estructura esperada:
    pruebas/
      condicionA/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...
      condicionB/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...
      condicionC/  normal_01.jpg, rayadura_01.jpg, defectuosa_01.jpg, ...

Las etiquetas se obtienen del prefijo de cada archivo mediante utils_comunes.py.
Se reutilizan ROI, tamaño, escala de grises y normalización del entrenamiento.
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

from config import CLASES, DIR_PROC, DIR_RES, IMG_BASE
from utils_comunes import (
    calcular_roi,
    detectar_clase,
    guardar_imagen,
    etiquetas_mostrar,
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

def inferir_una(
    img_bgr: np.ndarray,
    modelo: torch.nn.Module,
    cfg: dict,
    media: torch.Tensor,
    desv: torch.Tensor,
    device: torch.device,
) -> tuple[int, np.ndarray, float, float]:
    t0 = time.perf_counter()
    roi = calcular_roi(img_bgr)
    procesada = preprocesar_imagen(img_bgr, roi=roi, tam=int(cfg["img_size"]))
    guardar_imagen(DIR_PROC / (str(t0) + ".png"), procesada)
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
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    modelo, cfg = cargar_checkpoint(args.modelo, device)
    media, desv = cargar_normalizacion()
    args.salida.mkdir(parents=True, exist_ok=True)

    resumen: dict[str, Any] = {"modelo": str(args.modelo), "device": str(device), "condiciones": {}}
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
                pred, probs, t_pre, t_inf = inferir_una(img, modelo, cfg, media, desv, device)
                y_true.append(real)
                y_pred.append(pred)
                tiempos_pre.append(t_pre)
                tiempos_inf.append(t_inf)
                fila = {
                    "condicion": condicion, "archivo": str(ruta), "clase_real": CLASES[real],
                    "clase_predicha": CLASES[pred], "correcta": bool(real == pred),
                    "confianza": float(np.max(probs)),
                    "tiempo_preprocesamiento_ms": t_pre, "tiempo_inferencia_ms": t_inf,
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
