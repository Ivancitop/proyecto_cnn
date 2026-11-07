"""Paso 8: inspección visual en tiempo real, ejecutada completamente en la PC.

Modos:
  --modo camara : clasifica una pieza detectada dentro del encuadre completo.
  --modo banda  : inspecciona una zona fija; se recomienda proporcionar --roi.

La captura, el preprocesamiento, la inferencia y la decisión ACCEPTAR/RECHAZAR
se ejecutan localmente con OpenCV y PyTorch. El resultado se muestra en la ventana
de inspección y en consola; no se utiliza comunicación serial, GPIO ni hardware
externo para accionar salidas.

Cada pieza se clasifica una sola vez mientras permanece en la zona. Al desaparecer
se rearma el detector para la siguiente pieza. Las etiquetas reales se pueden
introducir con el teclado para calcular métricas online: n=normal, r=rayadura,
d=defectuosa. Estas teclas etiquetan la pieza pendiente más antigua.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config import CLASES, DIR_PROC, DIR_RES, MARGEN_ROI
from utils_comunes import hacer_cuadrado, leer_imagen, mascara_pieza, preprocesar_imagen
from paso4_entrenar_cnn import construir_modelo, imagen_a_tensor


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def cargar_normalizacion():
    ruta = DIR_PROC / "normalizacion.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}; primero ejecuta el paso 2 y paso 4.")
    with open(ruta, "r", encoding="utf-8") as f:
        norm = json.load(f)
    media = torch.tensor(norm["mean"], dtype=torch.float32).view(-1, 1, 1)
    desv = torch.tensor(norm["std"], dtype=torch.float32).view(-1, 1, 1).clamp_min(1e-8)
    return media, desv


def resolver_modelo(tipo: str, personalizado: str | None) -> tuple[Path, bool]:
    if personalizado:
        path = Path(personalizado)
        return path, tipo == "int8"
    base = DIR_RES / "cnn" / "modelo_mejor.pt"
    paso7 = DIR_RES / "paso7"
    if tipo == "recomendado":
        meta = paso7 / "seleccion_modelo.json"
        if meta.exists():
            try:
                tipo = json.loads(meta.read_text(encoding="utf-8")).get("modelo_recomendado", "podado")
            except Exception:
                tipo = "podado"
        else:
            tipo = "podado" if (paso7 / "modelo_podado.pt").exists() else "fp32"
    opciones = {
        "fp32": base,
        "podado": paso7 / "modelo_podado.pt",
        "int8": paso7 / "modelo_int8_torchscript.pt",
    }
    path = opciones[tipo]
    if tipo == "podado" and not path.exists():
        print("[AVISO] No se encontró el modelo podado del paso 7; se usará el modelo del paso 4.")
        path = base
    return path, tipo == "int8"


def cargar_modelo(path: Path, es_torchscript: bool, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo: {path}")
    if es_torchscript:
        model = torch.jit.load(str(path), map_location="cpu").eval()
        return model, None, torch.device("cpu")
    ck = torch_load(path, device)
    if not isinstance(ck, dict) or "state_dict" not in ck or "cfg" not in ck:
        raise ValueError("Checkpoint inválido: se esperaba state_dict y cfg.")
    if list(ck.get("clases", CLASES)) != list(CLASES):
        raise ValueError("Las clases guardadas en el checkpoint no coinciden con config.py.")
    model = construir_modelo(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck["cfg"], device


def parsear_roi(texto: str | None):
    if not texto:
        return None
    try:
        vals = tuple(int(v.strip()) for v in texto.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("La ROI debe tener formato x,y,w,h.") from exc
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise argparse.ArgumentTypeError("La ROI debe tener formato x,y,w,h con ancho y alto positivos.")
    return vals


def detectar_roi(img_bgr: np.ndarray, area_minima: float):
    """Detecta el contorno dominante igual que el ROI automático del proyecto, sin fallback central."""
    h, w = img_bgr.shape[:2]
    if h < 2 or w < 2:
        return None
    escala = min(1.0, 480.0 / max(h, w))
    reducido = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else img_bgr
    gris = cv2.cvtColor(reducido, cv2.COLOR_BGR2GRAY)
    mascara = mascara_pieza(gris)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    if cw * ch < area_minima * reducido.shape[0] * reducido.shape[1]:
        return None
    caja = tuple(int(round(v / escala)) for v in (x, y, cw, ch))
    return hacer_cuadrado(caja, w, h, MARGEN_ROI)



def calcular_metricas_registros(registros: list[dict[str, Any]]) -> dict[str, Any]:
    etiquetados = [r for r in registros if r.get("clase_real") in CLASES]
    total = len(registros)
    aceptadas = sum(r["decision"] == "ACEPTAR" for r in registros)
    rechazadas = total - aceptadas
    resumen: dict[str, Any] = {
        "piezas_inspeccionadas": total,
        "piezas_predichas_aceptar": aceptadas,
        "piezas_predichas_rechazar": rechazadas,
        "piezas_con_etiqueta_real": len(etiquetados),
        "piezas_sin_etiqueta_real": total - len(etiquetados),
    }
    if not etiquetados:
        resumen["metricas_clasificacion"] = None
        resumen["metricas_binarias_inspeccion"] = None
        return resumen

    y = np.array([CLASES.index(r["clase_real"]) for r in etiquetados], dtype=int)
    p = np.array([CLASES.index(r["clase_predicha"]) for r in etiquetados], dtype=int)
    cm = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    for a, b in zip(y, p):
        cm[a, b] += 1
    per_class = {}
    precisiones, recalls, f1s, fprs, fnrs = [], [], [], [], []
    n = int(cm.sum())
    for i, clase in enumerate(CLASES):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(n - tp - fn - fp)
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        per_class[clase] = {"precision": pr, "recall": re, "f1": f1, "soporte": int(cm[i].sum()),
                            "fpr_ovr": fpr, "fnr_ovr": fnr}
        precisiones.append(pr); recalls.append(re); f1s.append(f1); fprs.append(fpr); fnrs.append(fnr)

    actual_defecto = y != 0
    decision_rechazo = np.array([r["decision"] == "RECHAZAR" for r in etiquetados], dtype=bool)
    tp_b = int(np.sum(actual_defecto & decision_rechazo))
    tn_b = int(np.sum(~actual_defecto & ~decision_rechazo))
    fp_b = int(np.sum(~actual_defecto & decision_rechazo))  # falsa alarma/rechazo de pieza normal
    fn_b = int(np.sum(actual_defecto & ~decision_rechazo))  # falso aceptado de pieza defectuosa
    resumen["metricas_clasificacion"] = {
        "accuracy": float(np.mean(y == p)),
        "precision_macro": float(np.mean(precisiones)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "fpr_macro_ovr": float(np.mean(fprs)),
        "fnr_macro_ovr": float(np.mean(fnrs)),
        "por_clase": per_class,
        "matriz_confusion": cm.tolist(),
    }
    precision_rechazo = tp_b / (tp_b + fp_b) if tp_b + fp_b else 0.0
    recall_defecto = tp_b / (tp_b + fn_b) if tp_b + fn_b else 0.0
    resumen["metricas_binarias_inspeccion"] = {
        "correctamente_aceptadas": tn_b,
        "correctamente_rechazadas": tp_b,
        "falsos_positivos_pieza_buena_rechazada": fp_b,
        "falsos_negativos_pieza_defectuosa_aceptada": fn_b,
        "false_positive_rate": fp_b / (fp_b + tn_b) if fp_b + tn_b else 0.0,
        "false_negative_rate": fn_b / (fn_b + tp_b) if fn_b + tp_b else 0.0,
        "precision_rechazo": precision_rechazo,
        "recall_defectos": recall_defecto,
        "f1_defectos": 2 * precision_rechazo * recall_defecto / (precision_rechazo + recall_defecto)
        if precision_rechazo + recall_defecto else 0.0,
    }
    return resumen


def guardar_matriz_confusion(cm: list[list[int]], path: Path):
    mat = np.asarray(cm)
    fig, ax = cv2_to_plt(mat)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def cv2_to_plt(mat: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(CLASES)), CLASES, rotation=20)
    ax.set_yticks(range(len(CLASES)), CLASES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión - inspección en tiempo real")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description="Inspección CNN en tiempo real con cámara o banda transportadora.")
    ap.add_argument("--modo", choices=("camara", "banda"), default="camara")
    ap.add_argument("--camara", default="0", help="Índice de cámara (0) o ruta de dispositivo.")
    ap.add_argument("--roi", default=None, help="Zona de inspección x,y,w,h dentro del frame.")
    ap.add_argument("--tipo-modelo", choices=("recomendado", "fp32", "podado", "int8"), default="recomendado")
    ap.add_argument("--modelo", default=None, help="Ruta opcional del checkpoint o TorchScript.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--salida", type=Path, default=DIR_RES / "paso8_tiempo_real")
    ap.add_argument("--area-minima", type=float, default=0.05, help="Área mínima relativa de la caja detectada.")
    ap.add_argument("--frames-estables", type=int, default=3)
    ap.add_argument("--frames-salida", type=int, default=5)
    ap.add_argument("--umbral-normal", type=float, default=0.0,
                    help="Umbral mínimo de P(normal) para aceptar; 0 desactiva el umbral extra.")
    ap.add_argument("--velocidad-banda-mm-s", type=float, default=0.0)
    ap.add_argument("--longitud-zona-mm", type=float, default=0.0,
                    help="Longitud útil de la zona de inspección en dirección del movimiento.")
    ap.add_argument("--headless", action="store_true", help="Ejecuta sin ventana OpenCV.")
    ap.add_argument("--ancho", type=int, default=0, help="Ancho solicitado de cámara; 0 conserva el predeterminado.")
    ap.add_argument("--alto", type=int, default=0, help="Alto solicitado de cámara; 0 conserva el predeterminado.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Detiene la captura después de N frames; 0 = hasta pulsar q o Ctrl+C.")
    args = ap.parse_args()

    zona_roi = parsear_roi(args.roi)
    if args.modo == "banda" and zona_roi is None:
        raise SystemExit("En modo banda especifica --roi x,y,w,h para delimitar la zona fija de inspección.")
    if args.frames_estables < 1 or args.frames_salida < 1:
        raise SystemExit("frames-estables y frames-salida deben ser >= 1.")
    if not 0.0 <= args.area_minima <= 1.0:
        raise SystemExit("--area-minima debe estar entre 0 y 1.")
    if not 0.0 <= args.umbral_normal <= 1.0:
        raise SystemExit("--umbral-normal debe estar entre 0 y 1.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Se solicitó CUDA, pero no hay GPU CUDA disponible.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model_path, es_script = resolver_modelo(args.tipo_modelo, args.modelo)
    model, cfg, device_model = cargar_modelo(model_path, es_script, device)
    if es_script:
        # Usa el cfg original del checkpoint que se guardó en seleccion_modelo.json.
        metadata = model_path.parent / "seleccion_modelo.json"
        if metadata.exists():
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            cfg = meta.get("cfg_entrenamiento")
        if not cfg:
            with open(DIR_PROC / "normalizacion.json", "r", encoding="utf-8") as f:
                cfg = {"img_size": int(json.load(f).get("img_base", 128))}
    assert cfg is not None
    media, desv = cargar_normalizacion()

    cam_id: int | str = int(args.camara) if str(args.camara).isdigit() else str(args.camara)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara/dispositivo: {args.camara}")
    if args.ancho > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    if args.alto > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)

    args.salida.mkdir(parents=True, exist_ok=True)
    registros: list[dict[str, Any]] = []
    pendientes: deque[int] = deque()
    tiempos_captura, tiempos_pre, tiempos_inf, tiempos_total, tiempos_ciclo = [], [], [], [], []
    cpu_samples, ram_samples = [], []
    try:
        import psutil
        proceso = psutil.Process(os.getpid())
        proceso.cpu_percent(None)
        psutil_disponible = True
    except Exception:
        proceso = None
        psutil_disponible = False

    frames = 0
    inicio_sistema = time.perf_counter()
    primera_deteccion = None
    consecutivos_presentes = 0
    consecutivos_ausentes = 0
    pieza_ocupando_zona = False
    ultima_decision = "ESPERANDO PIEZA"
    ultimo_detalle = ""
    ultimo_roi_global = None
    cpu_muestreo = time.perf_counter()
    camara_nombre = str(cam_id)

    print(f"Modelo: {model_path} | modo={args.modo} | device={device_model}")
    if zona_roi:
        print(f"ROI de inspección: {zona_roi}")
    print("Teclas: n=etiqueta real normal | r=rayadura | d=defectuosa | q=salir")

    try:
        while True:
            t_loop = time.perf_counter()
            t_cap = time.perf_counter()
            ok, frame = cap.read()
            t_captura = (time.perf_counter() - t_cap) * 1000.0
            if not ok or frame is None:
                # Si se pasó un archivo de video en --camara, EOF termina el experimento.
                if isinstance(cam_id, str) and Path(cam_id).is_file():
                    break
                print("[AVISO] No se pudo leer un frame; se intenta continuar.")
                if not args.headless:
                    if cv2.waitKey(20) & 0xFF == ord("q"):
                        break
                continue
            frames += 1
            if args.max_frames > 0 and frames > args.max_frames:
                break
            tiempos_captura.append(t_captura)

            if zona_roi:
                xz, yz, wz, hz = zona_roi
                h_frame, w_frame = frame.shape[:2]
                x1, y1 = max(0, xz), max(0, yz)
                x2, y2 = min(w_frame, xz + wz), min(h_frame, yz + hz)
                zona = frame[y1:y2, x1:x2]
                if zona.size == 0:
                    raise RuntimeError("La ROI configurada queda fuera del frame de cámara.")
            else:
                x1, y1 = 0, 0
                zona = frame

            t_pre = time.perf_counter()
            roi_local = detectar_roi(zona, args.area_minima)
            t_pre_detect = (time.perf_counter() - t_pre) * 1000.0
            ahora = time.perf_counter()
            pieza_presente = roi_local is not None
            if pieza_presente:
                consecutivos_ausentes = 0
                if consecutivos_presentes == 0:
                    primera_deteccion = ahora
                consecutivos_presentes += 1
                if not pieza_ocupando_zona and consecutivos_presentes >= args.frames_estables:
                    t_pre2 = time.perf_counter()
                    procesada = preprocesar_imagen(zona, roi=roi_local, tam=int(cfg["img_size"]))
                    x = imagen_a_tensor(procesada, int(cfg["img_size"]))
                    x = ((x - media) / desv).unsqueeze(0).to(device_model)
                    t_pre_total = (time.perf_counter() - t_pre2) * 1000.0 + t_pre_detect
                    if device_model.type == "cuda":
                        torch.cuda.synchronize(device_model)
                    t_inf = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                    if device_model.type == "cuda":
                        torch.cuda.synchronize(device_model)
                    t_inf_ms = (time.perf_counter() - t_inf) * 1000.0
                    t_dec = time.perf_counter()
                    clase_idx = int(np.argmax(probs))
                    clase_pred = CLASES[clase_idx]
                    p_normal = float(probs[0])
                    aceptar = clase_idx == 0 and p_normal >= args.umbral_normal
                    decision = "ACEPTAR" if aceptar else "RECHAZAR"
                    t_dec_ms = (time.perf_counter() - t_dec) * 1000.0
                    ciclo_ms = (time.perf_counter() - primera_deteccion) * 1000.0 if primera_deteccion else None
                    total_ms = t_captura + t_pre_total + t_inf_ms + t_dec_ms
                    tiempos_pre.append(t_pre_total)
                    tiempos_inf.append(t_inf_ms)
                    tiempos_total.append(total_ms)
                    if ciclo_ms is not None:
                        tiempos_ciclo.append(ciclo_ms)

                    secuencia = len(registros) + 1
                    registro = {
                        "secuencia": secuencia,
                        "fecha_hora": datetime.now().isoformat(timespec="milliseconds"),
                        "clase_real": "",
                        "clase_predicha": clase_pred,
                        "confianza": float(np.max(probs)),
                        "prob_normal": float(probs[0]),
                        "prob_rayadura": float(probs[1]),
                        "prob_defectuosa": float(probs[2]),
                        "decision": decision,
                        "tiempo_captura_ms": t_captura,
                        "tiempo_preprocesamiento_ms": t_pre_total,
                        "tiempo_inferencia_ms": t_inf_ms,
                        "tiempo_decision_local_ms": t_dec_ms,
                        "latencia_total_ms": total_ms,
                        "tiempo_ciclo_ms": ciclo_ms,
                    }
                    registros.append(registro)
                    pendientes.append(len(registros) - 1)
                    if proceso is not None:
                        ram_samples.append(int(proceso.memory_info().rss))
                    pieza_ocupando_zona = True
                    ultima_decision = decision
                    ultimo_detalle = f"{clase_pred} | conf={np.max(probs):.2f} | {total_ms:.1f} ms"
                    print(f"Pieza {secuencia:04d}: {clase_pred.upper():12s} -> {decision:8s} | "
                          f"infer={t_inf_ms:.2f} ms | ciclo={ciclo_ms:.2f} ms")
                    if proceso is not None and (time.perf_counter() - cpu_muestreo) >= 1.0:
                        cpu_samples.append(float(proceso.cpu_percent(None)))
                        cpu_muestreo = time.perf_counter()

                # Dibuja la ROI real en coordenadas del frame.
                rx, ry, rw, rh = roi_local
                ultimo_roi_global = (x1 + rx, y1 + ry, rw, rh)
            else:
                consecutivos_presentes = 0
                if pieza_ocupando_zona:
                    consecutivos_ausentes += 1
                    if consecutivos_ausentes >= args.frames_salida:
                        pieza_ocupando_zona = False
                        consecutivos_ausentes = 0
                        primera_deteccion = None
                        ultima_decision = "ESPERANDO PIEZA"
                        ultimo_detalle = ""
                        ultimo_roi_global = None
                else:
                    primera_deteccion = None
                    consecutivos_ausentes = 0
                    ultimo_roi_global = None

            loop_ms = (time.perf_counter() - t_loop) * 1000.0
            if not args.headless:
                vista = frame.copy()
                if zona_roi:
                    cv2.rectangle(vista, (x1, y1), (x2, y2), (255, 180, 0), 2)
                if ultimo_roi_global:
                    rx, ry, rw, rh = ultimo_roi_global
                    cv2.rectangle(vista, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                color = (0, 170, 0) if ultima_decision == "ACEPTAR" else (
                    (0, 0, 220) if ultima_decision == "RECHAZAR" else (180, 180, 180)
                )
                # Indicador visual local: reemplaza cualquier LED/buzzer físico.
                cv2.rectangle(vista, (8, 8), (min(vista.shape[1] - 8, 420), 78), (35, 35, 35), -1)
                cv2.circle(vista, (28, 30), 9, color, -1)
                cv2.putText(vista, ultima_decision, (48, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                if ultimo_detalle:
                    cv2.putText(vista, ultimo_detalle, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)
                cv2.putText(vista, f"Piezas: {len(registros)} | Pendientes etiqueta: {len(pendientes)}",
                            (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("Inspeccion CNN - q salir | n normal | r rayadura | d defectuosa", vista)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("n"), ord("r"), ord("d")):
                    if pendientes:
                        idx = pendientes.popleft()
                        etiqueta = {ord("n"): "normal", ord("r"): "rayadura", ord("d"): "defectuosa"}[key]
                        registros[idx]["clase_real"] = etiqueta
                        print(f"Etiqueta real de pieza {registros[idx]['secuencia']}: {etiqueta}")
                    else:
                        print("No hay pieza pendiente de etiquetar.")
            if psutil_disponible and proceso is not None and (time.perf_counter() - cpu_muestreo) >= 1.0:
                cpu_samples.append(float(proceso.cpu_percent(None)))
                cpu_muestreo = time.perf_counter()
    except KeyboardInterrupt:
        print("\nInterrupción del usuario.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - inicio_sistema, 1e-9)
    n_frames = len(tiempos_captura)
    mean_capture = float(np.mean(tiempos_captura)) if tiempos_captura else None
    mean_pre = float(np.mean(tiempos_pre)) if tiempos_pre else None
    mean_inf = float(np.mean(tiempos_inf)) if tiempos_inf else None
    mean_total = float(np.mean(tiempos_total)) if tiempos_total else None
    mean_cycle = float(np.mean(tiempos_ciclo)) if tiempos_ciclo else None
    max_cycle = float(np.max(tiempos_ciclo)) if tiempos_ciclo else None
    fps = n_frames / elapsed if n_frames else 0.0
    resumen = calcular_metricas_registros(registros)
    resumen.update({
        "modo": args.modo,
        "camara": camara_nombre,
        "modelo": str(model_path),
        "dispositivo": str(device_model),
        "duracion_sesion_s": elapsed,
        "frames_capturados": n_frames,
        "fps_camara_procesamiento": fps,
        "tiempo_captura_ms_media": mean_capture,
        "tiempo_preprocesamiento_ms_media": mean_pre,
        "tiempo_inferencia_ms_media": mean_inf,
        "latencia_total_ms_media": mean_total,
        "tiempo_ciclo_ms_media": mean_cycle,
        "tiempo_ciclo_ms_maximo": max_cycle,
        "tasa_inspeccion_piezas_s": len(registros) / elapsed,
        "tasa_inspeccion_piezas_min": 60.0 * len(registros) / elapsed,
        "cpu_proceso_pct_media": float(np.mean(cpu_samples)) if cpu_samples else None,
        "ram_proceso_bytes_media": float(np.mean(ram_samples)) if ram_samples else None,
        "ram_proceso_bytes_maxima": int(max(ram_samples)) if ram_samples else None,
        "tamano_archivo_modelo_bytes": int(model_path.stat().st_size),
        "consumo_energetico": "No medido: requiere un medidor de potencia para el equipo de cómputo.",
        "etiquetado_real": "Manual con teclas n/r/d; métricas calculadas solo sobre piezas etiquetadas.",
    })
    if args.velocidad_banda_mm_s > 0 and args.longitud_zona_mm > 0:
        disponible_ms = 1000.0 * args.longitud_zona_mm / args.velocidad_banda_mm_s
        resumen["velocidad_banda_mm_s"] = args.velocidad_banda_mm_s
        resumen["longitud_zona_mm"] = args.longitud_zona_mm
        resumen["tiempo_disponible_en_zona_ms"] = disponible_ms
        resumen["cumple_tiempo_linea_media"] = (mean_cycle <= disponible_ms) if mean_cycle is not None else None
        resumen["margen_tiempo_ms"] = (disponible_ms - mean_cycle) if mean_cycle is not None else None
        resumen["velocidad_maxima_estimada_mm_s"] = (
            1000.0 * args.longitud_zona_mm / mean_cycle if mean_cycle and mean_cycle > 0 else None
        )
    else:
        resumen["tiempo_disponible_en_zona_ms"] = None
        resumen["cumple_tiempo_linea_media"] = None
        resumen["nota_tiempo_linea"] = "Proporciona --velocidad-banda-mm-s y --longitud-zona-mm para comparar contra la línea."

    # Resumen de latencias y resultados por pieza.
    campos = list(dict.fromkeys(k for r in registros for k in r))
    if campos:
        with open(args.salida / "registro_inspeccion.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(registros)
    with open(args.salida / "metricas_sistema.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(args.salida / "tiempos_sistema.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor", "unidad"])
        for k, unidad in [
            ("tiempo_captura_ms_media", "ms"), ("tiempo_preprocesamiento_ms_media", "ms"),
            ("tiempo_inferencia_ms_media", "ms"), ("latencia_total_ms_media", "ms"),
            ("tiempo_ciclo_ms_media", "ms"), ("tiempo_ciclo_ms_maximo", "ms"),
            ("fps_camara_procesamiento", "frames/s"), ("tasa_inspeccion_piezas_s", "piezas/s"),
            ("tasa_inspeccion_piezas_min", "piezas/min"), ("ram_proceso_bytes_media", "bytes"),
            ("tamano_archivo_modelo_bytes", "bytes"),
        ]:
            w.writerow([k, resumen.get(k), unidad])

    mc = resumen.get("metricas_clasificacion")
    if mc and mc.get("matriz_confusion"):
        guardar_matriz_confusion(mc["matriz_confusion"], args.salida / "matriz_confusion_tiempo_real.png")

    print("\n========== RESUMEN DEL SISTEMA ==========")
    print(f"Piezas inspeccionadas: {len(registros)}")
    print(f"Tiempo de captura medio: {mean_capture:.3f} ms" if mean_capture is not None else "Tiempo captura: N/D")
    print(f"Preprocesamiento medio: {mean_pre:.3f} ms" if mean_pre is not None else "Preprocesamiento: N/D")
    print(f"Inferencia media: {mean_inf:.3f} ms" if mean_inf is not None else "Inferencia: N/D")
    print(f"Latencia total media: {mean_total:.3f} ms" if mean_total is not None else "Latencia total: N/D")
    print(f"Tiempo de ciclo medio: {mean_cycle:.3f} ms" if mean_cycle is not None else "Tiempo de ciclo: N/D")
    print(f"FPS: {fps:.2f} | inspección: {len(registros) / elapsed:.2f} piezas/s")
    if resumen.get("cumple_tiempo_linea_media") is not None:
        print(f"Tiempo disponible: {resumen['tiempo_disponible_en_zona_ms']:.2f} ms | "
              f"cumple línea: {resumen['cumple_tiempo_linea_media']}")
    if mc:
        print(f"Accuracy={mc['accuracy']:.3f} | Precision macro={mc['precision_macro']:.3f} | "
              f"Recall macro={mc['recall_macro']:.3f} | F1 macro={mc['f1_macro']:.3f}")
        mb = resumen["metricas_binarias_inspeccion"]
        print(f"Correctamente aceptadas={mb['correctamente_aceptadas']} | "
              f"correctamente rechazadas={mb['correctamente_rechazadas']} | "
              f"falsos positivos={mb['falsos_positivos_pieza_buena_rechazada']} | "
              f"falsos negativos={mb['falsos_negativos_pieza_defectuosa_aceptada']}")
    else:
        print("No hay etiquetas reales: accuracy/precision/recall/F1 y matriz de confusión no se pueden calcular.")
    print(f"Resultados guardados en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
