"""Paso 7: poda, cuantización INT8 y selección de una CNN eficiente.

La evaluación para seleccionar el modelo usa SOLO validación (dataset_proc/val).
El test y las condiciones A/B/C se reservan para la evaluación final independiente.

La poda no estructurada genera ceros en los pesos, pero por sí sola no reduce las
formas de los tensores ni garantiza menos RAM/latencia en kernels densos. La
cuantización estática INT8 es la opción principal para reducir bytes por peso.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from config import CLASES, DIR_PROC, DIR_RES
from paso4_entrenar_cnn import DatasetImagenes, construir_modelo, contar_macs, contar_parametros


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_modelo(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero paso4_entrenar_cnn.py")
    ck = torch_load(path, device)
    if not isinstance(ck, dict) or "state_dict" not in ck or "cfg" not in ck:
        raise ValueError("Checkpoint inválido: se esperaba state_dict y cfg.")
    if list(ck.get("clases", CLASES)) != list(CLASES):
        raise ValueError("Las clases del checkpoint no coinciden con config.py.")
    model = construir_modelo(ck["cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, dict(ck["cfg"])


def metricas_clasificacion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    n = len(y_true)
    cm = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        cm[int(a), int(b)] += 1
    precision, recall, f1, fprs, fnrs = [], [], [], [], []
    for i in range(len(CLASES)):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(n - tp - fp - fn)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        fprs.append(fpr)
        fnrs.append(fnr)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if n else 0.0,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "fpr_macro_ovr": float(np.mean(fprs)),
        "fnr_macro_ovr": float(np.mean(fnrs)),
    }


def sincronizar(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluar(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ys.extend(y.numpy().tolist())
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    result = metricas_clasificacion(y_true, y_pred)
    result["n_validacion"] = int(len(y_true))
    return result


@torch.inference_mode()
def medir_latencia(model: nn.Module, dataset, device: torch.device, warmup: int = 20, repeticiones: int = 100):
    model.eval()
    if len(dataset) == 0:
        return {"latencia_media_ms": None, "latencia_p50_ms": None, "latencia_p95_ms": None, "fps_batch1": None}
    x0 = dataset[0][0].unsqueeze(0).to(device)
    for _ in range(warmup):
        model(x0)
    sincronizar(device)
    tiempos = []
    n = min(repeticiones, max(1, len(dataset)))
    for i in range(n):
        x = dataset[i][0].unsqueeze(0).to(device)
        sincronizar(device)
        t0 = time.perf_counter()
        model(x)
        sincronizar(device)
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    mean = float(np.mean(tiempos))
    return {
        "latencia_media_ms": mean,
        "latencia_p50_ms": float(np.percentile(tiempos, 50)),
        "latencia_p95_ms": float(np.percentile(tiempos, 95)),
        "fps_batch1": float(1000.0 / mean) if mean > 0 else None,
    }


def aplicar_poda(modelo: nn.Module, fraccion: float) -> nn.Module:
    modelo = copy.deepcopy(modelo).cpu().eval()
    parametros = [
        (m, "weight") for m in modelo.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None
    ]
    if fraccion > 0:
        prune.global_unstructured(
            parametros,
            pruning_method=prune.L1Unstructured,
            amount=float(fraccion),
        )
        for modulo, nombre in parametros:
            prune.remove(modulo, nombre)
    return modelo


def contar_pesos(modelo: nn.Module) -> tuple[int, int, float]:
    total = no_cero = 0
    for m in modelo.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None:
            w = m.weight.detach()
            total += w.numel()
            no_cero += int(torch.count_nonzero(w).item())
    sparsity = 1.0 - no_cero / total if total else 0.0
    return total, no_cero, float(sparsity)


def guardar_checkpoint(modelo: nn.Module, cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": modelo.state_dict(), "cfg": cfg, "clases": CLASES}, path)


class RedCuantizable(nn.Module):
    """Agrega QuantStub/DeQuantStub alrededor de la CNNSimple existente."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.backbone = backbone
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.backbone(self.quant(x)))


def crear_modelo_int8(modelo_fp32: nn.Module, train_loader: DataLoader, cfg: dict, max_batches: int = 100):
    """Cuantización estática eager INT8; devuelve modelo cuantizado y backend."""
    if not hasattr(torch, "ao") or not hasattr(torch.ao, "quantization"):
        raise RuntimeError("Esta versión de PyTorch no expone torch.ao.quantization.")
    q = torch.ao.quantization
    disponibles = list(torch.backends.quantized.supported_engines)
    backend = next((x for x in ("x86", "fbgemm", "qnnpack") if x in disponibles), None)
    if backend is None:
        raise RuntimeError(f"No hay backend INT8 disponible. Backends: {disponibles}")
    torch.backends.quantized.engine = backend

    backbone = copy.deepcopy(modelo_fp32).cpu().eval()
    # CNNSimple organiza cada bloque como Conv, BatchNorm, ReLU, MaxPool.
    n_bloques = len(backbone.features) // 4
    for i in range(n_bloques):
        base = 4 * i
        q.fuse_modules(backbone.features, [str(base), str(base + 1), str(base + 2)], inplace=True)

    wrapped = RedCuantizable(backbone).cpu().eval()
    wrapped.qconfig = q.get_default_qconfig(backend)
    prepared = q.prepare(wrapped, inplace=False)
    with torch.inference_mode():
        for i, (x, _) in enumerate(train_loader):
            prepared(x.cpu())
            if i + 1 >= max_batches:
                break
    quantized = q.convert(prepared, inplace=False).eval()
    return quantized, backend


def exportar_torchscript(modelo: nn.Module, ejemplo: torch.Tensor, destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    modelo = modelo.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(modelo, ejemplo.cpu(), strict=False, check_trace=False)
        try:
            traced = torch.jit.freeze(traced.eval())
        except Exception:
            traced = traced.eval()
        traced.save(str(destino))
    return destino.stat().st_size


def tamano_checkpoint(modelo: nn.Module, cfg: dict) -> int:
    buffer = io.BytesIO()
    torch.save({"state_dict": modelo.state_dict(), "cfg": cfg, "clases": CLASES}, buffer)
    return len(buffer.getvalue())


def bytes_pesos_cuantizados(modelo: nn.Module) -> int | None:
    """Estimación del almacenamiento de pesos/bias de módulos cuantizados."""
    total = 0
    encontrados = 0
    for modulo in modelo.modules():
        try:
            peso = modulo.weight()
        except Exception:
            peso = None
        if isinstance(peso, torch.Tensor) and peso.is_quantized:
            total += peso.numel() * peso.element_size()
            encontrados += 1
            try:
                bias = modulo.bias()
            except Exception:
                bias = None
            if isinstance(bias, torch.Tensor):
                total += bias.numel() * bias.element_size()
    return int(total) if encontrados else None


def analizar_ablacion_capas(modelo: nn.Module, loader: DataLoader, device: torch.device,
                             accuracy_base: float, f1_base: float) -> list[dict[str, Any]]:
    """Pone a cero los pesos de una capa a la vez y mide el impacto en validación."""
    filas = []
    for nombre, modulo in modelo.named_modules():
        if not isinstance(modulo, (nn.Conv2d, nn.Linear)):
            continue
        copia = copy.deepcopy(modelo).to(device).eval()
        modulo_copia = dict(copia.named_modules())[nombre]
        peso = modulo_copia.weight.detach()
        with torch.no_grad():
            modulo_copia.weight.zero_()
        metricas = evaluar(copia, loader, device)
        filas.append({
            "capa": nombre,
            "tipo": type(modulo).__name__,
            "pesos_capa": int(peso.numel()),
            "magnitud_media_abs_original": float(modulo.weight.detach().abs().mean().item()),
            "accuracy_sin_pesos_capa": metricas["accuracy"],
            "f1_macro_sin_pesos_capa": metricas["f1_macro"],
            "caida_accuracy_vs_base": float(accuracy_base - metricas["accuracy"]),
            "caida_f1_vs_base": float(f1_base - metricas["f1_macro"]),
        })
        del copia
    return sorted(filas, key=lambda r: r["caida_f1_vs_base"], reverse=True)


def reporte_sparsity_capas(modelo: nn.Module) -> list[dict[str, Any]]:
    filas = []
    for nombre, modulo in modelo.named_modules():
        if isinstance(modulo, (nn.Conv2d, nn.Linear)):
            w = modulo.weight.detach()
            total = int(w.numel())
            no_cero = int(torch.count_nonzero(w).item())
            filas.append({
                "capa": nombre,
                "tipo": type(modulo).__name__,
                "pesos_totales": total,
                "pesos_no_cero": no_cero,
                "pesos_cero": total - no_cero,
                "sparsity_pct": 100.0 * (total - no_cero) / total if total else 0.0,
                "magnitud_media_abs": float(w.abs().mean().item()),
                "magnitud_mediana_abs": float(w.abs().median().item()),
            })
    return filas


def main():
    ap = argparse.ArgumentParser(description="Optimización y cuantización de la CNN entrenada en el paso 4.")
    ap.add_argument("--modelo", type=Path, default=DIR_RES / "cnn" / "modelo_mejor.pt")
    ap.add_argument("--salida", type=Path, default=DIR_RES / "paso7")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--podas", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                    help="Fracciones globales de pesos a poner en cero (0 a 1).")
    ap.add_argument("--caida-accuracy", type=float, default=0.01,
                    help="Caída máxima permitida de accuracy frente a FP32, en escala 0-1.")
    ap.add_argument("--caida-f1", type=float, default=0.01,
                    help="Caída máxima permitida de F1 macro frente a FP32, en escala 0-1.")
    ap.add_argument("--sin-int8", action="store_true", help="Omite el intento de cuantización INT8.")
    ap.add_argument("--calibracion-max-batches", type=int, default=100)
    ap.add_argument("--repeticiones-latencia", type=int, default=100)
    args = ap.parse_args()

    if any(v < 0 or v >= 1 for v in args.podas):
        raise SystemExit("Cada fracción de poda debe estar en [0, 1).")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Se solicitó CUDA, pero no hay GPU CUDA disponible.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    base, cfg = checkpoint_modelo(args.modelo, device)
    # La comparación FP32/poda/INT8 se hace toda en CPU para no mezclar latencias
    # de GPU con CPU (INT8 eager se ejecuta en CPU en este flujo).
    base = base.cpu().eval()
    device = torch.device("cpu")
    val_ds = DatasetImagenes("val", cfg["img_size"], aumentar=False)
    train_ds = DatasetImagenes("train", cfg["img_size"], aumentar=False)
    if not len(val_ds):
        raise SystemExit("dataset_proc/val está vacío. Ejecuta los pasos 1–4 antes del paso 7.")
    if not len(train_ds):
        raise SystemExit("dataset_proc/train está vacío; se necesita para calibrar INT8.")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    args.salida.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo de evaluación FP32: {device}")
    print(f"Validación: {len(val_ds)} imágenes | Calibración: {len(train_ds)} imágenes")
    base_eval = evaluar(base, val_loader, device)
    base_time = medir_latencia(base, val_ds, device, repeticiones=args.repeticiones_latencia)
    # Ablación: apaga la matriz de pesos de una capa a la vez; cuantifica cuánto cae
    # la validación para localizar bloques/capas importantes.
    ablaciones = analizar_ablacion_capas(
        base, val_loader, device, base_eval["accuracy"], base_eval["f1_macro"]
    )
    if ablaciones:
        with open(args.salida / "ablacion_por_capa.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(ablaciones[0]))
            w.writeheader()
            w.writerows(ablaciones)
    base_total, base_nz, base_sp = contar_pesos(base)
    filas: list[dict[str, Any]] = []
    guardar_checkpoint(base.cpu(), cfg, args.salida / "modelo_fp32.pt")
    base_row = {
        "modelo": "FP32_base", "poda_solicitada": 0.0,
        **base_eval, **base_time,
        "pesos_totales_conv_linear": base_total, "pesos_no_cero": base_nz,
        "sparsity_real": base_sp, "tamano_archivo_bytes": tamano_checkpoint(base, cfg),
        "ram_pesos_fp32_estimados_bytes": sum(p.numel() * p.element_size() for p in base.parameters()),
        "backend": "float32",
    }
    filas.append(base_row)

    candidatos: list[tuple[float, nn.Module, dict]] = [(0.0, copy.deepcopy(base).cpu().eval(), base_eval)]
    device_prune = torch.device("cpu")
    for fraction in sorted(set(args.podas)):
        model_p = aplicar_poda(base, fraction)
        val_p = evaluar(model_p, val_loader, device_prune)
        time_p = medir_latencia(model_p, val_ds, device_prune, repeticiones=args.repeticiones_latencia)
        total, nonzero, sparsity = contar_pesos(model_p)
        row = {
            "modelo": f"pruned_{int(fraction * 100)}pct", "poda_solicitada": fraction,
            **val_p, **time_p,
            "pesos_totales_conv_linear": total, "pesos_no_cero": nonzero,
            "sparsity_real": sparsity, "tamano_archivo_bytes": tamano_checkpoint(model_p, cfg),
            "ram_pesos_fp32_estimados_bytes": sum(p.numel() * p.element_size() for p in model_p.parameters()),
            "backend": "float32_pruned_dense",
        }
        filas.append(row)
        candidatos.append((fraction, model_p, val_p))
        print(f"Poda {fraction:.0%}: acc={val_p['accuracy']:.4f}, F1={val_p['f1_macro']:.4f}, "
              f"latencia={time_p['latencia_media_ms']:.3f} ms, sparsity={sparsity:.1%}")

    # Se elige la poda más agresiva que respeta los límites de calidad en validación.
    candidatos_factibles = [
        c for c in candidatos
        if base_eval["accuracy"] - c[2]["accuracy"] <= args.caida_accuracy
        and base_eval["f1_macro"] - c[2]["f1_macro"] <= args.caida_f1
    ]
    fraccion_sel, modelo_sel, eval_sel = max(candidatos_factibles, key=lambda x: x[0])
    modelo_sel.eval().cpu()
    poda_path = args.salida / "modelo_podado.pt"
    guardar_checkpoint(modelo_sel, cfg, poda_path)
    sparsity_capas = reporte_sparsity_capas(modelo_sel)
    if sparsity_capas:
        with open(args.salida / "sparsity_por_capa.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(sparsity_capas[0]))
            w.writeheader()
            w.writerows(sparsity_capas)
    print(f"\nModelo podado seleccionado: {fraccion_sel:.0%} de poda solicitada; "
          f"accuracy={eval_sel['accuracy']:.4f}, F1={eval_sel['f1_macro']:.4f}")

    # INT8: primero cuantización estática de Conv+Linear. Si falla por backend/versión,
    # se prueba cuantización dinámica de Linear y se documenta que NO es INT8 completo.
    int8_path = args.salida / "modelo_int8_torchscript.pt"
    int8_ok = False
    int8_info: dict[str, Any] = {"disponible": False}
    if not args.sin_int8:
        try:
            qmodel, backend = crear_modelo_int8(
                modelo_sel, train_loader, cfg, max_batches=args.calibracion_max_batches
            )
            qloader = val_loader
            q_eval = evaluar(qmodel, qloader, torch.device("cpu"))
            q_time = medir_latencia(qmodel, val_ds, torch.device("cpu"), repeticiones=args.repeticiones_latencia)
            example = val_ds[0][0].unsqueeze(0)
            file_size = exportar_torchscript(qmodel, example, int8_path)
            int8_ok = True
            int8_info = {
                "disponible": True, "tipo": "INT8_estatica_conv_bn_relu_linear",
                "backend": backend, **q_eval, **q_time,
                "tamano_archivo_bytes": int(file_size),
                "ram_pesos_estimados_bytes": bytes_pesos_cuantizados(qmodel),
            }
            filas.append({
                "modelo": "INT8_estatico", "poda_solicitada": fraccion_sel,
                **q_eval, **q_time, "pesos_totales_conv_linear": None,
                "pesos_no_cero": None, "sparsity_real": None,
                "tamano_archivo_bytes": int(file_size),
                "ram_pesos_fp32_estimados_bytes": int8_info.get("ram_pesos_estimados_bytes"),
                "backend": backend,
            })
            print(f"INT8 estático ({backend}): acc={q_eval['accuracy']:.4f}, F1={q_eval['f1_macro']:.4f}, "
                  f"latencia={q_time['latencia_media_ms']:.3f} ms, archivo={file_size / 1024:.1f} KiB")
        except Exception as exc_static:
            print(f"[AVISO] Falló INT8 estático: {exc_static}")
            try:
                dynamic = torch.ao.quantization.quantize_dynamic(
                    copy.deepcopy(modelo_sel).cpu().eval(), {nn.Linear}, dtype=torch.qint8
                ).eval()
                dyn_eval = evaluar(dynamic, val_loader, torch.device("cpu"))
                dyn_time = medir_latencia(dynamic, val_ds, torch.device("cpu"), repeticiones=args.repeticiones_latencia)
                file_size = exportar_torchscript(dynamic, val_ds[0][0].unsqueeze(0), int8_path)
                int8_ok = True
                int8_info = {
                    "disponible": True, "tipo": "INT8_dinamica_solo_linear",
                    "advertencia": "Solo se cuantizó la capa Linear; las convoluciones permanecen FP32.",
                    "error_int8_estatico": str(exc_static), **dyn_eval, **dyn_time,
                    "tamano_archivo_bytes": int(file_size),
                    "ram_pesos_estimados_bytes": bytes_pesos_cuantizados(dynamic),
                }
                filas.append({
                    "modelo": "INT8_dinamico_solo_linear", "poda_solicitada": fraccion_sel,
                    **dyn_eval, **dyn_time, "pesos_totales_conv_linear": None,
                    "pesos_no_cero": None, "sparsity_real": None,
                    "tamano_archivo_bytes": int(file_size),
                    "ram_pesos_fp32_estimados_bytes": int8_info.get("ram_pesos_estimados_bytes"),
                    "backend": "dynamic_linear_only",
                })
                print(f"INT8 dinámico solo Linear: acc={dyn_eval['accuracy']:.4f}, F1={dyn_eval['f1_macro']:.4f}, "
                      f"latencia={dyn_time['latencia_media_ms']:.3f} ms")
            except Exception as exc_dynamic:
                int8_info = {
                    "disponible": False,
                    "error_estatico": str(exc_static),
                    "error_dinamico": str(exc_dynamic),
                }
                print(f"[AVISO] También falló INT8 dinámico: {exc_dynamic}")

    # Recomendación: modelos que cumplen tolerancia de accuracy y F1; entre ellos,
    # escoger la menor latencia batch=1 medida en este equipo.
    recomendables = [base_row]
    for row in filas:
        if row["modelo"] in ("FP32_base", "INT8_estatico", "INT8_dinamico_solo_linear"):
            continue
        if (base_eval["accuracy"] - row["accuracy"] <= args.caida_accuracy and
                base_eval["f1_macro"] - row["f1_macro"] <= args.caida_f1):
            recomendables.append(row)
    if int8_ok:
        ir = filas[-1]
        if (base_eval["accuracy"] - ir["accuracy"] <= args.caida_accuracy and
                base_eval["f1_macro"] - ir["f1_macro"] <= args.caida_f1):
            recomendables.append(ir)
    recomendado = min(
        recomendables,
        key=lambda r: float(r["latencia_media_ms"]) if r.get("latencia_media_ms") is not None else float("inf"),
    )
    tipo_recomendado = "int8" if recomendado["modelo"].startswith("INT8") else (
        "podado" if recomendado["modelo"] != "FP32_base" else "fp32"
    )

    # Tabla CSV sin objetos anidados.
    columnas = list(dict.fromkeys(k for fila in filas for k in fila))
    with open(args.salida / "comparacion_modelos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    resumen = {
        "checkpoint_origen": str(args.modelo), "device_fp32": str(device),
        "cfg_entrenamiento": cfg, "clases": CLASES,
        "criterio_seleccion": {
            "split": "val", "caida_accuracy_maxima": args.caida_accuracy,
            "caida_f1_macro_maxima": args.caida_f1,
            "criterio_final": "menor latencia batch=1 entre modelos que cumplen la tolerancia de calidad",
        },
        "baseline_fp32": base_eval | base_time,
        "fraccion_poda_seleccionada": fraccion_sel,
        "modelo_recomendado": tipo_recomendado,
        "nombre_candidato_recomendado": recomendado["modelo"],
        "int8": int8_info,
        "advertencias": [
            "La poda no estructurada genera pesos cero; no reduce por sí sola la dimensión de los tensores.",
            "La latencia se mide en este equipo con batch=1; puede cambiar en el dispositivo final.",
            "El tamaño del archivo representa el almacenamiento del modelo en el disco de la computadora.",
            "RAM y CPU dependen del equipo donde se ejecuta PyTorch; el consumo energético requiere un medidor de potencia externo.",
        ],
    }
    with open(args.salida / "seleccion_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # Curva de compromiso accuracy-latencia.
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in filas:
        lat = row.get("latencia_media_ms")
        if lat is not None:
            ax.scatter(float(lat), float(row["accuracy"]), label=row["modelo"])
            ax.annotate(row["modelo"], (float(lat), float(row["accuracy"])), fontsize=8)
    ax.axhline(base_eval["accuracy"] - args.caida_accuracy, linestyle="--", linewidth=1,
               label="accuracy mínima aceptable")
    ax.set_xlabel("Latencia de inferencia batch=1 (ms)")
    ax.set_ylabel("Accuracy de validación")
    ax.set_title("Compromiso entre precisión y latencia")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.salida / "tradeoff_accuracy_latencia.png", dpi=150)
    plt.close(fig)

    print(f"\nModelo recomendado: {tipo_recomendado} ({recomendado['modelo']})")
    print(f"Accuracy={recomendado['accuracy']:.4f} | F1={recomendado['f1_macro']:.4f} | "
          f"latencia={recomendado.get('latencia_media_ms')} ms")
    print(f"Archivos guardados en: {args.salida.resolve()}")
    if int8_ok:
        print(f"TorchScript INT8: {int8_path}")
    print("No uses las métricas de validación como resultado final; reserva test y A/B/C para el reporte.")


if __name__ == "__main__":
    main()
