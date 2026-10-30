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

from config import *  # [IMPORT_LOCAL]
from utils_comunes import *  # [IMPORT_LOCAL]
from paso2_preprocesamiento import crear_augmentacion  # [IMPORT_LOCAL]

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


if __name__ == "__main__":  # [QUITAR]
    ap = argparse.ArgumentParser()  # [QUITAR]
    ap.add_argument("--estudio", action="store_true")  # [QUITAR]
    main_estudio() if ap.parse_args().estudio else main_entrenar()  # [QUITAR]
