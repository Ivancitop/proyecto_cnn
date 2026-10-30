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

from config import *  # [IMPORT_LOCAL]
from utils_comunes import *  # [IMPORT_LOCAL]
from paso3_linea_base import extraer_caracteristicas, predecir_baseline  # [IMPORT_LOCAL]
from paso4_entrenar_cnn import *  # [IMPORT_LOCAL]


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


if __name__ == "__main__":  # [QUITAR]
    main_evaluar()  # [QUITAR]
