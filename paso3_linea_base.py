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

from config import *  # [IMPORT_LOCAL]
from utils_comunes import *  # [IMPORT_LOCAL]

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


if __name__ == "__main__":  # [QUITAR]
    main_linea_base()  # [QUITAR]
