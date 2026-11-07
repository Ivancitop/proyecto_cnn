# Inspección visual con CNN – cubo impreso en 3D

## Preparación
    pip install -r requirements.txt
Crea la carpeta `dataset_original/` junto a los scripts y copia ahí tus imágenes
(`normal_1.jpg`, `rayadura_1.jpg`, `defectuosa_1.jpg`, ...).
Opcional: `dataset_test_externo/` con fotos en otras condiciones (luz, altura, orientación) solo para prueba.

## Ejecución por pasos
    python paso1_dividir_dataset.py            # train / val / test  (añade --modo secuencial si las fotos son en ráfaga)
    python paso2_preprocesamiento.py           # revisa resultados/preprocesamiento/ejemplos_roi.png
    python paso3_linea_base.py                 # método convencional
    python paso4_entrenar_cnn.py               # entrena la CNN y evalúa en test
    python paso4_entrenar_cnn.py --estudio     # estudio de hiperparámetros (usa solo validación; es lento)
    python paso5_evaluar_errores.py            # tipos de fallo, Grad-CAM, robustez, CNN vs convencional

## Ejecución de una sola vez
    python pipeline_completo.py                # pasos 1 a 5
    python pipeline_completo.py --estudio      # + estudio de hiperparámetros
    python pipeline_completo.py --desde 4      # reanudar
Si modificas algún archivo individual, regenera el archivo único con `python construir_pipeline_completo.py`.

## Paso 6 — Domain shift (punto 9)

Crea esta estructura y coloca imágenes nuevas, capturadas en condiciones físicas distintas:

```text
pruebas/
├── condicionA/
│   ├── normal_01.jpg
│   ├── rayadura_01.jpg
│   └── defectuosa_01.jpg
├── condicionB/
│   ├── normal_01.jpg
│   ├── rayadura_01.jpg
│   └── defectuosa_01.jpg
└── condicionC/
    ├── normal_01.jpg
    ├── rayadura_01.jpg
    └── defectuosa_01.jpg
```

Las imágenes se etiquetan por el **prefijo de su nombre**, igual que en el dataset original. No uses imágenes del entrenamiento para este experimento.

```bash
python paso6_domain_shift.py
```

Opciones:

```bash
python paso6_domain_shift.py --pruebas pruebas --modelo resultados/cnn/modelo_mejor.pt
python paso6_domain_shift.py --device cpu
```

Salidas en `resultados/paso6_domain_shift/`:

- `metricas_por_condicion.json`: accuracy, precision, recall, F1, FPR/FNR one-vs-rest y de inspección binaria.
- `comparacion_condiciones.csv`: métricas y degradación frente a condición A.
- `predicciones_por_imagen.csv`: predicción y probabilidad de cada clase por imagen.
- `matriz_confusion_condicion*.png` y `comparacion_accuracy_f1.png`.

La degradación de accuracy se reporta en puntos porcentuales y de forma relativa. Un valor negativo significa que B o C rindió mejor que A. En la evaluación binaria, una pieza `normal` es positiva para aceptación; para las métricas de detección de defectos se define defecto como clase positiva: FP = pieza buena rechazada y FN = pieza defectuosa aceptada.

## Paso 7 — Optimización (punto 10)

```bash
python paso7_optimizar_cnn.py
```

El script compara FP32 contra distintos porcentajes de poda global por magnitud. Selecciona la poda más alta que mantenga accuracy y F1 macro dentro de las tolerancias configuradas. Después intenta cuantización estática INT8 (con calibración en `dataset_proc/train`); si el backend o la versión de PyTorch no lo permite, intenta INT8 dinámico en la capa `Linear` y marca claramente esa limitación.

Opciones útiles:

```bash
python paso7_optimizar_cnn.py --podas 0.1 0.2 0.3 0.4 --caida-accuracy 0.01 --caida-f1 0.01
python paso7_optimizar_cnn.py --device cpu --sin-int8
```

Salidas en `resultados/paso7/`:

- `modelo_fp32.pt`: checkpoint FP32 de referencia.
- `modelo_podado.pt`: checkpoint con los pesos podados (mismas dimensiones de la red).
- `modelo_int8_torchscript.pt`: exportación TorchScript cuantizada, si la conversión se pudo completar.
- `comparacion_modelos.csv`, `seleccion_modelo.json` y `tradeoff_accuracy_latencia.png`.
- `ablacion_por_capa.csv`: apaga los pesos de cada capa por separado y registra cuánto caen accuracy/F1 en validación.
- `sparsity_por_capa.csv`: porcentaje de pesos que quedaron en cero por convolución o capa lineal.

**Importante:** la poda no estructurada solo fuerza pesos a cero. En la red densa estándar no reduce automáticamente la RAM, el tamaño del tensor ni el tiempo de inferencia; sí permite estudiar qué proporción de pesos puede eliminarse sin degradar demasiado las métricas. La cuantización INT8 sí puede disminuir el almacenamiento de pesos y acelerar la inferencia cuando el backend dispone de kernels INT8. La latencia reportada corresponde a batch 1 en la computadora donde se ejecuta el experimento.

La recomendación se elige usando validación, no el conjunto test. Conserva test y las condiciones A/B/C para la evaluación final independiente.

## Paso 8 — Cámara y banda transportadora (puntos 10–14)

### Cámara fija / pieza en mesa

```bash
python paso8_inspeccion_tiempo_real.py --modo camara --camara 0
```

### Banda transportadora

También puedes pasar un archivo de video como fuente para repetir una prueba sin cámara física. El archivo termina al llegar al final; `--max-frames` permite limitar la prueba.

```bash
python paso8_inspeccion_tiempo_real.py --modo banda --camara video_banda.mp4 --roi 200,100,300,300 --headless
```

Delimita una zona fija de inspección dentro del frame. Ejemplo con ROI `x=200, y=100, ancho=300, alto=300`:

```bash
python paso8_inspeccion_tiempo_real.py --modo banda --camara 0 --roi 200,100,300,300 \
  --velocidad-banda-mm-s 80 --longitud-zona-mm 120
```

El programa detecta una pieza, espera `--frames-estables`, produce una decisión y no vuelve a clasificarla hasta que salga de la zona durante `--frames-salida` frames. En banda se asume una pieza a la vez en la ROI. Ajusta el tamaño de ROI, área mínima y estabilidad según tu montaje real.

El modelo predicho se convierte en una decisión ejecutada completamente en la computadora:

- `ACEPTAR`: clase `normal` y probabilidad normal por encima de `--umbral-normal`.
- `RECHAZAR`: clase `rayadura`, `defectuosa` o probabilidad normal por debajo del umbral configurado.

Por defecto `--umbral-normal 0` conserva la decisión por `argmax` de la CNN. Para una política más conservadora, prueba por ejemplo `--umbral-normal 0.60` y evalúa sus falsos rechazos/falsos aceptados.

### Indicador visual local

No se requiere Arduino, ESP32, microcontrolador, GPIO ni comunicación serial. Cuando se detecta una pieza, la aplicación muestra `ACEPTAR` en verde o `RECHAZAR` en rojo en la ventana de OpenCV y también imprime la decisión en la consola. Con `--headless`, la operación sigue siendo local y las decisiones se muestran en consola.

### Etiquetado para métricas reales

La predicción de la CNN no es una etiqueta real. Para calcular accuracy, precision, recall, F1, matriz de confusión, falsos positivos y falsos negativos durante el experimento, presiona:

- `n`: la pieza pendiente era normal.
- `r`: la pieza pendiente era rayadura.
- `d`: la pieza pendiente era defectuosa.
- `q`: salir.

Las teclas etiquetan la pieza pendiente más antigua. Si no se conoce la etiqueta real, no la asignes: esa pieza se incluye en conteos/tiempos, pero no en métricas supervisadas.

### Medición del tiempo de ciclo

El sistema mide captura, preprocesamiento, inferencia, salida/decisión, latencia total y ciclo desde la primera detección estable de una pieza hasta la decisión. FPS corresponde al ciclo de frames de la aplicación; la tasa de inspección cuenta piezas por segundo/minuto.

Si se proporciona velocidad de banda y longitud útil de zona:

\[
T_{disponible}=\frac{L_{zona}}{v_{banda}}\times1000\quad[ms]
\]

El script compara el tiempo de ciclo medio con ese tiempo disponible y estima una velocidad máxima de banda a partir del ciclo medido. Usa la longitud efectiva en la dirección del movimiento, no el ancho arbitrario de la ventana de cámara.

Salidas en `resultados/paso8_tiempo_real/`:

- `registro_inspeccion.csv`: una fila por pieza con probabilidades, decisión y tiempos.
- `metricas_sistema.json`: métricas de visión, inspección, tiempos, FPS, CPU/RAM y tamaño del modelo.
- `tiempos_sistema.csv` y, si se etiquetaron piezas, `matriz_confusion_tiempo_real.png`.

El tamaño del archivo del modelo representa el almacenamiento que ocupa en el disco de la computadora. La RAM y CPU reportadas corresponden al proceso de inspección en esa misma computadora. El consumo energético no se estima por software: para medirlo se necesita un medidor de potencia conectado al equipo.

## Dependencias

El repositorio ya incluye PyTorch, torchvision, OpenCV, NumPy, Matplotlib y scikit-learn. Para las salidas opcionales instala:

```bash
pip install psutil
```

`psutil` es opcional y se utiliza para registrar CPU y RAM del proceso. Sin esta biblioteca, la cámara, la inferencia y las decisiones locales siguen funcionando; únicamente esas métricas de recursos aparecerán como no disponibles.
