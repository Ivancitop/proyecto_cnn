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

## Resultados (carpeta `resultados/`)
- `cnn/`: curvas de loss/accuracy, historial, matriz de confusión, métricas, modelo
- `baseline/`: reglas del árbol, métricas, matrices
- `evaluacion/`: análisis de errores, Grad-CAM, robustez, comparación CNN vs convencional
- `estudio_hiperparametros/`: tabla y gráfica por factor
