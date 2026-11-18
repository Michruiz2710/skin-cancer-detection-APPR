# Skin Cancer Detection con DenseNet201 y Transfer Learning

Proyecto final de la materia Aprendizaje Profundo — Universidad del Valle de Guatemala.

Clasificacion automatica de lesiones dermoscopicas en 7 categorias usando DenseNet201,
Transfer Learning y un esquema de entrenamiento en 3 fases diseñado para enfrentar el
desbalanceo severo de 58:1 del dataset HAM10000.

---

## Descripcion del problema

El dataset HAM10000 contiene 10,015 imagenes de lesiones de piel divididas en 7 clases.
El principal reto es el desbalanceo extremo: la clase nv (nevos melanociticos) tiene 6,705
imagenes mientras que df (dermatofibroma) tiene solo 115, una relacion de 58:1.

Sin corregir este desbalanceo, cualquier modelo aprende a predecir siempre nv y obtiene
un accuracy inflado que no refleja capacidad real de diagnostico.

| Clase  | Nombre completo                                       | Imagenes | % del total |
|--------|-------------------------------------------------------|----------|-------------|
| nv     | Nevos melanociticos                                   | 6,705    | 67%         |
| mel    | Melanoma                                              | 1,113    | 11%         |
| bkl    | Lesiones queratosicas benignas                        | 1,099    | 11%         |
| bcc    | Carcinoma basocelular                                 | 514      | 5%          |
| akiec  | Queratosis actinicas / Carcinoma intraepitelial       | 327      | 3%          |
| vasc   | Lesiones vasculares                                   | 142      | 1.4%        |
| df     | Dermatofibroma                                        | 115      | 1.1%        |

El dataset se divide en train (~8,000 imagenes), validacion (~500) y test (~500).

---

## Referencia: paper EnsembleSkinNet (2025)

Este proyecto toma como referencia el paper *EnsembleSkinNet: A Transfer Learning-Based
Framework for Efficient Skin Cancer Detection with Explainable AI Integration* (2025).

El paper propone un ensemble de 4 modelos (VGG16, ResNet50, InceptionV3, DenseNet201)
con weighted averaging, augmentacion sintetica via SMOTE y busqueda bayesiana de
hiperparametros. Sus resultados sobre HAM10000:

| Metrica    | DenseNet201 solo | EnsembleSkinNet |
|------------|-----------------|-----------------|
| Accuracy   | 97.05%          | 98.32%          |
| F1-macro   | 98.15%          | 98.15%          |

Nuestro proyecto implementa unicamente DenseNet201 (uno de los 4 modelos del ensemble),
sin augmentacion sintetica ni busqueda automatica de hiperparametros, con el objetivo de
documentar el proceso iterativo de mejora desde cero.

---

## Arquitectura del modelo

```
DenseNet201 (pesos ImageNet, fine-tuning gradual)
     |
GlobalAveragePooling2D
     |
Dense(512, activacion=ReLU)
     |
Dropout(0.5)
     |
Dense(7, activacion=Softmax)
```

**DenseNet201** produce 1,920 caracteristicas en su capa final. La capa Dense(512) aprende
combinaciones no lineales de esas caracteristicas antes de clasificar, siguiendo la
arquitectura del paper EnsembleSkinNet. El Dropout(0.5) previene sobreajuste apagando
aleatoriamente el 50% de las neuronas durante el entrenamiento.

### Preprocesamiento

- **Hair removal:** morfologia blackhat para detectar pelos + cv2.INPAINT_TELEA para
  reconstruir la zona ocluida. Los pelos en imagenes dermoscopicas confunden al modelo.
- **Normalizacion:** estadisticas de ImageNet (mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]) porque DenseNet201 fue preentrenado con esos valores.
- **Tamano:** todas las imagenes redimensionadas a 224x224 pixeles.

### Funcion de perdida: FocalLossAlpha

La Focal Loss estandar agrega un factor `(1 - p_t)^gamma` que reduce el peso de los
ejemplos faciles y concentra el aprendizaje en los dificiles. Se extendio con pesos alpha
por clase:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Los pesos alpha se calculan desde el inverso de la frecuencia de cada clase, con
amplificacion adicional para mel (x3) y reduccion para nv (/5) en la version final (v9+).
Esto garantiza que todas las clases tengan alpha > 0 y que el melanoma reciba mas
atencion durante el entrenamiento.

### Esquema de entrenamiento: 3 fases

| Fase  | Base DenseNet201      | Funcion de perdida             | LR   | Epocas          |
|-------|-----------------------|--------------------------------|------|-----------------|
| 1     | Congelada             | CrossEntropy + class_weights   | 1e-3 | hasta 15 (early stop) |
| 2a    | Ultimas 50 capas      | FocalLossAlpha                 | 1e-4 | 25              |
| 2b    | Ultimas 100 capas     | FocalLossAlpha                 | 1e-5 | 35              |

La Fase 1 usa CrossEntropy porque Focal Loss bloquea la convergencia inicial cuando el
clasificador parte de pesos aleatorios (el gradiente ya viene reducido antes de que el
modelo aprenda nada). Una vez el modelo tiene capacidad predictiva, Fase 2 introduce
FocalLossAlpha para concentrarse en las clases dificiles.

---

## Estrategia de balanceo (v10)

Se aplico oversample ciclico de todas las clases minoritarias hasta TARGET_N = 1,500
imagenes, repitiendo ciclicamente las imagenes existentes con augmentacion agresiva
(rotacion 360 grados, color jitter, flips) para las clases mas escasas.

| Clase  | Imagenes originales | Post-oversample | Imagenes añadidas |
|--------|--------------------|-----------------|--------------------|
| akiec  | 327                | 1,500           | +1,173             |
| bcc    | 487                | 1,500           | +1,013             |
| bkl    | 1,048              | 1,500           | +452               |
| df     | 110                | 1,500           | +1,390             |
| mel    | 1,076              | 1,500           | +424               |
| nv     | 6,405              | 6,405           | sin cambio         |
| vasc   | 136                | 1,500           | +1,364             |

Total de imagenes de entrenamiento post-oversample: 15,405.

---

## Resultados

### Mejor modelo: v10

| Clase  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| akiec  | 0.94      | 0.79   | 0.86     | 42      |
| bcc    | 0.81      | 0.85   | 0.83     | 67      |
| bkl    | 0.52      | 0.92   | 0.66     | 142     |
| df     | 0.92      | 0.86   | 0.89     | 14      |
| mel    | 0.37      | 0.95   | 0.53     | 144     |
| nv     | 0.99      | 0.61   | 0.76     | 858     |
| vasc   | 0.95      | 1.00   | 0.97     | 18      |

**Test Accuracy: 70.89%**
**F1-macro: 78.62%**
**F1-mel: 53.20%**

### Evolucion iterativa del proyecto

| Version | Accuracy   | F1-macro | F1-mel | Cambio principal                                  |
|---------|------------|----------|--------|---------------------------------------------------|
| v1      | 54.33%     | --       | --     | Baseline sin balanceo; predecia siempre nv        |
| v2      | 34.35% val | --       | --     | Focal Loss en Fase 1 bloqueo la convergencia      |
| v3      | 66.77% val | 11.44%   | --     | CrossEntropy en Fase 1; aun colapso en nv         |
| v4      | 43.81%     | 15.29%   | --     | Bug .repeat() usaba solo 31% de datos por epoca   |
| v5      | 68.79%     | 43.62%   | 0.00%  | Pipeline corregido; mel ignorado completamente    |
| v8      | 68.64%     | 19.33%   | 47.97% | alpha[mel]=0.95 colapso las otras 5 clases a F1=0 |
| v9      | 60.10%     | 70.20%   | 43.70% | Alpha balanceado; todas las clases activas        |
| v10     | 70.89%     | 78.62%   | 53.20% | Oversample todas las clases a TARGET_N=1500       |

### Comparativa con el paper

| Metrica   | Nuestro v10 | DenseNet201 del paper | EnsembleSkinNet |
|-----------|-------------|-----------------------|-----------------|
| Accuracy  | 70.89%      | 97.05%                | 98.32%          |
| F1-macro  | 78.62%      | 98.15%                | 98.15%          |

La brecha de ~27 puntos se explica por tres diferencias estructurales:
1. El paper usa un ensemble de 4 modelos; nosotros usamos solo DenseNet201.
2. El paper genera imagenes sinteticas con SMOTE; nosotros hacemos oversample ciclico.
3. El paper optimizo hiperparametros con busqueda bayesiana; nosotros los ajustamos de
   forma razonada y documentada con una hipotesis explicita por cada version.

---

## Proceso de desarrollo: hipotesis por version

Cada version cambio exactamente una variable principal para medir su efecto de forma
aislada. Las decisiones no fueron por ensayo y error ciego sino por diagnostico de
metricas, hipotesis sobre la causa y correccion especifica.

### v5 a v8: primera hipotesis sobre el melanoma

**Diagnostico:** F1-mel = 0%. El modelo nunca predecia mel porque nv domina el
entrenamiento con 6x mas imagenes y ambas clases son visualmente similares.

**Hipotesis:** aumentar drasticamente el peso de mel en la funcion de perdida forzara
al modelo a aprender sus caracteristicas especificas.

**Resultado de v8:** alpha[mel]=0.95 produjo F1-mel=47.97% pero colapsó las otras
5 clases a F1=0% por un bug de renormalizacion que asignaba alpha=0 al resto.

**Aprendizaje:** el alpha si controla que clases aprende el modelo. El mecanismo fue
validado empiricamente aunque el experimento fallo como modelo final.

### v8 a v9: segunda hipotesis — balance del alpha

**Hipotesis:** calcular alpha desde 1/frecuencia para todas las clases y luego amplificar
mel manteniendo todas las demas en positivo deberia activar todas las clases.

**Resultado de v9:** F1-macro sube de 43.62% (v5) a 70.20%. Todas las 7 clases fueron
predichas por primera vez. El salto de 26.58 puntos porcentuales confirma la hipotesis.

### v9 a v10: tercera hipotesis — datos insuficientes para clases raras

**Hipotesis:** con solo 110 imagenes de df, el modelo no tiene suficientes ejemplos
distintos para aprender sus caracteristicas independientemente del alpha. Oversamplear
a 1,500 daria mas variacion visual.

**Resultado de v10:** F1-macro sube a 78.62%. Las mayores mejoras fueron en df (+0.19)
y akiec (+0.21), exactamente las clases con mas imagenes añadidas, confirmando la hipotesis.

---

## Notebooks

| Notebook                      | Descripcion                                              |
|-------------------------------|----------------------------------------------------------|
| 00_setup.ipynb                | Configuracion del entorno Kaggle                         |
| 01_exploracion.ipynb          | Analisis exploratorio del dataset HAM10000               |
| 02_preprocesamiento.ipynb     | Hair removal, normalizacion, conversion a .npy           |
| 02_entrenamiento_v1.ipynb     | Baseline — DenseNet201 sin balanceo de clases            |
| 03_entrenamiento_v2.ipynb     | Focal Loss en Fase 1; no convergio                       |
| 03_entrenamiento_v3.ipynb     | Fine-tuning gradual + CrossEntropy en Fase 1             |
| 03_entrenamiento_v4.ipynb     | Monitor val_macro_f1; bug .repeat() descubierto          |
| 04_entrenamiento_v5.ipynb     | Pipeline corregido + LR warmup; mel F1=0                 |
| 05_gradcam_v6.ipynb           | Grad-CAM sobre modelo v5                                 |
| 06_training_v7.ipynb          | Experimento intermedio                                   |
| 07_training_v8.ipynb          | FocalLossAlpha con alpha extremo (bug de renormalizacion)|
| 08_training_v9.ipynb          | FocalLossAlpha balanceado; todas las clases activas      |
| 09_training_v10.ipynb         | Mejor modelo — oversample + FocalLossAlpha               |

---

## Uso en Kaggle

Los notebooks estan disenados para ejecutarse en Kaggle con 2x Tesla T4 (30 GB VRAM).

### Datasets requeridos

Agregar en el panel de Kaggle antes de ejecutar los notebooks de entrenamiento v9 y v10:

- michellercarranza/ham10000-numpy — Dataset preprocesado en formato .npy
- michellercarranza/appr-models-v9-v10 — Modelos entrenados guardados

### Token de GitHub

Los notebooks que guardan modelos en GitHub usan un token personal:

```python
GITHUB_TOKEN = 'ghp_TU_TOKEN_AQUI'  # reemplazar con token real antes de ejecutar
```

No subir el token real al repositorio.

---

## Requisitos

```
tensorflow >= 2.15
keras >= 3.0
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
pillow
```

---

## Autores

Michel Ruiz — Universidad del Valle de Guatemala — 2026
