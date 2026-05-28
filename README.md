# Predicción de producción fotovoltaica multi-planta mediante aprendizaje automático

Sistema de predicción de producción fotovoltaica basado en aprendizaje automático, diseñado para evaluar de forma homogénea distintos modelos en un contexto operativo real.

Este trabajo desarrolla una serie de experimentos, buscando obtener conclusiones más consistentes y aplicables a entornos reales.

El desarrollo se estructura a partir de un flujo completo de tratamiento del dato: limpieza, tratamiento de valores faltantes, detección de anomalías e ingeniería de características. 

A partir del dataset preprocesado, que combina información meteorológica, variables temporales y términos autorregresivos, se estudia cómo la memoria temporal del sistema afecta al rendimiento predictivo. 

El análisis no se limita a la evaluación global de métricas, sino que incorpora un estudio segmentado por condición operativa (hora del día, nivel de irradiancia, estacionalidad) y un análisis de residuos orientado a la monitorización operativa de la instalación.

---

## Estructura del repositorio

```
TFM-Comparativa-FV/
│
├── notebooks/
│   ├── 00_preparacion_y_limpieza.ipynb           ← pipeline de limpieza (ejemplo con LECA1)
│   ├── 01_cargar_y_unificar.ipynb                ← unificación de plantas
│   ├── 02_feature_engineering.ipynb              ← construcción de variables
│   ├── 03_model_training_base.ipynb              ← entrenamiento base comparativo
│   ├── 04_analysis_and_results.ipynb             ← análisis de resultados
│   ├── 05_Experimento1_lag_vs_sinlag.ipynb       ← hipótesis: memoria temporal
│   ├── 06_Experimento2_Segmentacion.ipynb        ← hipótesis: error no homogéneo
│   ├── 07_Experimento3_adaptacion_LECA1.ipynb    ← hipótesis: adaptación local
│   ├── 08_Experimento4_Estacionalidad.ipynb      ← hipótesis: robustez estacional
│   ├── 09_Experimento5_Horizontes.ipynb          ← hipótesis: horizonte temporal
│   └── 10_Analisis_Residuos_Monitorizacion.ipynb ← análisis de residuos
│
├── src/                          ← módulos reutilizables del pipeline
│   ├── __init__.py
│   ├── data.py                   ← carga, validación y partición de datos
│   ├── features.py               ← pipeline base de ingeniería de variables
│   ├── features_horas.py         ← extensión con targets multi-horizonte (Exp. 5)
│   ├── models.py                 ← definición homogénea de los siete modelos
│   └── evaluation.py             ← métricas globales y segmentadas por horas de luz
│
├── utils/                        ← scripts de preparación de datos por planta
│   ├── Preparacion_Dataset_15min_LECA1.py    ← carga y agregación a 15 min
│   ├── Analisis_datos_LECA1.py               ← limpieza, outliers, huecos
│   ├── fill_radiation_data_LECA1.py          ← relleno de radiación con Weather Underground
│   ├── add_wunderground_temperature.py       ← temperatura desde Weather Underground
│   ├── add_openmeteo_temperature.py          ← temperatura desde Open-Meteo
│   └── config.example.py                    ← plantilla de rutas para utils
│
├── data/
│   └── README.md                 ← estructura esperada de los datos (no se incluyen datos)
│
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/beuchi8888/TFM-Comparativa-FV.git
cd TFM-Comparativa-FV

# 2. Crea un entorno virtual (recomendado)
conda create -n tfm python=3.11
conda activate tfm

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Configura las rutas locales
cp utils/config.example.py utils/config.py
# Edita config.py con las rutas a tus datos locales
```

---

## Configuración de rutas

Los datos son privados y no se incluyen en el repositorio. Copia `config.example.py` como `config.py` y ajusta las rutas a tu entorno local:

```python
# utils/config.py
from pathlib import Path

DATA_LECA1_RAW   = Path(r"ruta/a/tus/datos/LECA1")
DATA_AFRISOL_RAW = Path(r"ruta/a/tus/datos/Afrisol")
DATA_E03_RAW     = Path(r"ruta/a/tus/datos/E03")
```

Esta configuración es necesaria únicamente para el notebook `00`. Los notebooks `01`–`10` utilizan rutas relativas a la raíz del repositorio y no requieren configuración adicional.

---

## Datos

Los datos corresponden a tres instalaciones fotovoltaicas reales operadas en España con resolución temporal de **15 minutos**. No pueden hacerse públicos por razones de confidencialidad. Consulta `data/README.md` para ver la estructura esperada de los archivos de entrada.

El diseño experimental combina dos tipos de generalización simultáneamente:

- **Temporal:** los modelos se evalúan sobre datos futuros no vistos durante el entrenamiento.
- **Espacial:** la evaluación final se realiza sobre una planta no vista durante el entrenamiento (LECA1), entrenando sobre las otras dos (Afrisol y E03).

---

## Descripción de los notebooks

Los notebooks están numerados en orden de ejecución:

| Notebook | Descripción |
|----------|-------------|
| `00` | Unificación de CSVs brutos, relleno de radiación con Weather Underground, limpieza, corrección de huecos temporales y adición de temperatura con Open-Meteo | 
| `01` | Carga y unificación del dataset multi-planta, análisis exploratorio inicial y detección de anomalías | 
| `02` | Construcción de variables derivadas (lags, rolling, interacciones físicas, codificación cíclica) y partición temporal/espacial | 
| `03` | Entrenamiento y evaluación comparativa de 7 modelos base con métricas en kW | 
| `04` | Análisis detallado del mejor modelo: scatter por mes, comparativa visual de todos los modelos sobre una semana de referencia | 
| `05` | Experimento 1: cuantificación del valor de la memoria temporal comparando modelos con lag, sin lag y baseline naive de persistencia | 
| `06` | Experimento 2: segmentación del error por hora del día, nivel de irradiancia, tipo de rampa y nivel de producción | 
| `07` | Experimento 3: impacto de incorporar datos locales de la planta objetivo al entrenamiento | 
| `08` | Experimento 4: robustez estacional del modelo global evaluado en cada estación del año |
| `09` | Experimento 5: degradación del rendimiento al aumentar el horizonte de predicción (t+1, t+4, t+16) con análisis de importancia de variables por horizonte | 
| `10` | Análisis estadístico del residuo (distribución, Q-Q, ACF, Ljung-Box) y monitor operativo de anomalías persistentes | 

> **Nota:** El notebook `00` muestra el pipeline de limpieza con la planta LECA1 como ejemplo. El mismo proceso se aplica de forma equivalente a las plantas Afrisol y E03.
---

## Modelos evaluados

| Modelo | Tipo |
|--------|------|
| Ridge | Regresión lineal regularizada (baseline lineal) |
| Random Forest | Ensemble de árboles (bagging) |
| Extra Trees | Ensemble de árboles aleatorizados |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting orientado a eficiencia (leaf-wise) |
| CatBoost | Gradient boosting robusto |
| MLP | Red neuronal multicapa con escalado previo |

Todos los modelos se evalúan con los mismos datos, las mismas features y las mismas métricas para garantizar comparaciones objetivas.

---

## Métricas de evaluación

- **MAE** - Error absoluto medio (en p.u y en kW)
- **RMSE** - Raíz del error cuadrático medio (en p.u y en kW)
- **R²** - Coeficiente de determinación
- **Bias** - Error sistemático medio (en p. u y en kW)

Todas las métricas se calculan también restringuidas a **horas de producción** (`power_pu > 0`) para evitar el sesgo optimista que introduce la noche, donde la predicción de cero es trivialmente correcta.

per-unit = p.u = (`power_pu = power_KW / nominal_kW`)

---

## Experimentos

| Nº | Hipótesis | Notebook |
|----|-----------|----------|
| 1 | La memoria temporal (lags) aporta valor predictivo real frente a un modelo sin información histórica y frente a la persistencia naive | `05` |
| 2 | El error no es homogéneo: se concentra en condiciones operativas específicas (hora del día, nivel de irradiancia, tipo de rampa) | `06` |
| 3 | Incorporar datos locales de la planta objetivo mejora el rendimiento del modelo global | `07` |
| 4 | El modelo global es robusto frente a la estacionalidad, con comportamiento estable en todas las estaciones del año | `08` |
| 5 | El rendimiento degrada al aumentar el horizonte de predicción y la importancia de los lags disminuye frente a las variables físicas | `09` |
| 6 | El análisis del residuo permite detectar anomalías operativas persistentes | `10` |

---

## Dependencias principales

| Librería | Uso |
|----------|-----|
| `scikit-learn` | Ridge, RF, ExtraTrees, MLP, métricas |
| `xgboost` | Modelo XGBoost |
| `lightgbm` | Modelo LightGBM |
| `catboost` | Modelo CatBoost |
| `pandas` / `numpy` | Manipulación de datos |
| `matplotlib` / `plotly` | Visualización estática e interactiva |
| `statsmodels` | ACF, test de Ljung-Box, LOWESS |
| `scipy` | Tests de normalidad, correlación |
| `selenium` | Descarga de datos de Weather Underground |
| `astral` | Cálculo de amanecer y anochecer por día |
| `requests` | API de Open-Meteo |

---

## Notas metodológicas

- La evaluación se realiza globalmente y en **horas de producción** (`power_pu > 0`) para evitar el sesgo optimista que introduce la noche.
- Los lags y rolling windows se calculan **por planta** para evitar mezclar el final de una serie con el inicio de otra.
- El rolling usa `shift(1)` antes de la ventana para **evitar fuga de información** del presente al pasado.
- Los hiperparámetros priorizan **comparabilidad entre modelos** frente a optimización individual exhaustiva.
- La potencia se normaliza en **per-unit** (`power_pu = power_kW / nominal_kW`) para hacer comparables las tres plantas con distinta capacidad instalada.
