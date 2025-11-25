# TP Grupal – Telco Churn  
Proyecto MLOps de Predicción de Churn

## Proyecto ISTEA | Materia: Laboratorio de Minería de Datos

## 👥 Integrantes

- **Nadia Soledad Casá**
- **Karla Silva**

-----------------------------------

## 📋 Descripción del Proyecto

Pipeline reproducible de Machine Learning para predecir la rotación de clientes (churn) en una empresa de telefonía, aplicando buenas prácticas de MLOps con versionado de datos, tracking de experimentos y orquestación de assets.

**Contexto:**  
El objetivo es identificar qué clientes tienen mayor probabilidad de darse de baja, utilizando información de facturación, tipo de contrato y otros datos relacionados con el servicio. El proyecto no se queda en “un modelo suelto”, sino que integra:

- Código versionado en **Git/GitHub**.
- Datos y pipeline versionados con **DVC**.
- Experimentos y modelos registrados en **MLflow** (remoto en **DagsHub**).
- Visualización de métricas, gráficos y modelo campeón con **Dagster**.

-----------------------------------

## 🎯 Objetivos

- Construir un pipeline de ML completamente reproducible.
- Aplicar control de versiones con DVC y Git.
- Trackear experimentos y modelos con MLflow (DagsHub).
- Orquestar y visualizar resultados con Dagster.
- Implementar CI/CD con GitHub Actions para validar el pipeline.
- Seleccionar de forma sistemática un **modelo campeón** según `test_f1`.

-----------------------------------

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+** – Lenguaje principal
- **DVC** – Versionado de datos y modelos, definición de pipeline
- **Git/GitHub** – Control de versiones de código
- **DagsHub** – Remoto DVC + servidor MLflow
- **MLflow** – Tracking de experimentos y registro de modelos
- **Dagster** – Orquestación y visualización de assets
- **scikit-learn** – Modelado (Logistic Regression, etc.)
- **Pandas / NumPy** – Manipulación de datos

-----------------------------------

## 📊 Dataset

- **Nombre:** `telco_churn.csv`
- **Ubicación:** `data/raw/telco_churn.csv` (trackeado por DVC)
- **Target:** `churn` (1 = se da de baja, 0 = permanece)
- **Contenido:** información demográfica, de facturación y del tipo de contrato del cliente.

Ejemplos de variables:

- `customer_id`: identificador único
- `tenure` / `tenure_months`: tiempo como cliente
- `monthly_charges`: cargos mensuales
- `total_charges`: cargos acumulados
- `contract` / `contract_type`: tipo de contrato
- `churn`: variable objetivo

-----------------------------------

## ⚙️ Requisitos Previos

Antes de comenzar, es necesario contar con:

- Python 3.10+
- Conda / Anaconda
- Git
- DVC
- Cuenta en [DagsHub](https://dagshub.com/)
- Acceso al repositorio:

  - GitHub: `https://github.com/nanucasa/TP_grupal`
  - DagsHub: `https://dagshub.com/nanucasa/TP_grupal`

-----------------------------------

## 🚀 Instalación y Configuración

1. Clonar el repositorio
git clone https://github.com/nanucasa/TP_grupal.git
cd TP_grupal

. Crear entorno virtual con Conda
# Crear entorno
conda create -n tp_grupal python=3.10 -y

# Activar entorno
conda activate tp_grupal

3. Instalar dependencias
pip install -r requirements.txt

4. Configurar credenciales de DagsHub para DVC
Configurar el remoto origin de DVC con usuario y token de DagsHub:
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TU_USUARIO_DAGSHUB
dvc remote modify origin --local password TU_TOKEN_DAGSHUB

5. Descargar datos versionados
dvc pull

6. Ejecutar el pipeline completo
dvc repro train_fe

Esto ejecuta las etapas necesarias hasta train_fe, actualizando datos procesados, features, modelo y métricas.

-----------------------------------

📁 Estructura del Proyecto

TP_grupal/
├── data/
│   ├── raw/                      # Datos originales (DVC)
│   │   └── telco_churn.csv
│   ├── processed/                # Datos limpios (DVC)
│   └── features/                 # Features para entrenamiento (DVC)
├── src/
│   ├── data_prep.py              # Preparación de datos (limpieza + splits)
│   └── train.py                  # Entrenamiento + logging en MLflow
├── models/                       # Modelos entrenados (DVC)
│   └── model_fe.joblib
├── scripts/
│   ├── import_dvc_metrics_to_mlflow.py
│   └── update_champion_from_runs.py   # Selección del modelo campeón
├── tp_grupal_dagster/
│   └── tp_grupal_dagster/
│       ├── assets.py             # Definición de assets de Dagster
│       └── __init__.py           # Definitions de Dagster
├── artifacts/
│   └── champion_run.json         # Información del modelo campeón (generado)
├── reports/                      # Gráficos y reportes (curvas ROC/PR, etc.)
├── .dvc/                         # Configuración de DVC
├── .github/
│   └── workflows/                # GitHub Actions CI/CD
├── params.yaml                   # Parámetros configurables del pipeline
├── dvc.yaml                      # Definición del pipeline DVC (stages)
├── dvc.lock                      # Estado del pipeline (reproducibilidad)
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo

-----------------------------------

🔄 Pipeline de Trabajo (DVC)

El proyecto implementa un pipeline reproducible con varias etapas definidas en dvc.yaml.

Etapa 1: Preparación de Datos (data_prep)

- Script: src/data_prep.py

Funciones:
	- Carga del dataset crudo desde data/raw/telco_churn.csv.
	- Limpieza de datos (valores faltantes, tipos, etc.).
	- Transformaciones iniciales.
	- División en conjuntos de train / valid / test.
	- Exporta datasets limpios a data/processed/.

Entradas:
	- data/raw/telco_churn.csv
	- params.yaml

Salidas:
	- Archivos procesados en data/processed/ (train/valid/test).

Etapa 2: Ingeniería de Features y Tuning (feature_eng y tune)

Functions principales:

- Aplicación de ingeniería de características (codificación, escalado, etc.).
- Generación de archivos de features:
	- data/features/train_fe.csv
	- data/features/valid_fe.csv
	- y equivalentes de test
- Experimentos de tuning (distintos hiperparámetros/modelos).
- Registro de runs de tuning en MLflow.

Entradas:
- Datos de data/processed/
- params.yaml

Salidas:
	- Archivos de features en data/features/.
	- Artefactos y métricas de tuning en MLflow.

Etapa 3: Entrenamiento Final (train_fe)
Script: src/train.py
- Funciones:
	- Carga de datos de features (train/valid y, cuando corresponde, test).
	- Entrenamiento de un modelo de Logistic Regression dentro de un pipeline StandardScaler + logisticRegression.

- Cálculo de métricas:
	- Accuracy, Precision, Recall, F1 (valid/test)
	- ROC-AUC, PR-AUC (test)

- Logueo de parámetros, métricas y modelo en MLflow (tracking remoto en DagsHub).
- Registro del modelo en el Model Registry de MLflow bajo el nombre TelcoChurn_LogReg.

Entradas:
	- data/features/train_fe.csv
	- data/features/valid_fe.csv
	- test features
	- params.yaml

Salidas:
	- models/model_fe.joblib
	- Métricas locales (metrics_fe.json y/o similares).
	- Runs y modelos en MLflow (experimento de entrenamiento final).

-----------------------------------

📈 Reproducibilidad

Comandos útiles para trabajar con DVC:

# Ejecutar todo el pipeline (hasta train_fe)
dvc repro train_fe

# Ver el DAG del pipeline
dvc dag

# Verificar estado del pipeline
dvc status

# Ver diferencias en parámetros
dvc params diff

🔧 Configuración de Parámetros
Los parámetros configurables del pipeline están en params.yaml.
Ahí se definen, por ejemplo:

Parámetros de división de datos:
	- test_size
	- valid_size
	- random_state

Columna objetivo:
- target_column: churn

Parámetros del modelo (Logistic Regression):
	- C
	- max_iter
	- class_weight
	- etc.

Flujo típico:
	1- Editar params.yaml.
	2- Ejecutar: dvc repro train_fe
	3- DVC re-ejecuta solo las etapas afectadas por el cambio.
	4- Las métricas se actualizan y los nuevos runs se registran en MLflow.

-----------------------------------

🧪 Experimentos Realizados

Se ejecutaron múltiples experimentos (más de 200 runs) variando:

- Ingeniería de features (baseline vs feature engineering).
- Hiperparámetros del modelo (C, max_iter, semillas).
- Distintos enfoques de entrenamiento y evaluación.

Los experimentos están organizados principalmente en los experimentos de MLflow:

	- telco_churn_baseline
	- telco_churn_baseline_fe
	- telco_churn_baseline_rf
	- telco_churn_tune_xgb

En el experimento telco_churn_tune_xgb se concentraron los runs de evaluación final (metrics_test*), que se usan para elegir el modelo campeón por test_f1.

-----------------------------------

🏆 Modelo Seleccionado (Champion)

Experimento principal: telco_churn_tune_xgb
Run campeón: metrics_test_fe
Modelo: Logistic Regression sobre features ingenieradas (TelcoChurn_LogReg)

Métricas (aprox. conjunto de test):

| Métrica  | Valor aproximado   |
| -------- | ------------------ |
| F1-Score | ≈ 0.60 (`test_f1`) |
| ROC-AUC  | ≈ 0.74             |
| PR-AUC   | ≈ 0.60             |

La selección del champion se realiza con:

- scripts/update_champion_from_runs.py

Este script:

	1- Se conecta al experimento telco_churn_tune_xgb en MLflow.
	2- Filtra los runs cuyo nombre comienza con metrics_test.
	3- Ordena por metrics.test_f1 de mayor a menor.
	4- Elige el mejor run como modelo campeón.
	5- Guarda la información en: artifacts/champion_run.json

Ese JSON contiene:

	- experiment_name
	- experiment_id
	- primary_metric (test_f1)
	- run_id
	- run_name (ej: metrics_test_fe)
	- metric_value (valor de test_f1 del champion)

-----------------------------------

📊 Dagster: Métricas, Gráficos y Champion

El proyecto de Dagster vive en: tp_grupal_dagster/tp_grupal_dagster/

1- **Los assets principales son: test_metrics**

- Se conecta a MLflow remoto.
- Para cada experimento (baseline, baseline_fe, baseline_rf), obtiene el mejor run de test por f1_test.
- Devuelve un DataFrame con métricas de test y genera reports/dagster_mlflow_test_metrics.csv.

2- **f1_barchart**

- Usa test_metrics para graficar un barplot de f1_test por modelo.
- Guarda la imagen en reports/f1_bench_dagster.png.

3- **pr_curve_fe**

- Registra la curva Precision–Recall del modelo con features.
- Lee reports/pr_curve_fe.png.
- Asocia la métrica PR-AUC (desde MLflow o valor fijo configurado).

4- **roc_curve_fe**

- Registra la curva ROC del modelo con features.
- Lee reports/roc_curve_fe.png.
- Asocia la métrica ROC-AUC (desde MLflow o valor fijo configurado).

5- **champion_run**

- Lee artifacts/champion_run.json.
- Expone en Dagster quién es el modelo campeón y su test_f1.

-----------------------------------

🐙 **Cómo levantar Dagster**

	cd tp_grupal_dagster
	conda activate tp_grupal
	dagster dev

Luego abrir:
	http://127.0.0.1:3000

En la pestaña Assets se pueden materializar y visualizar:

- test_metrics
- f1_barchart
- pr_curve_fe
- roc_curve_fe
- champion_run

-----------------------------------

🔗 Enlaces del Proyecto

**Repositorio GitHub:**
	https://github.com/nanucasa/TP_grupal

**Proyecto DagsHub (DVC + MLflow):**
	https://dagshub.com/nanucasa/TP_grupal

**UI MLflow (experimentos y modelos):**
	https://dagshub.com/nanucasa/TP_grupal.mlflow

-----------------------------------

🚦 CI/CD con GitHub Actions

El proyecto incluye automatización con GitHub Actions en .github/workflows/, que:

- Instala dependencias (requirements.txt).
- Ejecuta dvc pull para traer datos y modelos desde el remoto.
- Ejecuta dvc repro para validar el pipeline.
- Utiliza secrets configurados en GitHub para conectarse a DagsHub.

Esto asegura que el pipeline sea reproducible en un entorno limpio ante cada push o pull request.

-----------------------------------

🐛 Resolución de Problemas

- Error: dvc pull falla

Verificar configuración del remote: dvc remote list

Reconfigurar credenciales (usuario/token DagsHub):
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TU_USUARIO
dvc remote modify origin --local password TU_TOKEN

- Error: dvc repro no detecta cambios

Forzar re-ejecución de una etapa específica: dvc repro -f data_prep
O de todo el pipeline hasta train_fe: dvc repro -f train_fe

- Error: falta algún archivo
dvc pull
dvc status

-----------------------------------

📌 Resultados Finales
Modelo en “Producción” (Champion interno)

Algoritmo seleccionado: Logistic Regression sobre features ingenieradas
Run: metrics_test_fe (experimento telco_churn_tune_xgb)
Métrica principal: test_f1 ≈ 0.60

El modelo logra un compromiso razonable entre:
	buena F1 en test,
	buen ROC-AUC (~0.74) y PR-AUC (~0.60),
	y una implementación simple y fácilmente desplegable.

Visualizaciones

El pipeline genera:
	1- Curva ROC y curva Precision–Recall del modelo FE en reports/.
	2- Gráfico de barras de F1 por modelo (f1_bench_dagster.png) vía Dagster.
	3- DataFrame consolidado de métricas de test (dagster_mlflow_test_metrics.csv).

-----------------------------------


🚀 Deployment

La estrategia de deployment propuesta (API REST, batch, monitoreo y reentrenamiento) se documenta en:

DEPLOYMENT.md

-----------------------------------

👤 Autoras

Nadia Soledad Casá
Karla Silva

Curso: Laboratorio de Minería de Datos – ISTEA
Año: 2025
