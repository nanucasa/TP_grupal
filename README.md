# TP Grupal – Telco Churn

- Proyecto MLOps de Predicción de Churn
- Proyecto ISTEA | Materia: Laboratorio de Minería de Datos

# esto es una nota para comprobar que el documento desde el branch del cliente ha sido subido satisfactoriamente por el PR

## 📋 Descripción del Proyecto

- Pipeline reproducible de Machine Learning para predecir la rotación de clientes (churn) en una empresa de telefonía, aplicando buenas prácticas de MLOps con versionado de datos, tracking de experimentos y orquestación automatizada de assets.

**Contexto:**
- El objetivo es identificar qué clientes tienen mayor probabilidad de darse de baja, utilizando información de facturación, tipo de contrato y otros datos relacionados con el servicio.

**El proyecto integra:**
- 1- Código versionado con Git/GitHub.
- 2- Datos y pipeline versionados con DVC.
- 3- Experimentos y modelos registrados en MLflow (remoto en DagsHub).
- 4- Visualización y automatización de selección de modelo campeón con Dagster.

## 🎯 Objetivos
Predecir la baja de clientes de una telco (churn) usando modelos de Machine Learning y armar un flujo reproducible de:

- Construir un pipeline de ML completamente reproducible.
- Versionado de datos y modelos en **DagsHub** (DVC + MLflow Registry)
- Preparación de datos con **DVC**  
- Trackear experimentos y modelos con MLflow (DagsHub).
- Orquestación y elección de modelo campeón con **Dagster**  
- **CI** y **CD** en **GitHub Actions**  
- Seleccionar de forma sistemática un modelo campeón según F1.

## 🛠️ Tecnologías Utilizadas

- Python 3.10+ – Lenguaje principal  
- DVC 3.63+ – Versionado de datos y modelos  
- Git/GitHub – Control de versiones  
- DagsHub – Hosting remoto (DVC + MLflow)  
- MLflow 2.22+ – Tracking y registro de modelos (remoto en DagsHub)  
- Dagster 1.12+ – Orquestación y monitoreo de assets  
- dagster-webserver / dagster-daemon – UI y demonios de orquestación  
- scikit-learn 1.5+ – Modelado  
- Pandas / NumPy – Manipulación de datos  
- Matplotlib – Gráficos de performance  

## 📊 Dataset

- Nombre: telco_churn.csv
- Ubicación: data/raw/telco_churn.csv (trackeado por DVC)
- Target: churn (1 = se da de baja, 0 = permanece)
- Contenido: información demográfica, de facturación y del tipo de contrato del cliente.
- Ejemplos de variables:
- customer_id: identificador único
- tenure_months: tiempo como cliente
- monthly_charges: cargos mensuales
- total_charges: cargos acumulados
- contract_type: tipo de contrato
- churn: variable objetivo

## ⚙️ Requisitos Previos

- Python 3.10 o superior
- Conda / Anaconda
- Git
- DVC
- Cuenta en DagsHub
- Repositorios:
- GitHub: https://github.com/nanucasa/TP_grupal
- DagsHub: https://dagshub.com/nanucasa/TP_grupal

## 🚀 Instalación y Configuración (guía paso a paso)

**Clonar el repositorio**
- git clone https://github.com/nanucasa/TP_grupal.git
- cd TP_grupal

**Crear y activar el entorno conda**
- conda create -n tp_grupal python=3.10 -y
- conda activate tp_grupal

**Instalar dependencias del proyecto**
- pip install -r requirements.txt

**Configurar el remoto de DVC (DagsHub) con credenciales personales**
- dvc remote modify origin --local auth basic
- dvc remote modify origin --local user TU_USUARIO_DAGSHUB
- dvc remote modify origin --local password TU_TOKEN_DAGSHUB

**Sincronizar datos versionados desde DagsHub**
- dvc pull

**Ejecutar la preparación de datos con DVC (solo data prep)**  
- `dvc repro data_prep`

Esto:
- Toma `data/raw/telco_churn.csv`.
- Genera `data/processed/train.csv`, `data/processed/valid.csv`,
  `data/processed/test.csv` y `data/processed/features.json`.

**Entrenar el modelo y loguear en MLflow (local + remoto)**  
- `python scripts/base_scripts_runs.py`

Esto:
- Usa los CSV de `data/processed/`.
- Entrena el modelo XGBoost (`TelcoChurn_XGB`).
- Registra un run en el experimento `telco_churn_tune_xgb`.
- Registra/actualiza el modelo `TelcoChurn_XGB` en el Model Registry remoto de DagsHub
  y gestiona el alias `champion`.

**Ver resultados en DagsHub / MLflow (tracking remoto)**
- Ir al repo en DagsHub.
- Abrir la pestaña “Experiments” (es la UI de MLflow remoto).
- Seleccionar el experimento telco_churn_tune_xgb para ver runs, métricas y modelos registrados.

**(Opcional pero recomendado)** 
- Levantar Dagster para monitoreo y automatización
- cd tp_grupal_dagster
- dagster dev

- Luego abrir en el navegador: http://127.0.0.1:3000
Desde allí se visualizan los assets, el sensor champion_sensor y el modelo campeón actualizado.

#### NOTA: El MLflow local (file:mlruns) puede existir, pero la fuente de verdad del proyecto y de Dagster es SIEMPRE el MLflow remoto de DagsHub.

## 📁 Estructura del Proyecto

- TP_grupal/
- ├── data/
- │ ├── raw/ # Datos originales (DVC)
- │ └── processed/ # Datos limpios (DVC)
- ├── src/
- │ ├── data_prep.py # Limpieza y split de datos
- │ └── train.py # Entrenamiento + MLflow logging
- ├── models/ # Modelos entrenados
- │ └── model.joblib
- ├── tp_grupal_dagster/
- │ └── tp_grupal_dagster/
- │ ├── assets.py # Assets Dagster
- │ ├── definitions.py # Definitions Dagster
- │ └── init.py
- ├── artifacts/
- │ └── champion_metadata.json # Información del modelo campeón
- ├── reports/ # Gráficos (ROC, PR)
- ├── params.yaml # Parámetros del modelo
- ├── dvc.yaml # Definición del pipeline
- ├── dvc.yaml
- ├── dvc.lock
- ├── params.yaml
- ├── requirements.txt
- └── .github/
-     └── workflows/
-        ├── ci.yml
-        └── cd_retrain.yml
- └── README.md

## 🔄 Pipeline de Trabajo (DVC)

### Stage 1 – `data_prep` (DVC)

**Script:** `src/data_prep.py`

**Funciones:**
- Carga del dataset crudo.
- Limpieza / preprocesamiento.
- Split en train / valid / test.
- Generación del archivo `features.json` con la lista de features.

**Entradas:**
- `data/raw/telco_churn.csv`
- `params.yaml`

**Salidas:**
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `data/processed/features.json`

**Este stage se ejecuta automáticamente cuando se corre:**
- `dvc repro data_prep`
- o simplemente `dvc repro` (es el único stage actual del pipeline).

### Stage 2 – Entrenamiento y logging (fuera de DVC)

**Script:** `scripts/base_scripts_runs.py`

**Funciones:**
- Carga los datos de `data/processed/`.
- Entrena el modelo XGBoost (`TelcoChurn_XGB`).
- Calcula métricas (F1, accuracy, precision, recall, etc.).
- Loguea resultados en MLflow local y remoto (DagsHub).
- Registra y actualiza el modelo `TelcoChurn_XGB` en el Model Registry,
  incluyendo la administración del alias `champion`.

**Comando:**
- python scripts/base_scripts_runs.py

## 📚 Comandos útiles
# Ver estado del pipeline
- dvc status

# Reproducir solo la preparación de datos
- dvc repro data_prep

# Ver el grafo del pipeline
- dvc dag

## 📚 Guía rápida paso a paso (resumen)

**1- Preparar entorno**
- Git clone del repositorio.
- Crear y activar entorno conda tp_grupal.
- Instalar requirements.txt.

**2- Sincronizar datos**
- Configurar remoto DVC con usuario/token de DagsHub.
- Ejecutar dvc pull.

**3- Ejecutar el pipeline completo**
- Ejecutar `dvc repro data_prep`.

**4- Entrenar el modelo y registrar experimentos**  
- Ejecutar `python scripts/base_scripts_runs.py`.

**5- Verificar que se generen:**  
- `data/processed/train.csv`, `valid.csv`, `test.csv`, `features.json`.  
- Nuevos runs en el experimento `telco_churn_tune_xgb` en la pestaña **Experiments** de DagsHub.  
- Nuevas versiones del modelo `TelcoChurn_XGB` en la pestaña **Models** de DagsHub, con el alias `champion` actualizado.

**6- Monitoreo y automatización con Dagster**  
- Desde la raíz del proyecto de orquestación:  
  - `cd tp_grupal_dagster`  
  - `dagster dev`  

- Abrir `http://127.0.0.1:3000` y revisar:
  - Assets de champion.  
  - Sensor `champion_sensor` (detecta nuevo campeón).  

**7- Confirmar modelo campeón:**  
- Revisar `tp_grupal_dagster/artifacts/champion_metadata.json` (actualizado por Dagster a partir del MLflow remoto de DagsHub).  
- Verificar en el Model Registry de DagsHub que la versión correspondiente de 
  `TelcoChurn_XGB` tenga el alias `champion`.

**8- Confirmar modelo campeón:**
- Revisar artifacts/champion_metadata.json (actualizado por Dagster; siempre refleja el MLflow remoto de DagsHub).
- Verificar que en MLflow Model Registry (en la UI de DagsHub) el modelo tenga alias champion.

## 🧩 Monitoreo y Automatización con Dagster

- Dagster monitorea automáticamente el experimento telco_churn_tune_xgb en el MLflow remoto de DagsHub y selecciona el mejor run según la métrica F1.

### Cuando detecta un nuevo campeón:

**Materializa los assets:**
- select_champion_from_mlflow
- persist_champion_json
- set_mlflow_champion_alias

- Actualiza el archivo local artifacts/champion_metadata.json con la información del nuevo campeón (lectura siempre desde DagsHub).
- Asigna el alias champion al modelo correspondiente en el Model Registry de MLflow (también en DagsHub).
- El sensor champion_sensor solo se dispara si existe un nuevo run con F1 superior al actual, evitando ejecuciones en bucle innecesarias.

### ¿En qué MLflow se ve el champion y cómo llegar?

- Ir a: **https://dagshub.com/nanucasa/TP_grupal**
- Abrir la pestaña “Experiments” (UI de MLflow remoto).
- Seleccionar el experimento telco_churn_tune_xgb.
- Ordenar la columna f1 de mayor a menor.
- El primer run de la tabla es el campeón; su run_id coincide con el que aparece en artifacts/champion_metadata.json.

**Para ver el modelo en el registry:**
- Desde la misma UI de MLflow en DagsHub, ir a la pestaña “Models”.
- Abrir el modelo TelcoChurn_XGB.
- Verificar que la versión correspondiente tenga el alias champion.
- Todo esto se realiza SIEMPRE en el MLflow remoto de DagsHub; Dagster nunca consulta el mlruns local.

### Otra opción para ver el run champiion del experimento:

**Documentalmente**
- Dagster genera un documento .json de metadata donde podemos ver la infromación del champion:
dvc_prueba\tp_grupal_dagster\artifacts\champion_metadata.json

**Desde el Anaconda prompt3 del servidor de dagster:**
- El champion sensor automatizado, genera una lectura cada 30 segundos en busqueda de los runs actuales para detectar el nuevo champion, en ese mismo prompt3 podemos acceder rapidamente al run_id champion con el mejor f1. 

## 🧪 Experimentos y Modelo Campeón

- El mejor modelo actual proviene del experimento telco_churn_tune_xgb.
- Dagster detectó automáticamente el siguiente champion registrado en artifacts/champion_metadata.json:

| Atributo          | Valor                                      |
|-------------------|--------------------------------------------|
| Experimento       | `telco_churn_tune_xgb`                     |
| Modelo            | `TelcoChurn_XGB`                           |
| Run ID            | `53e572e30c7a46a49764166eb55a7302`         |
| Métrica principal | `F1`                                       |
| Valor F1          | ≈ **0.603** (`0.6028037383…`)              |

-Este modelo es el que queda marcado con el alias champion en el Model Registry de MLflow.

## 📈 Reproducibilidad y CI/CD

**Comandos útiles DVC:**
- `dvc repro data_prep`
- `dvc dag`
- `dvc status`

### Workflows de GitHub Actions

**1) CI (`.github/workflows/ci.yml`)**

- Se ejecuta en cada `push` y `pull_request` a la rama `main`.
- Pasos principales:
  - Checkout del repositorio.
  - Configuración de Python 3.11.
  - Instalación de dependencias desde `requirements.txt`.
  - `dvc pull` usando los secrets de DagsHub (si falta algo en el remoto, el job no falla).
  - Chequeo de sintaxis del código:

    ```bash
    python scripts/base_scripts_runs.py
    ```

  - Verificación del `MLFLOW_TRACKING_URI`
    (imprime el valor de la variable de entorno y el `mlflow.get_tracking_uri()`).

**2) CD – Retrain model and push to Dagshub (`.github/workflows/cd_retrain.yml`)**

- Se dispara manualmente desde la pestaña **Actions** (`workflow_dispatch`).
- Usa secrets del repo:
  - `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`
  - `MLFLOW_TRACKING_URI` (URL del tracking remoto en DagsHub).

- Pasos principales:
  - Checkout del repositorio.
  - Configuración de Python 3.11.
  - Instalación de dependencias.
  - Configuración del remoto DVC apuntando a DagsHub y `dvc pull` (mejor esfuerzo).
  - **Sanity check MLflow**: crea el experimento `ci_cd_sanity` y el run
    `gh_actions_smoke` en el MLflow remoto para confirmar credenciales/URI.
  - Ejecuta:

    ```bash
    python scripts/base_scripts_runs.py
    ```

    para reentrenar el modelo y loguear nuevos runs en el experimento
    `telco_churn_tune_xgb`, además de actualizar el modelo `TelcoChurn_XGB`
    y el alias `champion` en el Model Registry de DagsHub.
  - `dvc push` para subir datos/artefactos versionados al remoto de DagsHub
    (si falla, el workflow no se rompe).

## 🧩 Visualizaciones

**El flujo genera:**
- Curvas ROC y PR en la carpeta reports/.
- Tabla de métricas consolidada desde MLflow en la UI de DagsHub.
- Imagen comparativa de F1 por modelo (reports/f1_bench_dagster.png).
- Asset de champion actualizado (artifacts/champion_metadata.json).

## 📌 Resultados Finales

**Modelo en “producción” (Champion actual):**
- Algoritmo: TelcoChurn_XGB (clasificador XGBoost para churn)
- Experimento: telco_churn_tune_xgb
- F1-score ≈ 0.603
- Registrado automáticamente en el MLflow remoto de DagsHub como alias champion
- El sistema Dagster + MLflow permite mantener actualizado este modelo sin intervención manual, garantizando trazabilidad total del pipeline.

## 🚀 Deployment

- La estrategia de deployment propuesta (API REST, batch, monitoreo y reentrenamiento) se documenta en:
**DEPLOYMENT.md**

## 👤 Autores

- Nadia Soledad Casá
- Karla Silva

- Curso: Laboratorio de Minería de Datos – ISTEA
- Año: 2025


"Nota: esta sección fue editada desde la rama demo_cliente para la demo."