# TP Grupal – Telco Churn

- Proyecto MLOps de Predicción de Churn
- Proyecto ISTEA | Materia: Laboratorio de Minería de Datos

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

- Construir un pipeline de ML completamente reproducible.
- Aplicar control de versiones con DVC y Git.
- Trackear experimentos y modelos con MLflow (DagsHub).
- Orquestar, visualizar y automatizar la selección del modelo campeón con Dagster.
- Implementar CI/CD con GitHub Actions para validar el pipeline.
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

**Ejecutar el pipeline de punta a punta con DVC (data prep + train)**
- dvc repro train

Esto:
- Toma data/raw/telco_churn.csv
- Genera data/processed/train.csv y data/processed/valid.csv
- Entrena el modelo y guarda models/model.joblib
- Genera metrics.json y gráficos en reports/
- Loguea el experimento y el modelo en MLflow remoto en DagsHub (tracking URI: https://dagshub.com/nanucasa/TP_grupal.mlflow).

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
- ├── requirements.txt
- └── README.md

## 🔄 Pipeline de Trabajo (DVC)
**Stage 1 – data_prep**
Script: src/data_prep.py

- Funciones:
- Carga del dataset crudo.
- Limpieza, encoding y división en train/valid.

- Entradas:
- data/raw/telco_churn.csv
- params.yaml

- Salidas:
- data/processed/train.csv
- data/processed/valid.csv

**Este stage se ejecuta automáticamente cuando se corre:**
- dvc repro train 
- detecta si cambió el CSV o params.yaml.

**Stage 2 – train**
- Script: src/train.py

**Funciones:**
- Entrena modelo LogisticRegression con StandardScaler.
- Calcula métricas (accuracy, precision, recall, F1, ROC-AUC).
- Loguea resultados en MLflow (local y remoto en DagsHub).

**Entradas:**
- data/processed/train.csv
- data/processed/valid.csv
- params.yaml

**Salidas:**
- models/model.joblib
- metrics.json

## 📚 Guía rápida paso a paso (resumen)

**1- Preparar entorno**
- Git clone del repositorio.
- Crear y activar entorno conda tp_grupal.
- Instalar requirements.txt.

**2- Sincronizar datos**
- Configurar remoto DVC con usuario/token de DagsHub.
- Ejecutar dvc pull.

**3- Ejecutar el pipeline completo**
- Ejecutar dvc repro train.

**4- Verificar que se generen:**
- data/processed/train.csv y valid.csv
- models/model.joblib
- metrics.json y gráficos en reports/
- Ver experimentos en MLflow (DagsHub)
- Ir al repo de DagsHub.
- Abrir “Experiments” → experimento telco_churn_tune_xgb.
- Ver los runs con sus métricas (F1, accuracy, etc.) y modelos registrados.
- Monitoreo y automatización con Dagster

**5- Desde la raíz del proyecto de orquestación:**
- cd tp_grupal_dagster
- dagster dev

**6- Abrir http://127.0.0.1:3000**

**7- Revisar:**
- Assets de champion.
- Sensor champion_sensor (detecta nuevo campeón).

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

- Atributo            Valor
- ____________________________________________________
- Experimento         telco_churn_tune_xgb
- Modelo              TelcoChurn_XGB
- Run ID              53e572e30c7a46a49764166eb55a7302
- Métrica principal   F1
- Valor F1            ≈ 0.603 (0.6028037383…)

-Este modelo es el que queda marcado con el alias champion en el Model Registry de MLflow.

## 📈 Reproducibilidad y CI/CD

**Comandos útiles DVC:**
- dvc repro train
- dvc dag
- dvc status
- dvc params diff

**Automatización GitHub Actions:**
- Instala dependencias.
- Ejecuta dvc pull y dvc repro.
- Conecta con DagsHub usando secrets del repositorio.

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