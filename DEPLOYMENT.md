# Estrategia de Deployment - TP Grupal Telco Churn

📝 ## Resumen del Modelo

- **Algoritmo:** Logistic Regression (pipeline `StandardScaler + LogisticRegression` sobre features ingenieradas)
- **Dataset:** `telco_churn.csv` (~10.000 clientes de telecomunicaciones)
- **Métrica principal:** F1-Score en test (`test_f1`)
- **Métricas aproximadas (conjunto de test):**
  - Accuracy ≈ 66 %
  - F1-Score ≈ 60 %
  - ROC-AUC ≈ 74 %
  - PR-AUC ≈ 60 %

El modelo campeón se selecciona automáticamente a partir de los runs registrados en MLflow (tracking remoto en DagsHub), usando el script `scripts/update_champion_from_runs.py`, que:

- Toma el experimento `telco_churn_tune_xgb`.
- Filtra los runs cuyo nombre comienza con `metrics_test`.
- Ordena por `metrics.test_f1`.
- Elige el mejor como **champion** y guarda la información en `artifacts/champion_run.json`.

-----------------------------------

## Propuesta de Arquitectura

### Opción 1: API REST con FastAPI

Servicio REST que expone un endpoint de scoring para predecir churn en línea para uno o varios clientes.

```python
from fastapi import FastAPI
import joblib
import pandas as pd
import json
from pathlib import Path

app = FastAPI()

# Cargar información del champion (run_id, métrica, etc.)
champion_path = Path("artifacts/champion_run.json")
with open(champion_path, "r", encoding="utf-8") as f:
    champion = json.load(f)

# Versión simple: cargar el modelo final entrenado
# En este TP el modelo final se guarda como models/model_fe.joblib
model = joblib.load("models/model_fe.joblib")


@app.post("/predict")
def predict(customer_data: dict):
    df = pd.DataFrame([customer_data])

    # Probabilidad de churn (clase positiva)
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "churn": pred,
        "probability": float(proba),
        "champion_run_id": champion["run_id"],
        "champion_experiment": champion["experiment_name"],
    }
```
---------------------------------

Stack sugerido:
	- FastAPI + Uvicorn
	- Docker container
	- Deploy en Azure Container Apps o GCP Cloud Run
	- MLflow (en DagsHub) como registry de modelos y runs

El servicio puede actualizarse leyendo periódicamente artifacts/champion_run.json o exponiendo un endpoint interno para recargar el modelo cuando cambie el champion.

---------------------------------

Opción 2: Batch Processing

1- Pipeline batch programado (por ejemplo con Dagster) que:
2- Lee periódicamente nuevos clientes desde una base de datos (clientes activos).
3- Genera predicciones de churn usando el modelo campeón.
4- Guarda resultados en una tabla de scoring (ej. churn_scores) con:
	- customer_id
	- score_churn
	- flag_alto_riesgo
	- fecha_scoring
4- Dispara reportes o dashboards para el área de negocio (lista de clientes de alto riesgo para acciones de retención).

Stack sugerido:
- Dagster para orquestación de jobs batch.
- Base de datos: PostgreSQL o BigQuery (según infraestructura).
- Jobs containerizados (Docker) ejecutados en:
	- VM con cron + Dagster, o
	- Cluster ligero (Docker Swarm / Kubernetes) si la escala lo justifica.
- MLflow (DagsHub) como fuente de verdad del champion y de las métricas históricas.

---------------------------------

🔎 Monitoreo

Métricas a trackear
1- Performance del modelo
	- Accuracy, F1, ROC-AUC, PR-AUC en ventanas de tiempo recientes.
	- Comparación vs. métricas del conjunto de test original.

Data Drift
- Distribución de features críticas (ej. tenure_months, monthly_charges, contract_type) vs. entrenamiento.
- Detección de cambios en la proporción de clases (churn vs no churn).

Prediction Drift
- Distribución de probabilidades de churn a lo largo del tiempo.
- Porcentaje de clientes marcados como alto riesgo.

Herramientas
- MLflow + DagsHub para:
	Registrar nuevos runs.
	Comparar métricas entre versiones de modelo.
	Llevar historial de champions.

- Evidently AI (u otra librería similar) para:
	Reportes de data drift y prediction drift.
	Comparación entre dataset de entrenamiento y datos recientes.
- Grafana / Loki o servicios equivalentes para:
	Monitorear tiempos de respuesta, tasa de errores, throughput.
	Visualizar logs estructurados del endpoint /predict o de los jobs batch.

---------------------------------

♻️ Actualización del Modelo

Triggers de reentrenamiento

- Degradación de performance:
	- test_f1 de datos recientes cae por debajo de un umbral (ej. 0.55).
- Data drift significativo en uno o más features clave.
- Política de calendario:
	- Reentrenar cada N meses (ej. trimestralmente) aunque no haya alarmas.

Proceso propuesto:

1- Reentrenamiento con DVC
- Actualizar datos (nuevos períodos / nuevos clientes) en el remoto de DVC.
- Ejecutar:
	- dvc pull
	- dvc repro train_fe
- Esto vuelve a correr el pipeline de preparación, features y entrenamiento.

2- Registro de experimentos
- Los nuevos runs se registran en MLflow (en DagsHub), incluyendo:
	- parámetros,
	- métricas de train/valid/test,
	- artefactos (modelo, gráficos, etc.).
3- Selección automática de champion
	- Ejecutar: python scripts/update_champion_from_runs.py
El script:

- toma el experimento telco_churn_tune_xgb,
- filtra metrics_test*,
- elige el mejor test_f1,
- actualiza artifacts/champion_run.json.

4- Actualización de servicios

- API REST:  El servicio recarga el modelo (por lectura de champion_run.json o por reinicio controlado del contenedor).

- Batch (Dagster): Los jobs de scoring usan el modelo asociado al nuevo champion en los siguientes ciclos.

5- Validación
- Antes de exponer el nuevo modelo a 100 % del tráfico:
	- Revisar métricas en MLflow.
	- Comparar champion nuevo vs anterior.
Opcional: A/B testing sobre una fracción de clientes o tráfico.

---------------------------------

🔐 Consideraciones de Seguridad

Autenticación y autorización
	API protegida con tokens (API Keys) o JWT.
	Control de acceso a endpoints internos (ej. recarga de modelo).

Rate limiting
	Limitar cantidad de requests por unidad de tiempo para cada cliente/aplicación.

Validación de entrada
	Validar tipos, rangos y categorías de cada campo (ej. contract_type, payment_method).
	Descartar o registrar inputs malformados.

Cifrado
	Todo el tráfico hacia la API debe ir sobre HTTPS (TLS).

Logs de auditoría
	Registrar:
		quién llamó al servicio (ID de cliente o sistema),
		qué payload envió (resumen),
		qué score se devolvió,
		timestamp.
	Guardar logs de errores y excepciones.

Protección de secretos
	Tokens de DagsHub, credenciales de BD y claves de acceso en: 
		variables de entorno,
		secretos en el orquestador (Dagster),
		secret manager del cloud.
	Nunca en el código fuente o en el repositorio.

---------------------------------

🫰Estimación de Costos

Ejemplo de estimación para una arquitectura ligera:

API REST (FastAPI + Docker) en servicio gestionado (Azure Container Apps / GCP Cloud Run)
	Tráfico moderado (hasta cientos de miles de requests/mes):
		Costos de cómputo en el orden de pocos dólares mensuales.

Almacenamiento
	Modelos y artefactos en almacenamiento de objetos (Blob Storage / GCS / S3 vía DagsHub):
		Muy bajo costo (centavos de dólar por GB/mes).

Monitoreo
	Stack de observabilidad básico:
		Puede ser auto-hosted (bajo costo infra) o servicio administrado (costo variable).

Los valores exactos dependen del proveedor cloud, región y volumen de tráfico y datos.

---------------------------------

👣 Próximos Pasos

1- Definir el modo principal de uso del modelo:
	online (API REST),
	batch (scoring periódico),
	o combinación de ambos.
2- Crear un Dockerfile que incluya:
	código del proyecto,
	entorno (requirements.txt),
	lógica para cargar el champion desde MLflow / champion_run.json.
3- Integrar el build y deploy en GitHub Actions:
	job de test + dvc repro para validar el pipeline,
	build de imagen,
	deploy automatizado a un entorno de staging.
4- Configurar monitoreo:
	reportes de drift (Evidently / similar),
	dashboards de métricas técnicas (latencia, errores),
	alertas por caída de test_f1 o aumento de errores.
5- Documentar un runbook operacional:
	qué hacer si la API deja de responder,
	qué hacer si las métricas bajan,
	cómo ejecutar reentrenamientos,
	cómo forzar un rollback al champion anterior en caso de problemas.

