# Proyecto: Telco Churn – Pipeline reproducible (DVC + MLflow)

Pipeline de Minería de Datos con **DVC** (datos/artefactos), **MLflow** (experimentos y registro de modelos), y control de versiones en **Git** (GitHub + DagsHub).

---

## Requisitos
- Python/Conda (entorno: `tp_grupal`)
- Instalar dependencias del repo:
  ```bash
  pip install -r requirements.txt


ESTRUCTURA DE NUESTRAS CARPETAS

C:\dvc_prueba
├─ data
│  ├─ raw\            # datos crudos (DVC)
│  └─ processed\      # datos procesados (DVC)
├─ models\            # modelos entrenados (.joblib, DVC)
├─ artifacts\         # parámetros/umbrales (JSON, Git)
├─ predictions\       # inferencias (CSV, DVC)
├─ reports\           # gráficos (PNG, DVC)
├─ src\               # código (Python, Git)
├─ scripts\           # utilidades (Git)
├─ dvc.yaml / dvc.lock
├─ params.yaml
└─ README.md


STAGES DEL PIPELINE DVC

data_prep
  ├─ tune ──► train ──► threshold ──► predict_test ──► evaluate_test
  └─ tune_rf ──► train_rf ──► predict_test_rf ──► evaluate_test_rf


CÓMO REPRODUCIR EL PIPELINE

dvc repro
dvc metrics show

COMO EJECUTAR ETAPAS PUNTUALES

dvc repro -f tune
dvc repro -f train
dvc repro -f threshold
dvc repro -f evaluate_test
dvc repro -f tune_rf
dvc repro -f train_rf
dvc repro -f evaluate_test_rf
dvc repro -f predict_test
dvc repro -f predict_test_rf


MLflow (tracking local)
CÓMO INICIAR UI LOCAL: 

mlflow ui --backend-store-uri sqlite:///C:/dvc_prueba/mlflow.db --default-artifact-root file:///C:/dvc_prueba/mlruns --host 127.0.0.1 --port 5001

UI: http://127.0.0.1:5001

Experimentos: telco_churn_tune, telco_churn_baseline, telco_churn_threshold, telco_churn_eval, telco_churn_tune_rf, telco_churn_baseline_rf

Model Registry: TelcoChurn_LogReg, TelcoChurn_RF (Staging)

DATOS Y ARTEFACTOS:

DVC: datasets procesados, modelos (models/*.joblib), predicciones (predictions/*.csv), y gráficos (reports/*.png).

Git: código (src/, scripts/), configuración (params.yaml), métricas (metrics_*.json) y JSONs chicos en artifacts/.
dvc push

RESULTADOS (Benchmark)

Esta tabla se actualiza automáticamente con python scripts/update_readme_metrics.py.

<!-- METRICS_START -->

## Resultados (Benchmark)

_Las métricas provienen de los JSON versionados por DVC (valid/test)._

| Model | Split | Threshold | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AP |
|---|---|---|---|---|---|---|---|---|
| LogReg | valid | 0.4542 | 0.6520 | 0.5134 | 0.7934 | 0.6234 | 0.2582 | 0.6001 |
| LogReg | test | 0.4542 | 0.6330 | 0.4969 | 0.7772 | 0.6062 | 0.7362 | 0.5989 |
| RF | valid |  | 0.6650 | 0.5277 | 0.7355 | 0.6145 |  |  |
| RF | test | 0.4542 | 0.5965 | 0.4691 | 0.8363 | 0.6011 | 0.7316 | 0.6071 |
<!-- METRICS_END -->

FLUJO DE LOS COMMIT Y EL PUSH 

Versionar cambios en Git:
git add .
git commit -m "feat/fix: descripción corta del cambio"

Push a DagsHub (primero):
git push -u dagshub main

Push a GitHub (Credential Manager):
git remote set-url origin https://github.com/nanucasa/TP_grupal.git
git config --global credential.helper manager
git push -u origin main --verbose

Subir artefactos DVC:
dvc push

Scripts auxiliares:
scripts/update_readme_metrics.py: actualiza la sección de métricas del README leyendo metrics_*.json y artifacts/best_threshold.json.

Reproducibilidad rápida
# Procesar datos, entrenar, sintonizar, umbral, evaluar, predecir
dvc repro

# Actualizar tabla de métricas en README
python scripts/update_readme_metrics.py

# Ver métricas
dvc metrics show
