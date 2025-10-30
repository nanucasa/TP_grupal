Proyecto: Telco Churn — Pipeline reproducible (DVC + MLflow)

Pipeline de Minería de Datos con DVC (datos/artefactos), MLflow (experimentos y registro de modelos) y control de versiones en Git (GitHub + DagsHub). Entorno de trabajo: Windows, conda env tp_grupal, raíz del repo: C:\dvc_prueba.

REQUISITOS

Python/Conda (entorno: tp_grupal)

Instalar dependencias: pip install -r requirements.txt

Directorio raíz de trabajo: C:\dvc_prueba

ESTRUCTURA DE CARPETAS

C:\dvc_prueba
├─ data
│ ├─ raw\ (DVC)
│ ├─ processed\ (DVC)
│ └─ features\ (DVC)
├─ models\ (DVC)
├─ artifacts\ (Git; JSON chicos p/decisiones y selección)
├─ predictions\ (DVC)
├─ reports\ (DVC)
├─ src\ (Git; código)
├─ scripts\ (Git; utilidades)
├─ dvc.yaml / dvc.lock (Git)
├─ params.yaml (Git)
└─ README.md (Git)

Nota DVC vs Git:

DVC (no Git): data/, models/, predictions/, reports/.

Git: src/, scripts/, dvc.yaml, dvc.lock, params.yaml, métricas *.json, artifacts/selection.json.

STAGES DEL PIPELINE (DVC)

data_prep
├─ tune ──► train ──► threshold ──► predict_test ──► evaluate_test
├─ tune_rf ──► train_rf ──► evaluate_test_rf (y opcional predict_test_rf)
└─ feature_eng ──► train_fe ──► threshold_fe ──► predict_test_fe ──► evaluate_test_fe

Selección y Registro:
select_register (elige campeón por métrica de test y crea/actualiza aliases en MLflow Model Registry; genera artifacts/selection.json)

CÓMO REPRODUCIR EL PIPELINE

Ejecutar todo el pipeline:
dvc repro

Ver métricas agregadas:
dvc metrics show

Ejecutar etapas puntuales (ejemplos):
dvc repro -f data_prep
dvc repro -f tune
dvc repro -f train
dvc repro -f threshold
dvc repro -f predict_test
dvc repro -f evaluate_test
dvc repro -f tune_rf
dvc repro -f train_rf
dvc repro -f evaluate_test_rf
dvc repro -f feature_eng
dvc repro -f train_fe
dvc repro -f threshold_fe
dvc repro -f predict_test_fe
dvc repro -f evaluate_test_fe
dvc repro -f select_register

Después de cada corrida relevante:
dvc push

MLFLOW (TRACKING LOCAL + REGISTRY)

Iniciar servidor MLflow (2.22.2) en local:
mlflow server --backend-store-uri sqlite:///C:/dvc_prueba/mlflow_222.db --default-artifact-root file:///C:/dvc_prueba/mlruns --host 127.0.0.1 --port 5001

UI local:
http://127.0.0.1:5001

Variable de entorno para scripts:
Windows (cmd): set MLFLOW_TRACKING_URI=http://127.0.0.1:5001

Experimentos usados:
telco_churn_tune, telco_churn_baseline, telco_churn_threshold, telco_churn_eval
telco_churn_tune_rf, telco_churn_baseline_rf
telco_churn_tune_xgb (y análogos si aplica FE: *_fe)

Model Registry:
TelcoChurn_LogReg, TelcoChurn_RF, TelcoChurn_XGB
Aliases: “champion” (mejor en test), “challenger” (segundo)

RESULTADOS (BENCHMARK)

La tabla siguiente se actualiza automáticamente con scripts/update_readme_metrics.py, leyendo metrics*.json y artifacts/best_threshold*.json.

<!-- METRICS_START -->

| Modelo | Split | Accuracy | Precision | Recall | F1 | Threshold |
|---|---|---:|---:|---:|---:|---:|
| FE | test | - | - | - | - | 0.4525 |
| LogReg | test | - | - | - | - | 0.4525 |
| RF | test | - | - | - | - | - |
| XGB | test | - | - | - | - | 0.2973 |
| FE | valid | - | - | - | - | 0.4525 |
| LogReg | valid | - | - | - | - | 0.4525 |
| RF | valid | - | - | - | - | - |
| XGB | valid | - | - | - | - | 0.2973 |

<!-- METRICS_END -->

FLUJO DE VERSIONADO (GIT + DAGSHUB + DVC)

Versionar cambios en Git (código, dvc.yaml/lock, métricas y artifacts/selection.json):
git add .
git commit -m "feat: descripción breve del cambio"

Push a DagsHub (primero):
git push -u dagshub main

Push a GitHub (Credential Manager):
git remote set-url origin https://github.com/nanucasa/TP_grupal.git

git config --global credential.helper manager
git push -u origin main --verbose

Subir artefactos DVC (datasets procesados, modelos, predicciones, gráficos):
dvc push

Comentario de commit sugerido (breve y claro):
feat: agrega nueva stage / fix: corrige bug / docs: actualiza README / chore: mantenimiento

EJECUCIONES TÍPICAS

Entrenamiento base + evaluación:
dvc repro

Rama RF:
dvc repro -f tune_rf
dvc repro -f train_rf
dvc repro -f evaluate_test_rf

Rama FE:
dvc repro -f feature_eng
dvc repro -f train_fe
dvc repro -f threshold_fe
dvc repro -f evaluate_test_fe

Selección y registro del mejor modelo (actualiza aliases en Registry y artifacts/selection.json):
dvc repro -f select_register

Predicciones por lote (según rama/modelo):
dvc repro -f predict_test
dvc repro -f predict_test_fe
(dvc generará predictions/*.csv; subir luego con dvc push)

PARÁMETROS Y MÉTRICAS

Parámetros (params.yaml):

train.C, train.max_iter, train.seed

Grillas de tuneo por modelo (p. ej. tune.C_grid, tune_rf., tune_xgb.) si aplica

Métricas (Git):

metrics.json, metrics_test.json, metrics_threshold.json, metrics_tune*.json

variantes RF/FE/XGB análogas (metrics_rf.json, metrics_test_rf.json, metrics_fe.json, metrics_test_fe.json, etc.)

artifacts/selection.json (resumen del campeón/challenger)

NOTAS

No agregar manualmente a Git salidas de DVC (data/, models/, predictions/, reports/). Usar siempre dvc push para publicarlas.

Si alguna regla .gitignore impide versionar artifacts/selection.json, añadir excepción: !artifacts/selection.json

MLflow local corre en 127.0.0.1:5001 usando C:/dvc_prueba/mlflow_222.db y C:/dvc_prueba/mlruns como raíz de artefactos.

CRÉDITOS

Materia: Laboratorio de Minería de Datos. Proyecto Telco Churn — pipeline reproducible con DVC + MLflow, tracking/registry local y sincronización a DagsHub y GitHub.