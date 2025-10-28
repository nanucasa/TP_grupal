EL TP GRUPAL MÁS LARGO DE NUESTRAS VIDAS



Telco Churn (DVC + MLflow + Git/GitHub + DagsHub)

Objetivo: Pipeline reproducible para predicción de churn con:



DVC para orquestación y versionado de datos/modelos.



Git/GitHub para código y metadatos.



DagsHub como espejo (y MLflow opcional).



scikit-learn como baseline (Logistic Regression).



Entorno:

Windows + Python 3.11 (conda tp\_grupal)



Dependencias en requirements.txt (incluye pandas, numpy, scikit-learn, mlflow, dvc, etc.)



Estructura principal:

C:\\dvc\_prueba

├─ data

│  ├─ raw\\ telco\_churn.csv

│  └─ processed\\ {train.csv, valid.csv, test.csv, features.json}

├─ src\\ {data\_prep.py, train.py, evaluate.py}

├─ models\\ model.joblib

├─ artifacts\\ best\_params.json   (si tuning)

├─ metrics.json                  (valid)

├─ metrics\_test.json             (test)

├─ params.yaml                   (si tuning)

├─ dvc.yaml, dvc.lock

└─ README.md



Remotos



GitHub (origin): nanucasa/TP\_grupal



DagsHub (dagshub): espejo del repo

DVC remote (data): Google Drive (service account en .dvc/config)



Pipeline (DVC)



data\_prep

Cmd:

python src\\data\_prep.py --input data\\raw\\telco\_churn.csv --outdir data\\processed --seed 42 --test-size 0.2 --val-size 0.1

Out: data/processed/{train.csv, valid.csv, test.csv, features.json}

Tareas: limpieza, normalización de columnas, casting numéricos, one-hot, splits estratificados.



train (scikit-learn + MLflow)

Cmd:

python src\\train.py --train data\\processed\\train.csv --valid data\\processed\\valid.csv --out-model models\\model.joblib --metrics metrics.json --best-params artifacts\\best\_params.json --params params.yaml

Out: models/model.joblib, metrics.json (valid).

Nota: usa class\_weight=balanced. Trackeo MLflow local o DagsHub.



evaluate\_test

Cmd:

python src\\evaluate.py --test data\\processed\\test.csv --model models\\model.joblib --metrics-out metrics\_test.json

Out: metrics\_test.json.



tune (GridSearchCV)

Cmd:

python src\\tune.py --train data\\processed\\train.csv --valid data\\processed\\valid.csv --params params.yaml --best-out artifacts\\best\_params.json --metrics-out metrics\_tune.json

Out: artifacts/best\_params.json, metrics\_tune.json.

params.yaml controla grids y parámetros; train los rastrea con -p.



Ejecutar/Reproducir

conda activate tp\_grupal

dvc repro

dvc dag

dvc metrics show



MLflow

Local: crea mlruns/ (ignorado en Git).



DagsHub (recomendado para UI):

set MLFLOW\_TRACKING\_URI=https://dagshub.com/nanucasa/TP\_grupal.mlflow

set MLFLOW\_TRACKING\_USERNAME=nanucasa

set MLFLOW\_TRACKING\_PASSWORD=<TOKEN\_DAGSHUB>



Flujo de pushes (orden acordado)

1\) DagsHub

git push -u dagshub main



2\) GitHub (Credential Manager, SIEMPRE)

git remote set-url origin https://github.com/nanucasa/TP\_grupal.git

git config --global credential.helper manager

git push -u origin main --verbose



3\) Datos al remoto DVC (GDrive)

dvc push



Comparar métricas:

dvc metrics show

dvc metrics diff



Alcance actual

Pipeline: data\_prep → train → evaluate\_test (+ tune opcional).



Modelos y métricas versionado:

Mirroring a DagsHub HOTED y tracking con MLflow.



Próximos pasos

Otros modelos (árboles, XGBoost/LightGBM).



Balanceo/threshold tuning.



Reporte en docs/ versionado con DVC (manteniendo este README en Git para tener referencia de hasta donde lleguamos y a donde vamos. Pues es un trabajo por etapas.).


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
