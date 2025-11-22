import mlflow
from mlflow.tracking import MlflowClient
import json
import os

EXPERIMENT_NAME = "telco_churn_tune_xgb"
PRIMARY_METRIC = "test_f1"
RUNNAME_PREFIX = "metrics_test"          # solo runs finales de test
OUTPUT_PATH = "artifacts/champion_run.json"


def main():
    # 1) Buscar experimento
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"No existe el experimento: {EXPERIMENT_NAME}")

    # 2) Solo runs cuyo nombre empieza con metrics_test*
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f'tags.mlflow.runName LIKE "{RUNNAME_PREFIX}%"',
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
        max_results=100,
    )

    if df.empty:
        print(
            f"No hay runs de test con nombre que empiece por '{RUNNAME_PREFIX}' "
            f"y métrica {PRIMARY_METRIC}."
        )
        return

    metric_col = f"metrics.{PRIMARY_METRIC}"
    if metric_col not in df.columns:
        print("La métrica buscada no existe en los runs de test:", metric_col)
        print("Métricas disponibles en esos runs:")
        for col in sorted(c for c in df.columns if c.startswith("metrics.")):
            print(" -", col)
        return

    best_row = df.sort_values(metric_col, ascending=False).iloc[0]
    best_run_id = best_row["run_id"]
    best_metric = float(best_row[metric_col])
    run_name = best_row.get("tags.mlflow.runName", "")

    print(
        f"Champion (solo runs de test, por {PRIMARY_METRIC}):\n"
        f"  run_id  = {best_run_id}\n"
        f"  name    = {run_name}\n"
        f"  {PRIMARY_METRIC} = {best_metric:.4f}"
    )

    # 3) Guardar JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": exp.experiment_id,
        "primary_metric": PRIMARY_METRIC,
        "run_id": best_run_id,
        "run_name": run_name,
        "metric_value": best_metric,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Información del champion guardada en {OUTPUT_PATH}")

    # 4) Intentar marcar tag is_champion en MLflow
    client = MlflowClient()

    # limpiar campeones anteriores (si falla, no rompemos)
    try:
        old_champs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string='tags.is_champion = "true"',
            max_results=50,
        )
        for rid in old_champs["run_id"]:
            try:
                client.set_tag(rid, "is_champion", "false")
            except Exception as e:
                print(f"Aviso: no pude limpiar is_champion en {rid}: {e}")
    except Exception as e:
        print("Aviso: no pude buscar campeones anteriores (tags). Detalle:", e)

    # marcar nuevo champion
    try:
        client.set_tag(best_run_id, "is_champion", "true")
        print(f"Tag is_champion=true seteado para run {best_run_id}")
    except Exception as e:
        print(
            "Aviso: no pude setear el tag is_champion en MLflow "
            f"para el run {best_run_id}. El JSON sigue siendo la fuente de verdad."
        )
        print("Detalle técnico:", e)


if __name__ == "__main__":
    main()
