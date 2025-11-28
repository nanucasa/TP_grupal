# scripts/run_nanu_runs.py
# Lanza ~50 runs de train.py con nombres nanu_run_001..nanu_run_050

import os
import sys
import subprocess
from pathlib import Path


def main():
    # Raíz del proyecto: C:\dvc_prueba
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train.py"

    train_path = project_root / "data" / "processed" / "train.csv"
    valid_path = project_root / "data" / "processed" / "valid.csv"
    params_path = project_root / "params.yaml"
    best_params_path = project_root / "artifacts" / "best_params.json"

    if not train_script.exists():
        raise FileNotFoundError(f"No se encontró train.py en {train_script}")
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "Faltan data/processed/train.csv o data/processed/valid.csv. Corre primero dvc repro."
        )
    if not params_path.exists():
        raise FileNotFoundError("No se encontró params.yaml en el directorio raíz del proyecto.")

    use_best = best_params_path.exists()

    for i in range(1, 51):
        run_name = f"nanu_run_{i:03d}"
        out_model = project_root / "models" / f"model_{run_name}.joblib"
        metrics_path = project_root / "reports" / f"metrics_{run_name}.json"

        cmd = [
            sys.executable,
            str(train_script),
            "--train", str(train_path),
            "--valid", str(valid_path),
            "--out-model", str(out_model),
            "--metrics", str(metrics_path),
            "--params", str(params_path),
            "--run-name", run_name,
        ]
        if use_best:
            cmd.extend(["--best-params", str(best_params_path)])

        print(f"=== Ejecutando {run_name} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARN] {run_name} terminó con código {result.returncode}, deteniendo el loop.")
            break


if __name__ == "__main__":
    main()
