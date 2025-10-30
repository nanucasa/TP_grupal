
# Lee métricas de test, elige el mejor modelo por F1 y asigna alias "champion"/"challenger" en MLflow Model Registry.
# Guarda un resumen en artifacts/selection.json

import argparse, json, os, time
from pathlib import Path
from typing import Dict, List, Tuple

from mlflow.tracking import MlflowClient

METRICS_CANDIDATES = {
    "LogReg": "metrics_test.json",
    "RF":     "metrics_test_rf.json",
    "XGB":    "metrics_test_xgb.json",
    "FE":     "metrics_test_fe.json",
}

MODEL_NAMES = {
    "LogReg": "TelcoChurn_LogReg",
    "FE":     "TelcoChurn_LogReg",   # FE también registró como LogReg (nueva versión)
    "RF":     "TelcoChurn_RF",
    "XGB":    "TelcoChurn_XGB",
}

def load_metric_file(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # soportar ambos formatos {test:{...}} o plano {...}
    return data.get("test", data)

def best_two_by_f1() -> List[Tuple[str, Dict]]:
    rows: List[Tuple[str, Dict]] = []
    for key, rel in METRICS_CANDIDATES.items():
        p = Path(rel)
        if p.exists():
            try:
                m = load_metric_file(p)
                f1 = float(m.get("f1", 0.0))
                rows.append((key, {"f1": f1, **m}))
            except Exception:
                pass
    rows.sort(key=lambda kv: kv[1].get("f1", 0.0), reverse=True)
    return rows[:2]

def latest_version_for(client: MlflowClient, model_name: str) -> int:
    vers = client.search_model_versions(f"name='{model_name}'")
    if not vers:
        raise RuntimeError(f"No hay versiones en el registry para {model_name}")
    return max(int(v.version) for v in vers)

def clear_alias_everywhere(client: MlflowClient, alias: str):
    # Quita alias de todos los modelos que lo tengan
    for rm in client.search_registered_models():
        try:
            client.delete_registered_model_alias(rm.name, alias)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    ap.add_argument("--write-out", required=True)  # artifacts/selection.json
    args = ap.parse_args()

    os.makedirs(Path(args.write_out).parent, exist_ok=True)

    top = best_two_by_f1()
    if not top:
        raise SystemExit("No se encontraron métricas de test.")

    client = MlflowClient(tracking_uri=args.mlflow_uri)

    # Campeón
    champ_key, champ_metrics = top[0]
    champ_model = MODEL_NAMES[champ_key]
    champ_ver = latest_version_for(client, champ_model)

    # Retador (si existe)
    chall = None
    if len(top) > 1:
        chall_key, chall_metrics = top[1]
        chall_model = MODEL_NAMES[chall_key]
        try:
            chall_ver = latest_version_for(client, chall_model)
            chall = (chall_key, chall_model, chall_ver, chall_metrics)
        except Exception:
            chall = None

    # Asignar aliases (nuevo esquema recomendado por MLflow)
    clear_alias_everywhere(client, "champion")
    client.set_registered_model_alias(champ_model, "champion", str(champ_ver))

    if chall:
        clear_alias_everywhere(client, "challenger")
        client.set_registered_model_alias(chall[1], "challenger", str(chall[2]))

    # Persistir resumen
    out = {
        "ts": int(time.time()),
        "mlflow_uri": args.mlflow_uri,
        "champion": {
            "key": champ_key,
            "model_name": champ_model,
            "version": champ_ver,
            "f1": champ_metrics.get("f1"),
            "metrics": champ_metrics,
        },
        "challenger": None,
        "considered": [
            {"key": k, "f1": v.get("f1"), "metrics": v} for k, v in top
        ],
    }
    if chall:
        out["challenger"] = {
            "key": chall[0],
            "model_name": chall[1],
            "version": chall[2],
            "f1": chall[3].get("f1"),
            "metrics": chall[3],
        }

    with open(args.write_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] champion={champ_model} v{champ_ver} | summary -> {args.write_out}")

if __name__ == "__main__":
    main()
