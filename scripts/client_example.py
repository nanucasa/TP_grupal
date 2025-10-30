
import argparse
import json
import os
import pandas as pd
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9090/invocations",
                        help="Endpoint del servidor MLflow pyfunc")
    parser.add_argument("--csv", required=True,
                        help="CSV de entrada (sin la columna target)")
    parser.add_argument("--out", default="predictions/preds_from_client.json",
                        help="Archivo de salida con las predicciones (JSON)")
    parser.add_argument("--drop-target", default="Churn",
                        help="Nombre de la columna target a descartar si aparece")
    args = parser.parse_args()

    # Leer CSV y descartar target si viniera incluido
    df = pd.read_csv(args.csv)
    if args.drop_target in df.columns:
        df = df.drop(columns=[args.drop_target])

    # Llamar al endpoint (formato CSV)
    resp = requests.post(
        args.url,
        headers={"Content-Type": "text/csv"},
        data=df.to_csv(index=False)
    )
    resp.raise_for_status()
    payload = resp.json()

    # Guardar salida
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    n = len(payload.get("predictions", []))
    print(f"[OK] {n} predicciones -> {args.out}")
    # Muestra breve en consola
    print(json.dumps(payload, indent=2)[:500])

if __name__ == "__main__":
    main()
