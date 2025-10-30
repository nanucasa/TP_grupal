
# Pasa de processed -> features (puede añadir FE ligera; por ahora, passthrough seguro)
import argparse
import os
import pandas as pd

def fe_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    # Lugar para crear features. Dejamos passthrough para mantener compatibilidad.
    # Ejemplo suave (opcional y seguro):
    # if "tenure" in df.columns and "MonthlyCharges" in df.columns:
    #     df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    return df

def run_one(in_path: str, out_path: str, name: str):
    df = pd.read_csv(in_path)
    out = fe_passthrough(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK][FE] {name}: {len(out)} filas x {out.shape[1]} columnas -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--valid-in", required=True)
    ap.add_argument("--test-in",  required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--valid-out", required=True)
    ap.add_argument("--test-out",  required=True)
    args = ap.parse_args()

    run_one(args.train_in, args.train_out, "train")
    run_one(args.valid_in, args.valid_out, "valid")
    run_one(args.test_in,  args.test_out,  "test")

if __name__ == "__main__":
    main()
