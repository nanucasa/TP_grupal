
# Genera datasets enriquecidos con features derivados seguros a partir de CSVs procesados.
# - Aplica sobre columnas numéricas (excluye 'churn').
# - Nuevas columnas: sq_<col>, log1p_<col> (si min>=0), sqrt_<col> (si min>=0).
# - Mantiene columnas originales y deja 'churn' al final si existe.

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def enrich(df: pd.DataFrame, target="churn") -> pd.DataFrame:
    df = df.copy()
    has_target = target in df.columns
    if has_target:
        y = df[target].astype(int)
        X = df.drop(columns=[target])
    else:
        X = df

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Evita explosión si no hay numéricas
    if not num_cols:
        return df

    for col in num_cols:
        col_series = X[col].astype(float)

        # cuadrado (siempre posible)
        new_sq = f"sq_{col}"
        if new_sq not in X.columns:
            X[new_sq] = col_series ** 2

        # log1p y sqrt: sólo si todos los valores >= 0
        if col_series.min() >= 0:
            new_log = f"log1p_{col}"
            if new_log not in X.columns:
                X[new_log] = np.log1p(col_series)

            new_sqrt = f"sqrt_{col}"
            if new_sqrt not in X.columns:
                X[new_sqrt] = np.sqrt(col_series)

    out = X.copy()
    if has_target:
        out[target] = y.values
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--valid-in", required=True)
    ap.add_argument("--test-in",  required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--valid-out", required=True)
    ap.add_argument("--test-out",  required=True)
    ap.add_argument("--target", default="churn")
    args = ap.parse_args()

    Path(Path(args.train_out).parent).mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.train_in)
    va = pd.read_csv(args.valid_in)
    te = pd.read_csv(args.test_in)

    tr_fe = enrich(tr, target=args.target)
    va_fe = enrich(va, target=args.target)
    te_fe = enrich(te, target=args.target)

    tr_fe.to_csv(args.train_out, index=False)
    va_fe.to_csv(args.valid_out, index=False)
    te_fe.to_csv(args.test_out,  index=False)

    print("[OK] feature_eng:",
          f"train_in={args.train_in} -> {args.train_out} ({tr_fe.shape[1]} cols)",
          f"valid_in={args.valid_in} -> {args.valid_out} ({va_fe.shape[1]} cols)",
          f"test_in={args.test_in} -> {args.test_out} ({te_fe.shape[1]} cols)",
          sep="\n")

if __name__ == "__main__":
    main()
