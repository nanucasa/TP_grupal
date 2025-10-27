# src/data_prep.py
# Limpieza y generación de features para Telco Churn (sin scikit-learn).
# Requisitos: pandas, numpy (incluidos en tu requirements).
import argparse
import os
import json
import numpy as np
import pandas as pd

def stratified_split(df, target, test_size=0.2, val_size=0.1, seed=42):
    rng = np.random.RandomState(seed)
    parts = {"train": [], "valid": [], "test": []}
    for cls, sub in df.groupby(target):
        sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(sub)
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))
        n_test = min(n_test, n)
        n_val  = min(n_val, max(0, n - n_test))
        test = sub.iloc[:n_test]
        valid = sub.iloc[n_test:n_test + n_val]
        train = sub.iloc[n_test + n_val:]
        # mezclar cada partición para mayor aleatoriedad, pero determinista
        parts["test"].append(test.sample(frac=1.0, random_state=seed))
        parts["valid"].append(valid.sample(frac=1.0, random_state=seed))
        parts["train"].append(train.sample(frac=1.0, random_state=seed))
    return (pd.concat(parts["train"]).sample(frac=1.0, random_state=seed).reset_index(drop=True),
            pd.concat(parts["valid"]).sample(frac=1.0, random_state=seed).reset_index(drop=True),
            pd.concat(parts["test"]).sample(frac=1.0, random_state=seed).reset_index(drop=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta al CSV crudo")
    ap.add_argument("--outdir", required=True, help="Directorio de salida (se creará si no existe)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    # Validaciones mínimas
    req_cols = {
        "customer_id","age","gender","region","contract_type","tenure_months",
        "monthly_charges","total_charges","internet_service","phone_service",
        "multiple_lines","payment_method","churn"
    }
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {sorted(missing)}")

    # Asegurar tipos
    num_cols = ["age","tenure_months","monthly_charges","total_charges"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[num_cols].isna().any().any():
        # si aparece algún NA, imputar con la mediana (no debería según el dataset provisto)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Quitar identificadores de X; target asegurado a int
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn","customer_id"])

    # One-hot para categóricas
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    X[cat_cols] = X[cat_cols].apply(lambda s: s.astype("category"))
    X = pd.get_dummies(X, drop_first=True)

    # Ensamblar de nuevo con target
    data = pd.concat([X, y.rename("churn")], axis=1)

    # Particiones estratificadas por churn (sin scikit-learn)
    train, valid, test = stratified_split(data, target="churn",
                                          test_size=args.test_size,
                                          val_size=args.val_size,
                                          seed=args.seed)

    # Guardar
    train_fp = os.path.join(args.outdir, "train.csv")
    valid_fp = os.path.join(args.outdir, "valid.csv")
    test_fp  = os.path.join(args.outdir, "test.csv")
    train.to_csv(train_fp, index=False)
    valid.to_csv(valid_fp, index=False)
    test.to_csv(test_fp, index=False)

    # Metadata de features
    meta = {
        "n_rows_raw": int(len(df)),
        "n_cols_raw": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "features": X.columns.tolist(),
        "splits": {
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test":  int(len(test))
        },
        "target": "churn",
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size
    }
    with open(os.path.join(args.outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Procesado. train={len(train)} valid={len(valid)} test={len(test)} | n_features={X.shape[1]}")

if __name__ == "__main__":
    main()
