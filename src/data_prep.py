# src/data_prep.py
# Preprocesamiento genérico para Telco Churn (robusto al CSV de IBM).
# Uso:
#   python src\data_prep.py --input data\raw\telco_churn.csv --outdir data\processed --seed 42 --test-size 0.2 --val-size 0.1
import argparse, os, json, re
import numpy as np
import pandas as pd

def canonicalize_cols(cols):
    out = []
    for c in cols:
        cc = re.sub(r"\s+", "_", c.strip())
        cc = cc.replace("-", "_").replace("/", "_").lower()
        out.append(cc)
    return out

def to_numeric_if_possible(s: pd.Series):
    try:
        s2 = pd.to_numeric(s, errors="coerce")
        # si convirtió al menos algunos valores, usamos s2
        if s2.notna().sum() > 0 and (s2.notna().sum() >= s.notna().sum() * 0.5):
            return s2
        return s
    except Exception:
        return s

def stratified_split(df, target, test_size=0.2, val_size=0.1, seed=42):
    rng = np.random.RandomState(seed)
    parts = {"train": [], "valid": [], "test": []}
    for _, sub in df.groupby(target):
        sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(sub)
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))
        test = sub.iloc[:n_test]
        valid = sub.iloc[n_test:n_test+n_val]
        train = sub.iloc[n_test+n_val:]
        parts["test"].append(test)
        parts["valid"].append(valid)
        parts["train"].append(train)
    train = pd.concat(parts["train"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid = pd.concat(parts["valid"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test  = pd.concat(parts["test"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, valid, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    # Normalizar nombres de columnas
    df.columns = canonicalize_cols(df.columns)

    # Detectar/normalizar target 'churn'
    target_col = None
    for c in df.columns:
        if c.lower() == "churn":
            target_col = c
            break
    if target_col is None:
        raise ValueError("No se encontró la columna objetivo 'Churn' (insensible a mayúsculas).")

    # Mapear Yes/No a 1/0 si aplica
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.lower().map({"yes": 1, "no": 0})
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # Quitar IDs (customerID, customer_id, etc.)
    id_like = [c for c in df.columns if re.search(r"(customer.*id$)|(^id$)|(_id$)", c)]
    X = df.drop(columns=id_like)

    # Convertir campos numéricos guardados como texto (ej. totalcharges)
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].str.strip()
            X[c] = to_numeric_if_possible(X[c])

    # Imputación simple numérica
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # One-hot a categóricas
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Ensamblar dataset final con target
    y = X[target_col]
    if target_col not in X.columns:
        y = df[target_col]
    X = X.drop(columns=[target_col], errors="ignore")
    data = pd.concat([X, y.rename("churn")], axis=1)

    # Splits estratificados por churn
    train, valid, test = stratified_split(
        data, target="churn",
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    # Guardar
    train.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    valid.to_csv(os.path.join(args.outdir, "valid.csv"), index=False)
    test.to_csv(os.path.join(args.outdir, "test.csv"), index=False)

    meta = {
        "n_rows_raw": int(len(df)),
        "n_cols_raw": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "features": X.columns.tolist(),
        "splits": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "target": "churn",
        "seed": args.seed, "test_size": args.test_size, "val_size": args.val_size
    }
    with open(os.path.join(args.outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Procesado. train={len(train)} valid={len(valid)} test={len(test)} | n_features={X.shape[1]}")

if __name__ == "__main__":
    main()
