
import argparse, json, re, os
import pandas as pd
import mlflow

def parse_selection(path):
    with open(path, "r", encoding="utf-8") as f:
        sel = json.load(f)
    champ = sel.get("champion")
    if champ is None:
        raise ValueError("selection.json no contiene 'champion'.")

    # Caso string: "Name v34" o "Name@alias" o "Name"
    if isinstance(champ, str):
        s = champ.strip()
        # models:/Name@alias o models:/Name/34
        if s.lower().startswith("models:/"):
            spec = s.split("models:/", 1)[1]
            if "@" in spec:
                name, alias = spec.split("@", 1)
                return {"name": name.strip(), "version": None, "alias": alias.strip()}
            if "/" in spec:
                name, ver = spec.split("/", 1)
                return {"name": name.strip(), "version": int(ver), "alias": None}
            return {"name": spec.strip(), "version": None, "alias": None}

        # "Name@alias"
        if "@" in s and " v" not in s.lower():
            name, alias = s.split("@", 1)
            return {"name": name.strip(), "version": None, "alias": alias.strip()}

        # "Name v34"
        m = re.match(r"^(?P<name>.+?)\s+v(?P<ver>\d+)$", s, re.I)
        if m:
            return {"name": m.group("name").strip(), "version": int(m.group("ver")), "alias": None}

        # Solo nombre
        return {"name": s, "version": None, "alias": None}

    # Caso dict: claves variadas
    if isinstance(champ, dict):
        lk = {str(k).lower(): v for k, v in champ.items()}
        name = lk.get("name") or lk.get("model") or lk.get("registered_model") or lk.get("model_name")
        version = lk.get("version") or lk.get("model_version") or lk.get("v")
        alias = lk.get("alias")

        # Si vino un URI directo
        uri = lk.get("uri")
        if (not name) and isinstance(uri, str) and uri.lower().startswith("models:/"):
            spec = uri.split("models:/", 1)[1]
            if "@" in spec:
                name, alias = spec.split("@", 1)
                return {"name": name.strip(), "version": None, "alias": alias.strip()}
            if "/" in spec:
                name, ver = spec.split("/", 1)
                return {"name": name.strip(), "version": int(ver), "alias": None}
            return {"name": spec.strip(), "version": None, "alias": None}

        if not name:
            raise ValueError(f"Estructura inesperada en champion dict: {champ}")
        try:
            version = int(version) if version is not None else None
        except Exception:
            version = None
        return {"name": str(name), "version": version, "alias": alias}

    raise ValueError(f"Tipo de 'champion' no soportado: {type(champ)}")

def load_threshold(thr_file, default=0.5):
    if not thr_file:
        return default
    try:
        with open(thr_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("best_threshold", data.get("threshold", default)))
    except Exception:
        return default

def infer_threshold_file(model_name):
    n = (model_name or "").lower()
    candidates = []
    if "xgb" in n:
        candidates.append("artifacts/best_threshold_xgb.json")
    elif "rf" in n or "randomforest" in n:
        candidates.append("artifacts/best_threshold_rf.json")
    # genérico al final
    candidates.append("artifacts/best_threshold.json")
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlflow-uri", required=True)
    ap.add_argument("--selection", default="artifacts/selection.json")
    ap.add_argument("--threshold-file", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)

    sel = parse_selection(args.selection)
    name, version, alias = sel["name"], sel["version"], sel["alias"]

    if alias:
        model_uri = f"models:/{name}@{alias}"
    elif version is not None:
        model_uri = f"models:/{name}/{version}"
    else:
        # fallback razonable si no hay version/alias
        model_uri = f"models:/{name}/Staging"

    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_csv(args.test)
    ycol = "Churn" if "Churn" in df.columns else None
    X = df.drop(columns=[ycol]) if ycol else df

    # Probabilidades si existen
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            try:
                sk = model.unwrap_python_model().model  # para sklearn envuelto
                proba = sk.predict_proba(X)[:, 1]
            except Exception:
                pass
    except Exception:
        pass

    # Umbral
    thr_file = args.threshold_file or infer_threshold_file(name)
    thr = args.threshold if args.threshold is not None else load_threshold(thr_file, 0.5)

    if proba is None:
        pred = model.predict(X)
        out = pd.DataFrame({"pred": pred})
        out.to_csv(args.out, index=False)
        print(f"[OK] preds (sin proba) -> {args.out}")
        return

    pred = (proba >= thr).astype(int)
    out = pd.DataFrame({"prob_1": proba, "pred": pred})
    out.to_csv(args.out, index=False)
    print(f"[OK] preds con umbral={thr:.4f} -> {args.out}")

if __name__ == "__main__":
    main()
