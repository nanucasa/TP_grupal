
# -*- coding: utf-8 -*-
"""
Lee artifacts/selection.json y fija el alias 'champion' en el Model Registry
apuntando al modelo y versión seleccionados.
Formados soportados de selection.json:
  A) {"champion": {"name": "...", "version": 34, ...}}
  B) {"name": "...", "version": 34, ...}
"""

import argparse
import json
import sys
from mlflow.tracking import MlflowClient

def _extract_name_version(obj):
    # Caso A
    if "champion" in obj and isinstance(obj["champion"], dict):
        champ = obj["champion"]
    else:
        champ = obj

    # tolerante a "name"/"model_name" y version como str/int
    name = champ.get("name") or champ.get("model_name")
    if not name:
        raise KeyError("No encuentro 'name'/'model_name' en selection.json")

    ver = champ.get("version")
    if ver is None:
        # a veces guardan 'v' (e.g., "v34") o 'model_version'
        if "v" in champ:
            v = champ["v"]
            ver = int(str(v).lstrip("vV"))
        elif "model_version" in champ:
            ver = champ["model_version"]
        else:
            raise KeyError("No encuentro 'version' en selection.json")

    return name, int(ver)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="artifacts/selection.json")
    ap.add_argument("--mlflow-uri", default="http://127.0.0.1:5001")
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    with open(args.selection, "r", encoding="utf-8") as f:
        data = json.load(f)

    name, version = _extract_name_version(data)

    client = MlflowClient(tracking_uri=args.mlflow_uri)
    # MLflow 2.x: alias por nombre de modelo registrado
    client.set_registered_model_alias(name, args.alias, str(version))

    print(f"[OK] Alias '{args.alias}' -> {name} v{version}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
