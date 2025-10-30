
import argparse, json, re
import mlflow
from mlflow.tracking.client import MlflowClient

def parse_selection(path):
    with open(path, "r", encoding="utf-8") as f:
        sel = json.load(f)
    champ = sel.get("champion")
    if isinstance(champ, str):
        m = re.match(r"^(?P<name>.+)\sv(?P<ver>\d+)$", champ.strip())
        if not m:
            raise ValueError(f"Formato inesperado de champion: {champ}")
        return m.group("name").strip(), int(m.group("ver"))
    elif isinstance(champ, dict):
        return champ["name"], int(champ["version"])
    else:
        raise ValueError("selection.json no contiene 'champion' válido")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", required=True)
    ap.add_argument("--selection", default="artifacts/selection.json")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--write-out", default="artifacts/aliases_applied.json")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    name, version = parse_selection(args.selection)

    c = MlflowClient()
    # Limpia alias existente y lo apunta al campeón actual
    c.set_registered_model_alias(name=name, alias=args.alias, version=str(version))

    with open(args.write_out, "w", encoding="utf-8") as f:
        json.dump({"name": name, "alias": args.alias, "version": version}, f, ensure_ascii=False, indent=2)

    print(f"[OK] alias '{args.alias}' -> {name} v{version}")

if __name__ == "__main__":
    main()
