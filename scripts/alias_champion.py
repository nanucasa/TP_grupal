
# -*- coding: utf-8 -*-
"""
Aplica un alias (e.g., 'champion') a un Registered Model/version en MLflow
leyendo artifacts/selection.json, tolerando múltiples formatos.
Valida el nombre contra los modelos realmente existentes en el registry.
"""

import argparse
import json
import os
import re
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


def _known_model_names(client: MlflowClient):
    names = {rm.name for rm in client.search_registered_models()}
    return names


def _extract_from_text(text: str, known_names: set[str]):
    # Nombre: el que aparezca entre los conocidos
    name_candidates = [n for n in known_names if n in text]
    name = None
    if len(name_candidates) == 1:
        name = name_candidates[0]
    elif len(name_candidates) > 1:
        # Si hay varios, priorizamos LogReg si existe, luego XGB, luego RF
        for pref in ("TelcoChurn_LogReg", "TelcoChurn_XGB", "TelcoChurn_RF"):
            if pref in name_candidates:
                name = pref
                break
        if not name:
            name = name_candidates[-1]

    # Versión: preferir patrones tipo "v34", "@34", "version 34"
    m = re.search(r'(?:\bv|\@|version[^0-9]{0,10})(\d{1,5})', text, flags=re.IGNORECASE)
    version = int(m.group(1)) if m else None
    if version is None:
        # Fallback: últimos números "pequeños" (evitamos IDs enormes). <= 10000
        nums = [int(n) for n in re.findall(r'\d{1,5}', text)]
        nums = [n for n in nums if 0 < n <= 10000]
        if nums:
            version = nums[-1]

    if name and version:
        return name, version
    return None


def _walk_for_name_version(obj, known_names: set[str]):
    # Busca recursivamente dicts con ('name','version') válidos o strings parseables
    if isinstance(obj, dict):
        if "name" in obj and "version" in obj and str(obj["name"]) in known_names:
            return str(obj["name"]), int(str(obj["version"]).strip())
        # Variantes comunes
        if "model_name" in obj and "model_version" in obj and str(obj["model_name"]) in known_names:
            return str(obj["model_name"]), int(str(obj["model_version"]).strip())

        for k, v in obj.items():
            # Intentar parsear strings tipo "TelcoChurn_LogReg v34"
            if isinstance(v, str):
                parsed = _extract_from_text(v, known_names)
                if parsed:
                    return parsed
            else:
                parsed = _walk_for_name_version(v, known_names)
                if parsed:
                    return parsed

    elif isinstance(obj, list):
        for x in obj:
            parsed = _walk_for_name_version(x, known_names)
            if parsed:
                return parsed

    elif isinstance(obj, str):
        return _extract_from_text(obj, known_names)

    return None


def parse_selection(path: str, client: MlflowClient):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    known = _known_model_names(client)

    # Intentar JSON primero
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None

    if obj is not None:
        parsed = _walk_for_name_version(obj, known)
        if parsed:
            return parsed

    # Fallback: texto plano
    parsed = _extract_from_text(raw, known)
    if parsed:
        return parsed

    # Último recurso: si existe TelcoChurn_LogReg, usar su última versión;
    # si no, el primer modelo conocido y su última versión.
    fallback_name = ("TelcoChurn_LogReg" if "TelcoChurn_LogReg" in known
                     else (sorted(known)[0] if known else None))
    if not fallback_name:
        raise ValueError("No hay modelos registrados en MLflow para aplicar alias.")

    versions = client.search_model_versions(f"name='{fallback_name}'")
    if not versions:
        raise ValueError(f"El modelo '{fallback_name}' no tiene versiones en el registry.")
    last_version = max(int(v.version) for v in versions)
    return fallback_name, last_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--write-out", required=True)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    name, version = parse_selection(args.selection, client)

    # Validación final: que exista esa versión
    client.get_model_version(name, str(version))  # lanza si no existe

    # Aplicar alias
    client.set_registered_model_alias(name=name, alias=args.alias, version=str(version))

    os.makedirs(os.path.dirname(args.write_out), exist_ok=True)
    out = {
        "applied": True,
        "alias": args.alias,
        "name": name,
        "version": int(version),
        "tracking_uri": args.mlflow_uri,
        "selection_file": args.selection,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(args.write_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] alias '{args.alias}' -> {name} v{version} | resumen -> {args.write_out}")


if __name__ == "__main__":
    main()
