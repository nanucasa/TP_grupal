import argparse
import json
import os

import matplotlib.pyplot as plt


def infer_label(path: str) -> str:
    """
    Devuelve un nombre legible a partir del nombre de archivo de métricas.
    """
    name = os.path.basename(path)
    if name == "metrics_test.json":
        return "LogReg baseline"
    if name == "metrics_test_rf.json":
        return "RandomForest"
    if name == "metrics_test_xgb.json":
        return "XGBoost"
    if name == "metrics_test_fe.json":
        return "LogReg FE"
    return name


def load_f1(path: str) -> float:
    """
    Lee un archivo JSON de métricas y devuelve el F1.
    Prioriza la clave 'test', si no está usa 'valid'.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "test" in data and "f1" in data["test"]:
        return data["test"]["f1"]

    if "valid" in data and "f1" in data["valid"]:
        return data["valid"]["f1"]

    raise KeyError(f"No se encontró 'f1' en {path}")


def generate_report(sources, out_md: str, out_fig: str):
    """
    Genera:
    - Un markdown con tabla de F1 en test.
    - Una figura tipo barra con F1 por modelo.
    """
    rows = []
    for src in sources:
        f1 = load_f1(src)
        label = infer_label(src)
        rows.append({"label": label, "f1": f1})

    # Ordenar por F1 descendente (mejor primero)
    rows.sort(key=lambda r: r["f1"], reverse=True)

    # Markdown (UTF-8, sin caracteres raros)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Benchmark modelos (F1 test)\n\n")
        f.write("| Modelo | F1 test |\n")
        f.write("|--------|---------|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['f1']:.4f} |\n")

        best = rows[0]
        f.write(
            f"\nModelo ganador (mayor F1): **{best['label']}** "
            f"con F1 = {best['f1']:.4f}.\n"
        )

    # Figura de barras
    labels = [r["label"] for r in rows]
    scores = [r["f1"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores)
    plt.ylabel("F1 test")
    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Genera benchmark de modelos a partir de métricas en JSON."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Rutas a archivos de métricas JSON (test).",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        help="Ruta de salida para el reporte en Markdown.",
    )
    parser.add_argument(
        "--out-fig",
        required=True,
        help="Ruta de salida para la figura (PNG).",
    )

    args = parser.parse_args()

    rows = generate_report(args.sources, args.out_md, args.out_fig)

    # Mensaje simple para el log
    best = rows[0]
    print(
        f"[OK] Reporte generado: {args.out_md} | {args.out_fig} "
        f"| mejor modelo = {best['label']} (F1={best['f1']:.4f})"
    )


if __name__ == "__main__":
    main()
