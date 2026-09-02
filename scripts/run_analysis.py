"""Ejecuta el pipeline y crea resultados consumidos por notebook y PDF."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arboles_decision.pipeline import create_outputs, load_data  # noqa: E402


def main() -> None:
    frame = load_data(ROOT / "data" / "raw" / "riesgo_clinico_simulado.csv")
    metrics = create_outputs(
        frame,
        figures_dir=ROOT / "docs" / "figures",
        results_dir=ROOT / "output" / "results",
        generated_dir=ROOT / "docs" / "generated",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
