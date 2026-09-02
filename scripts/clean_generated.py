"""Elimina solo artefactos regenerables conocidos, nunca datos fuente."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "docs" / "generated" / "results.tex",
    *list((ROOT / "docs" / "figures").glob("*.png")),
    *list((ROOT / "output" / "results").glob("*")),
    ROOT / "output" / "pdf" / "arboles_decision.pdf",
    ROOT / "output" / "pdf" / "arboles-decision-guia.pdf",
]

for target in TARGETS:
    if target.is_file():
        target.unlink()
        print(f"Eliminado: {target.relative_to(ROOT)}")
