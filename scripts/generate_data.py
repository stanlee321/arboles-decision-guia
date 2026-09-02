"""Genera el conjunto de datos pedagogico con una semilla fija."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "riesgo_clinico_simulado.csv"
SEED = 20260902
N_CASES = 480


def main() -> None:
    rng = np.random.default_rng(SEED)
    age = rng.integers(18, 81, N_CASES)
    systolic = np.clip(rng.normal(126 + 0.22 * (age - 45), 17, N_CASES), 88, 205)
    cholesterol = np.clip(rng.normal(192 + 0.55 * (age - 45), 33, N_CASES), 115, 355)
    smoker = rng.binomial(1, 0.24, N_CASES)
    activity = np.clip(rng.gamma(shape=2.1, scale=1.35, size=N_CASES), 0, 10)
    family_history = rng.binomial(1, 0.31, N_CASES)

    # Relacion deliberadamente sencilla, con ruido, solo para fines didacticos.
    linear_score = (
        -0.45
        + 0.050 * (age - 50)
        + 0.042 * (systolic - 125)
        + 0.016 * (cholesterol - 190)
        + 0.85 * smoker
        + 0.78 * family_history
        - 0.28 * activity
        + rng.normal(0, 0.85, N_CASES)
    )
    probability = 1 / (1 + np.exp(-linear_score))
    high_risk = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "id": np.arange(1, N_CASES + 1),
            "edad": age,
            "presion_sistolica": np.round(systolic, 1),
            "colesterol": np.round(cholesterol, 1),
            "fumador": smoker,
            "actividad_horas_semana": np.round(activity, 1),
            "antecedente_familiar": family_history,
            "riesgo_alto": high_risk,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Datos escritos: {OUTPUT.relative_to(ROOT)} ({len(frame)} filas)")


if __name__ == "__main__":
    main()
