"""Logica compartida por scripts, notebook y guia.

El modulo mantiene separados los pasos de lectura, particion, entrenamiento y
evaluacion para que cada resultado pueda comprobarse de forma independiente.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FEATURES = [
    "edad",
    "presion_sistolica",
    "colesterol",
    "fumador",
    "actividad_horas_semana",
    "antecedente_familiar",
]
TARGET = "riesgo_alto"
RANDOM_STATE = 42


def load_data(path: str | Path) -> pd.DataFrame:
    """Lee el CSV y valida las columnas necesarias."""
    frame = pd.read_csv(path)
    required = {"id", *FEATURES, TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("El ejemplo introductorio no admite valores ausentes")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("La variable objetivo debe ser binaria (0/1)")
    return frame


def split_data(frame: pd.DataFrame):
    """Crea una particion estratificada 75/25 y conserva indices originales."""
    x = frame[FEATURES]
    y = frame[TARGET]
    return train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_model(x_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Ajusta un arbol pequeno, elegido para ser interpretable."""
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    return model.fit(x_train, y_train)


def evaluate_model(
    model: DecisionTreeClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calcula metricas solo sobre los casos reservados para prueba."""
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }
    return metrics, prediction, probability


def gini_impurity(labels: Iterable[int]) -> float:
    """Calcula Gini = 1 - sum(p_k^2) para una secuencia de clases."""
    values = np.asarray(list(labels))
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 - np.square(probabilities).sum())


def weighted_gini(left: Iterable[int], right: Iterable[int]) -> float:
    """Combina las impurezas de dos hijos segun su numero de observaciones."""
    left_values = list(left)
    right_values = list(right)
    total = len(left_values) + len(right_values)
    if total == 0:
        return 0.0
    return (
        len(left_values) * gini_impurity(left_values)
        + len(right_values) * gini_impurity(right_values)
    ) / total


def _style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def create_outputs(
    frame: pd.DataFrame,
    figures_dir: str | Path,
    results_dir: str | Path,
    generated_dir: str | Path,
) -> dict[str, float]:
    """Ejecuta el analisis y guarda todos los artefactos derivados."""
    figures_path = Path(figures_dir)
    results_path = Path(results_dir)
    generated_path = Path(generated_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    generated_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = split_data(frame)
    model = fit_model(x_train, y_train)
    metrics, prediction, probability = evaluate_model(model, x_test, y_test)
    baseline = float(max(y_test.mean(), 1.0 - y_test.mean()))

    metadata = {
        **metrics,
        "baseline_accuracy": baseline,
        "n_total": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "tree_depth": int(model.get_depth()),
        "tree_leaves": int(model.get_n_leaves()),
        "random_state": RANDOM_STATE,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    (results_path / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions = x_test.copy()
    predictions.insert(0, "id", frame.loc[x_test.index, "id"])
    predictions["observado"] = y_test
    predictions["predicho"] = prediction
    predictions["probabilidad_riesgo_alto"] = probability
    predictions.sort_values("id").to_csv(
        results_path / "predicciones_ejemplo.csv", index=False
    )

    matrix = confusion_matrix(y_test, prediction)
    pd.DataFrame(
        matrix,
        index=["observado_0", "observado_1"],
        columns=["predicho_0", "predicho_1"],
    ).to_csv(results_path / "confusion_matrix.csv")
    (results_path / "tree_rules.txt").write_text(
        export_text(model, feature_names=FEATURES), encoding="utf-8"
    )

    colors = ["#246A73", "#F08A5D"]
    counts = frame[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(["Riesgo bajo (0)", "Riesgo alto (1)"], counts.values, color=colors)
    for index, value in enumerate(counts.values):
        ax.text(index, value + 5, str(value), ha="center", fontweight="bold")
    _style_axis(ax, "Balance de la variable objetivo", ylabel="Numero de casos")
    fig.tight_layout()
    fig.savefig(figures_path / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7.5))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["bajo", "alto"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=2,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Arbol entrenado: profundidad maxima = 3", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_path / "decision_tree.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["bajo", "alto"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion - conjunto de prueba", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_path / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(false_positive_rate, true_positive_rate, color=colors[0], linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    _style_axis(
        ax,
        f"Curva ROC (AUC = {metrics['roc_auc']:.3f})",
        "Tasa de falsos positivos",
        "Tasa de verdaderos positivos",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figures_path / "roc_curve.png", dpi=180)
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.barh(importance.index, importance.values, color=colors[0])
    _style_axis(ax, "Importancia de variables en este arbol", "Importancia", "")
    fig.tight_layout()
    fig.savefig(figures_path / "feature_importance.png", dpi=180)
    plt.close(fig)

    depths = range(1, 11)
    train_scores: list[float] = []
    test_scores: list[float] = []
    for depth in depths:
        candidate = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        candidate.fit(x_train, y_train)
        train_scores.append(candidate.score(x_train, y_train))
        test_scores.append(candidate.score(x_test, y_test))
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(list(depths), train_scores, marker="o", label="Entrenamiento", color=colors[0])
    ax.plot(list(depths), test_scores, marker="o", label="Prueba", color=colors[1])
    _style_axis(ax, "Profundidad y capacidad de generalizacion", "Profundidad maxima", "Exactitud")
    ax.set_xticks(list(depths))
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_path / "depth_curve.png", dpi=180)
    plt.close(fig)

    tex = f"""% Archivo generado automaticamente por scripts/run_analysis.py
\\newcommand{{\\NTotal}}{{{metadata['n_total']}}}
\\newcommand{{\\NTrain}}{{{metadata['n_train']}}}
\\newcommand{{\\NTest}}{{{metadata['n_test']}}}
\\newcommand{{\\ModelAccuracy}}{{{metrics['accuracy']:.3f}}}
\\newcommand{{\\ModelPrecision}}{{{metrics['precision']:.3f}}}
\\newcommand{{\\ModelRecall}}{{{metrics['recall']:.3f}}}
\\newcommand{{\\ModelFOne}}{{{metrics['f1']:.3f}}}
\\newcommand{{\\ModelAUC}}{{{metrics['roc_auc']:.3f}}}
\\newcommand{{\\BaselineAccuracy}}{{{baseline:.3f}}}
\\newcommand{{\\TreeDepth}}{{{metadata['tree_depth']}}}
\\newcommand{{\\TreeLeaves}}{{{metadata['tree_leaves']}}}
\\newcommand{{\\TrueNegative}}{{{int(matrix[0, 0])}}}
\\newcommand{{\\FalsePositive}}{{{int(matrix[0, 1])}}}
\\newcommand{{\\FalseNegative}}{{{int(matrix[1, 0])}}}
\\newcommand{{\\TruePositive}}{{{int(matrix[1, 1])}}}
"""
    (generated_path / "results.tex").write_text(tex, encoding="utf-8")

    example_rows = predictions.sort_values("id").head(6)
    rows = []
    for _, row in example_rows.iterrows():
        rows.append(
            f"{int(row['id'])} & {int(row['edad'])} & "
            f"{row['presion_sistolica']:.1f} & {int(row['fumador'])} & "
            f"{int(row['observado'])} & {int(row['predicho'])} & "
            f"{row['probabilidad_riesgo_alto']:.3f} \\\\"
        )
    (generated_path / "example_predictions.tex").write_text(
        "% Archivo generado automaticamente\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n",
        encoding="utf-8",
    )
    return metadata
