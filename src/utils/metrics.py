import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-7) -> dict[str, float]:
    """
    Calcula las métricas clave para evaluar la segmentación binaria.

    y_true: numpy array con la máscara real (valores en [0, 1] o [0, 255])
    y_pred: numpy array con las predicciones (probabilidades continuas en [0, 1])
    threshold: umbral para binarizar la predicción
    epsilon: valor pequeño para evitar división por cero
    """
    # Normalizar y binarizar
    true_bin = (y_true > 0.5).astype(bool)
    pred_bin = (y_pred > threshold).astype(bool)

    # Calcular componentes de la matriz de confusión a nivel de píxel
    tp = np.sum(true_bin & pred_bin)
    fp = np.sum(~true_bin & pred_bin)
    fn = np.sum(true_bin & ~pred_bin)
    tn = np.sum(~true_bin & ~pred_bin)

    # Fórmulas de métricas
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity)
    }


def plot_learning_curves(history_data: dict | str | Path, output_dir: str | Path) -> None:
    """
    Lee las métricas de entrenamiento (como dict o desde un archivo JSON)
    y genera gráficos de comparación de Loss y Accuracy guardándolos en output_dir.
    """
    if isinstance(history_data, (str, Path)):
        with open(history_data, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = history_data

    epochs = range(1, len(history["loss"]) + 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(1, 2, figsize=(16, 5))

    # Loss Plot
    axis[0].plot(epochs, history["loss"], color="red", marker="o", label="Train Loss")
    if "val_loss" in history:
        axis[0].plot(epochs, history["val_loss"], color="blue", marker="s", label="Val Loss")
    axis[0].set_title("Loss Comparison")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Loss")
    axis[0].legend()
    axis[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy Plot
    axis[1].plot(epochs, history["accuracy"], color="red", marker="o", label="Train Accuracy")
    if "val_accuracy" in history:
        axis[1].plot(epochs, history["val_accuracy"], color="blue", marker="s", label="Val Accuracy")
    axis[1].set_title("Accuracy Comparison")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Accuracy")
    axis[1].legend()
    axis[1].grid(True, linestyle="--", alpha=0.6)

    plot_path = output_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curvas de aprendizaje guardadas en: {plot_path}")
