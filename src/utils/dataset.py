import os 
from pathlib import Path
from typing import TypedDict
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

class Sample (TypedDict):
    image : str 
    masks : list[str]

def load_dataset(dataset_dir: str) -> list[Sample]:
    """Cargando el dataset de imagenes y mascaras
    Cada carpeta tiene que tener la estructura : 
     - image/{id}.png -> una sola imagen
     - masks/*.png
    
    Regresa una lista de la direccion de la imagen y su mascara"""

    dataset_path = Path(dataset_dir)
    samples : list[Sample] = []

    for sample_dir in sorted(dataset_path.iterdir()):
        if not sample_dir.is_dir():
            continue

        images_dir = sample_dir / "images"
        masks_dir = sample_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            continue

        image_files = sorted(images_dir.glob("*.png"))
        if not image_files:
            continue

        masks_files = sorted(masks_dir.glob("*.png"))

        samples.append({
            "image": str(image_files[0]),
            "masks": [str(m) for m in masks_files]
        })
    
    return samples


def merge_masks(masks_paths: list[str]) -> np.ndarray:
    """ Combinar multiples instancias de mascaras en una sola"""
    combined = None
    for path in masks_paths:
        mask = np.array(Image.open(path).convert("L")) > 0
        if combined is None:
            combined = mask
        else:
            combined = combined | mask
    
    return combined.astype(np.uint8) * 255


def visualize_samples(samples: list[Sample], n: int = 5) -> None:
    """ Mostrar n imagenes con sus mascaras"""
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))

    for i in range(n):

        img = Image.open(samples[i]["image"])
        mask = merge_masks(samples[i]["masks"])

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"Mask {i} ({len(samples[i]['masks'])}) instancias")
        axes[i, 1].axis("off")
    
    plt.savefig("samples_preview.png", dpi=100, bbox_inches="tight")


def build_merged_dataset(samples: list[Sample], output_dir: str, target_size: tuple[int, int] = (256, 256)) -> None:
    """Crea un dataset con las mascaras fusionadas y redimensionadas a un tamaño objetivo"""
    output_path = Path(output_dir)
    images_out = output_path / "images"
    masks_out = output_path / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        sample_id = Path(sample["image"]).stem

        img = Image.open(sample["image"]).resize(target_size)
        img.save(images_out / f"{sample_id}.png")

        merged = merge_masks(sample["masks"])
        mask_img = Image.fromarray(merged).resize(target_size, resample=Image.NEAREST)
        mask_img.save(masks_out / f"{sample_id}.png")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)} procesados...")

    print(f"Dataset guardado en: {output_path.resolve()}")



if __name__ == "__main__":
    DATASET_DIR = "dataset"
    samples = load_dataset(DATASET_DIR)

    # Antes de merge de mascaras
    print(f"Total de samples: {len(samples)}")
    print(f"Sample 55 - Imagen: {samples[55]['image']}")
    print(f"Sample 55 - Mascaras: {len(samples[55]['masks'])}")

    # Despues de merge de imagens
    mask = merge_masks(samples[0]['masks'])
    print(mask.shape)

    # Visualizar ejemplo
    visualize_samples(samples, n=5)

    # Creacion del dataset fusionado
    DATASET_MERGED_DIR = "dataset_fusionado"
    build_merged_dataset(samples, DATASET_MERGED_DIR)

    print(f"\nDataset fusionado creado en: {DATASET_MERGED_DIR}")
