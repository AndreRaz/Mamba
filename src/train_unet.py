import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path


IMAGE_SIZE = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net segmentation model.")
    parser.add_argument("--model", choices=("unet", "mamba"), default="unet", help="Model variant to train.")
    parser.add_argument("--images-dir", default="dataset_fusionado/images")
    parser.add_argument("--masks-dir", default="dataset_fusionado/masks")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--filters", type=int, default=32, help="Base number of U-Net filters.")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/unet", help="Directory where the trained model and metrics are saved.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device preference. 'auto' uses CUDA GPUs when TensorFlow can see them.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="CUDA GPU index to expose before TensorFlow initializes. Applies to auto/gpu modes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check dataset pairing without importing TensorFlow or training.")
    args = parser.parse_args()

    if args.gpu_index is not None and args.gpu_index < 0:
        parser.error("--gpu-index must be 0 or greater.")
    if args.filters < 1:
        parser.error("--filters must be 1 or greater.")

    return args


def collect_pairs(images_dir: str, masks_dir: str):
    img_paths = sorted(Path(images_dir).glob("*.png"))
    mask_paths = sorted(Path(masks_dir).glob("*.png"))

    img_by_stem = {path.stem: path for path in img_paths}
    mask_by_stem = {path.stem: path for path in mask_paths}

    missing_masks = sorted(set(img_by_stem) - set(mask_by_stem))
    missing_images = sorted(set(mask_by_stem) - set(img_by_stem))
    if missing_masks or missing_images:
        raise ValueError(
            f"Dataset mismatch: {len(missing_masks)} images without masks, "
            f"{len(missing_images)} masks without images."
        )

    stems = sorted(img_by_stem)
    return [str(img_by_stem[stem]) for stem in stems], [str(mask_by_stem[stem]) for stem in stems]


def estimate_array_memory_gib(sample_count: int) -> float:
    image_values = sample_count * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
    mask_values = sample_count * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1
    return (image_values + mask_values) * 4 / 1024**3


def train_validation_split(image_paths: list[str], mask_paths: list[str], validation_fraction: float = 0.2):
    paired_paths = list(zip(image_paths, mask_paths))
    random.Random(42).shuffle(paired_paths)

    validation_count = max(1, int(len(paired_paths) * validation_fraction))
    validation_pairs = paired_paths[:validation_count]
    train_pairs = paired_paths[validation_count:]

    train_images, train_masks = zip(*train_pairs)
    valid_images, valid_masks = zip(*validation_pairs)

    return list(train_images), list(valid_images), list(train_masks), list(valid_masks)


def configure_cuda_visibility(device: str, gpu_index: int | None) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return

    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def configure_tensorflow(device: str, gpu_index: int | None):
    configure_cuda_visibility(device, gpu_index)

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is not installed. Install a CUDA-enabled TensorFlow build to train with GPU support.") from exc

    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        print(f"TensorFlow detected {len(physical_gpus)} CUDA GPU(s):")
        for index, gpu in enumerate(physical_gpus):
            print(f"  GPU {index}: {gpu.name}")

        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                print(f"Could not set memory growth for {gpu.name}: {exc}")

        print("GPU memory growth enabled where supported.")
    else:
        if device == "gpu":
            requested = f" index {gpu_index}" if gpu_index is not None else ""
            print(f"Requested GPU{requested}, but TensorFlow cannot see a CUDA GPU. Falling back to CPU.")
        else:
            print("TensorFlow did not detect a CUDA GPU. Using CPU.")

    print(f"Logical devices: {[device.name for device in tf.config.list_logical_devices()]}")

    return tf


def load_image_mask(image_path, mask_path, tf):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask


def build_dataset(image_paths, mask_paths, batch_size: int, shuffle: bool, tf):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda image, mask: load_image_mask(image, mask, tf), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(model_name: str, filters: int):
    if model_name == "unet":
        from models.unet import build_unet

        return build_unet(input_size=(256, 256, 3), n_filters=filters, n_classes=1)

    from models.mamba import build_mamba_unet

    return build_mamba_unet(input_size=(256, 256, 3), n_filters=filters, n_classes=1)


def evaluate_segmentation_metrics(model, dataset, steps: int | None):
    import numpy as np
    from utils.metrics import compute_segmentation_metrics

    y_true_batches = []
    y_pred_batches = []
    for step_index, (images, masks) in enumerate(dataset):
        if steps is not None and step_index >= steps:
            break
        y_true_batches.append(masks.numpy())
        y_pred_batches.append(model.predict(images, verbose=0))

    if not y_true_batches:
        return {}

    return compute_segmentation_metrics(np.concatenate(y_true_batches), np.concatenate(y_pred_batches))


def main():
    args = parse_args()
    image_paths, mask_paths = collect_pairs(args.images_dir, args.masks_dir)

    if not image_paths:
        raise ValueError("No PNG image/mask pairs found.")

    print(f"Dataset pairs: {len(image_paths)}")
    print(f"Estimated RAM if loaded as NumPy arrays: {estimate_array_memory_gib(len(image_paths)):.2f} GiB")

    if args.dry_run:
        print("Dry run completed without importing TensorFlow or starting training.")
        return

    if len(image_paths) < 2:
        raise ValueError("At least 2 image/mask pairs are required for train/validation split.")

    tf = configure_tensorflow(args.device, args.gpu_index)

    train_images, valid_images, train_masks, valid_masks = train_validation_split(image_paths, mask_paths)

    train_ds = build_dataset(train_images, train_masks, args.batch_size, shuffle=True, tf=tf)
    valid_ds = build_dataset(valid_images, valid_masks, args.batch_size, shuffle=False, tf=tf)

    model = build_model(args.model, args.filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    started_at = time.time()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=valid_ds,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        callbacks=[early_stopping],
    )

    duration_seconds = time.time() - started_at
    segmentation_metrics = evaluate_segmentation_metrics(model, valid_ds, args.validation_steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.keras"
    history_path = output_dir / "history.json"
    metrics_path = output_dir / "metrics.json"

    model.save(model_path)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "parameter_count": model.count_params(),
        "duration_seconds": round(duration_seconds, 2),
        "dataset_pairs": len(image_paths),
        "train_pairs": len(train_images),
        "validation_pairs": len(valid_images),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "filters": args.filters,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "history": history.history,
        "final_metrics": {name: values[-1] for name, values in history.history.items() if values},
        "segmentation_metrics": segmentation_metrics,
        "model_path": str(model_path),
    }
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    from utils.metrics import plot_learning_curves
    plot_learning_curves(history.history, output_dir)

    print(f"Training duration: {duration_seconds:.2f} seconds")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
