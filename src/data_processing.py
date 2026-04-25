"""

Dataset folder structure 
    medical_mnist/
        AbdomenCT/    → class 0
        BreastMRI/    → class 1
        ChestCT/      → class 2
        CXR/          → class 3
        Hand/         → class 4
        HeadCT/       → class 5
"""

import pathlib
import random
from typing import Tuple, List
import tensorflow as tf
IMG_SIZE    = 28          
BUFFER_SIZE = 10_000
AUTOTUNE    = tf.data.AUTOTUNE



# Private helpers
def _parse_image(file_path: tf.Tensor, img_size: int) -> tf.Tensor:
    """Read, decode, resize, and normalise a single image to [0, 1]."""
    raw   = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=1)           # grayscale
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _build_dataset(
    file_paths: List[str],
    img_size:   int,
    batch_size: int,
    shuffle:    bool,
) -> tf.data.Dataset:
    """Return a batched, prefetched tf.data.Dataset (images only — no labels)."""
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=BUFFER_SIZE, seed=42)
    ds = ds.map(lambda fp: _parse_image(fp, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# Public API
def load_medical_mnist(
    data_dir:   str,
    img_size:   int   = IMG_SIZE,
    batch_size: int   = 128,
    val_split:  float = 0.10,
    test_split: float = 0.10,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    data_dir    = pathlib.Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    # Collect every image path
    exts      = {".jpg", ".jpeg", ".png"}
    all_paths = [
        str(p)
        for p in data_dir.rglob("*")
        if p.suffix.lower() in exts
    ]

    if not all_paths:
        raise FileNotFoundError(
            f"No images found under '{data_dir}'. "
            "Check that Medical MNIST is extracted correctly"
        )

    print(f"[Data] {len(all_paths):,} images | {len(class_names)} classes: {class_names}")

    # Deterministic shuffle before splitting
    random.seed(42)
    random.shuffle(all_paths)

    n        = len(all_paths)
    n_test   = int(n * test_split)
    n_val    = int(n * val_split)
    n_train  = n - n_test - n_val

    train_paths = all_paths[:n_train]
    val_paths   = all_paths[n_train : n_train + n_val]
    test_paths  = all_paths[n_train + n_val :]

    print(f"[Data] Train: {len(train_paths):,} | Val: {len(val_paths):,} | Test: {len(test_paths):,}")

    train_ds = _build_dataset(train_paths, img_size, batch_size, shuffle=True)
    val_ds   = _build_dataset(val_paths,   img_size, batch_size, shuffle=False)
    test_ds  = _build_dataset(test_paths,  img_size, batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, class_names


def load_with_labels(
    data_dir:   str,
    img_size:   int = IMG_SIZE,
    batch_size: int = 128,
) -> Tuple[tf.data.Dataset, List[str]]:
    data_dir    = pathlib.Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_idx   = {name: i for i, name in enumerate(class_names)}

    exts        = {".jpg", ".jpeg", ".png"}
    all_paths   = []
    all_labels  = []

    for cls_name, idx in class_idx.items():
        for p in (data_dir / cls_name).iterdir():
            if p.suffix.lower() in exts:
                all_paths.append(str(p))
                all_labels.append(idx)

    path_ds  = tf.data.Dataset.from_tensor_slices(all_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
    image_ds = path_ds.map(
        lambda fp: _parse_image(fp, img_size),
        num_parallel_calls=AUTOTUNE,
    )

    ds = (
        tf.data.Dataset.zip((image_ds, label_ds))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds, class_names
