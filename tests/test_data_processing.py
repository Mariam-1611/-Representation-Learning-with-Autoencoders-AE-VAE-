import numpy as np
import tensorflow as tf
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import _parse_image, _build_dataset

# Helpers
def _make_dummy_jpeg(tmp_path, filename="img.jpg", size=(64, 64)):
    """Create a dummy grayscale JPEG for testing."""
    import struct, zlib

    # Use TF to encode a random image
    img   = tf.random.uniform((size[0], size[1], 1), 0, 255, dtype=tf.float32)
    img   = tf.cast(img, tf.uint8)
    enc   = tf.image.encode_jpeg(img, format="grayscale")
    fpath = str(tmp_path / filename)
    with open(fpath, "wb") as f:
        f.write(enc.numpy())
    return fpath


# Tests
class TestParseImage:
    def test_output_shape(self, tmp_path):
        fpath  = _make_dummy_jpeg(tmp_path)
        tensor = _parse_image(tf.constant(fpath), img_size=28)
        assert tensor.shape == (28, 28, 1), f"Expected (28,28,1), got {tensor.shape}"

    def test_pixel_range(self, tmp_path):
        fpath  = _make_dummy_jpeg(tmp_path)
        tensor = _parse_image(tf.constant(fpath), img_size=28)
        assert float(tf.reduce_min(tensor)) >= 0.0
        assert float(tf.reduce_max(tensor)) <= 1.0

    def test_dtype(self, tmp_path):
        fpath  = _make_dummy_jpeg(tmp_path)
        tensor = _parse_image(tf.constant(fpath), img_size=28)
        assert tensor.dtype == tf.float32


class TestBuildDataset:
    def test_batch_shape(self, tmp_path):
        paths = [_make_dummy_jpeg(tmp_path, f"img{i}.jpg") for i in range(10)]
        ds    = _build_dataset(paths, img_size=28, batch_size=4, shuffle=False)
        batch = next(iter(ds))
        assert batch.shape == (4, 28, 28, 1)

    def test_num_batches(self, tmp_path):
        paths = [_make_dummy_jpeg(tmp_path, f"img{i}.jpg") for i in range(12)]
        ds    = _build_dataset(paths, img_size=28, batch_size=4, shuffle=False)
        assert sum(1 for _ in ds) == 3   # 12 / 4 = 3 batches

    def test_shuffle_changes_order(self, tmp_path):
        paths   = [_make_dummy_jpeg(tmp_path, f"img{i}.jpg") for i in range(20)]
        ds_shuf = _build_dataset(paths, img_size=28, batch_size=20, shuffle=True)
        ds_ord  = _build_dataset(paths, img_size=28, batch_size=20, shuffle=False)
        b_shuf  = next(iter(ds_shuf)).numpy()
        b_ord   = next(iter(ds_ord)).numpy()
        # Very unlikely to be identical after shuffle
        assert not np.allclose(b_shuf, b_ord), "Shuffled dataset matches ordered — shuffle may not work"
