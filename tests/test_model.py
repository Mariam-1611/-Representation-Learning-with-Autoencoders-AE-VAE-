"""
test_model.py
=============
Unit tests for src/model.py (AE & VAE).
Run with:  pytest tests/test_model.py -v
"""

import numpy as np
import tensorflow as tf
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import Encoder, Decoder, Autoencoder, VAEEncoder, Sampling, VAE

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
BATCH      = 8
IMG_SIZE   = 28
LATENT_DIM = 16
DUMMY_IMG  = tf.random.uniform((BATCH, IMG_SIZE, IMG_SIZE, 1))


def _make_ds(batch_size: int = BATCH) -> tf.data.Dataset:
    """
    Build a minimal tf.data.Dataset that mimics real training data.
    from_tensor_slices splits (N, H, W, C) into N individual images,
    then batch() groups them back — no double-batching.
    """
    images = tf.random.uniform((16, IMG_SIZE, IMG_SIZE, 1))
    return (
        tf.data.Dataset.from_tensor_slices(images)   # 16 items of (28,28,1)
        .batch(batch_size)                            # batches of (8,28,28,1)
        .repeat(2)
    )


# ================================================================== #
# AE tests
# ================================================================== #
class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(latent_dim=LATENT_DIM)
        out = enc(DUMMY_IMG)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_output_dtype(self):
        enc = Encoder(latent_dim=LATENT_DIM)
        out = enc(DUMMY_IMG)
        assert out.dtype == tf.float32


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder()
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = dec(z)
        assert out.shape == (BATCH, IMG_SIZE, IMG_SIZE, 1)

    def test_output_range(self):
        dec = Decoder()
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = dec(z)
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0


class TestAutoencoder:
    def test_forward_shape(self):
        ae  = Autoencoder(latent_dim=LATENT_DIM)
        out = ae(DUMMY_IMG)
        assert out.shape == DUMMY_IMG.shape

    def test_reconstruction_range(self):
        ae  = Autoencoder(latent_dim=LATENT_DIM)
        out = ae(DUMMY_IMG)
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0

    def test_train_step_returns_loss(self):
        ae = Autoencoder(latent_dim=LATENT_DIM)
        ae.compile(optimizer="adam")
        history = ae.fit(_make_ds(), epochs=1, verbose=0)
        assert "reconstruction_loss" in history.history
        assert history.history["reconstruction_loss"][0] > 0


# ================================================================== #
# VAE tests
# ================================================================== #
class TestSampling:
    def test_output_shape(self):
        z_mean    = tf.zeros((BATCH, LATENT_DIM))
        z_log_var = tf.zeros((BATCH, LATENT_DIM))
        z         = Sampling()((z_mean, z_log_var))
        assert z.shape == (BATCH, LATENT_DIM)

    def test_stochastic(self):
        z_mean    = tf.zeros((BATCH, LATENT_DIM))
        z_log_var = tf.zeros((BATCH, LATENT_DIM))
        z1 = Sampling()((z_mean, z_log_var)).numpy()
        z2 = Sampling()((z_mean, z_log_var)).numpy()
        assert not np.allclose(z1, z2), "Sampling layer should be stochastic"


class TestVAEEncoder:
    def test_output_shapes(self):
        enc                  = VAEEncoder(latent_dim=LATENT_DIM)
        z_mean, z_log_var, z = enc(DUMMY_IMG)
        assert z_mean.shape    == (BATCH, LATENT_DIM)
        assert z_log_var.shape == (BATCH, LATENT_DIM)
        assert z.shape         == (BATCH, LATENT_DIM)


class TestVAE:
    def test_forward_shape(self):
        vae = VAE(latent_dim=LATENT_DIM)
        out = vae(DUMMY_IMG)
        assert out.shape == DUMMY_IMG.shape

    def test_encode_returns_three_tensors(self):
        vae             = VAE(latent_dim=LATENT_DIM)
        z_mean, z_lv, z = vae.encode(DUMMY_IMG)
        assert z_mean.shape == (BATCH, LATENT_DIM)

    def test_decode_output_shape(self):
        vae = VAE(latent_dim=LATENT_DIM)
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = vae.decode(z)
        assert out.shape == (BATCH, IMG_SIZE, IMG_SIZE, 1)

    def test_output_range(self):
        vae = VAE(latent_dim=LATENT_DIM)
        out = vae(DUMMY_IMG)
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0

    def test_train_step_keys(self):
        vae = VAE(latent_dim=LATENT_DIM, beta=1.0)
        vae.compile(optimizer="adam")
        history = vae.fit(_make_ds(), epochs=1, verbose=0)
        assert "total_loss"          in history.history
        assert "reconstruction_loss" in history.history
        assert "kl_loss"             in history.history
