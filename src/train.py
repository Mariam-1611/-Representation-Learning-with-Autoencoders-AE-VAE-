import tensorflow as tf
from .model import Autoencoder, VAE

# Shared callbacks factory
def _make_callbacks(monitor: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# AE
def train_ae(
    train_ds:   tf.data.Dataset,
    val_ds:     tf.data.Dataset,
    latent_dim: int   = 16,
    epochs:     int   = 30,
    lr:         float = 1e-3,
) -> tuple:
    model = Autoencoder(latent_dim=latent_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    print(f"\n{'='*55}")
    print(f"  Training AE  |  latent_dim={latent_dim}  |  max_epochs={epochs}")
    print(f"{'='*55}\n")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=_make_callbacks("val_reconstruction_loss"),
    )
    return model, history


# VAE
def train_vae(
    train_ds:   tf.data.Dataset,
    val_ds:     tf.data.Dataset,
    latent_dim: int   = 16,
    beta:       float = 1.0,
    epochs:     int   = 30,
    lr:         float = 1e-3,
) -> tuple:
    model = VAE(latent_dim=latent_dim, beta=beta)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    print(f"\n{'='*55}")
    print(f"  Training VAE  |  latent_dim={latent_dim}  |  beta={beta}  |  max_epochs={epochs}")
    print(f"{'='*55}\n")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=_make_callbacks("val_total_loss"),
    )
    return model, history
