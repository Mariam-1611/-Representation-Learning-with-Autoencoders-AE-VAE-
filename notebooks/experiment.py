import sys
import os

# Add project root to path so src/ imports work 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_processing import load_medical_mnist, load_with_labels
from src.train           import train_ae, train_vae


# CONFIG 
DATA_DIR    = os.path.join("data", "raw", "medical_mnist")  
WEIGHTS_DIR = os.path.join("models")                         
BATCH_SIZE  = 128
LATENT_DIM  = 16    
EPOCHS      = 30
LR          = 1e-3
BETA        = 1.0   

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Check TensorFlow
print("=" * 55)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPUs       : {tf.config.list_physical_devices('GPU')}")
print("=" * 55)


#Load Data
train_ds, val_ds, test_ds, class_names = load_medical_mnist(
    data_dir   = DATA_DIR,
    batch_size = BATCH_SIZE,
)
labeled_ds, _ = load_with_labels(DATA_DIR, batch_size=BATCH_SIZE)
print("Classes:", class_names)

#Visualise Sample Images
sample = next(iter(train_ds))
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample[i].numpy().squeeze(), cmap="gray")
    ax.axis("off")
fig.suptitle("Sample Medical MNIST Images", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(WEIGHTS_DIR, "sample_images.png"), dpi=150)
plt.show()

# HELPER FUNCTIONS

def show_reconstructions(model, dataset, n=10, title="", save_name=None):
    """Plot original vs reconstructed images side by side."""
    batch = next(iter(dataset))
    if isinstance(batch, tuple):
        batch = batch[0]
    orig  = batch[:n]
    recon = model(orig, training=False)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for i in range(n):
        axes[0, i].imshow(orig[i].numpy().squeeze(),  cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].numpy().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original",      fontsize=8)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=8)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(WEIGHTS_DIR, save_name), dpi=150)
    plt.show()


def show_latent_space(model, labeled_dataset, class_names, title="", save_name=None):
    """Scatter plot of latent space coloured by class label."""
    z_all, lbl_all = [], []
    for imgs, lbls in labeled_dataset:
        if hasattr(model, "encode"):
            z = model.encode(imgs)[0].numpy()   
        else:
            z = model.encoder(imgs).numpy()      
        z_all.append(z[:, :2])
        lbl_all.append(lbls.numpy())
    z_all   = np.concatenate(z_all)
    lbl_all = np.concatenate(lbl_all)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(z_all[:, 0], z_all[:, 1], c=lbl_all,
                    cmap="tab10", alpha=0.5, s=6)
    cb = plt.colorbar(sc, ax=ax, ticks=range(len(class_names)))
    cb.set_ticklabels(class_names)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(WEIGHTS_DIR, save_name), dpi=150)
    plt.show()


def show_denoising(model, dataset, noise_factor=0.3, n=10, title="", save_name=None):
    """Show original → noisy → reconstructed triplets."""
    batch = next(iter(dataset))
    if isinstance(batch, tuple):
        batch = batch[0]
    orig  = batch[:n]
    noisy = tf.clip_by_value(
        orig + tf.random.normal(tf.shape(orig)) * noise_factor, 0.0, 1.0
    )
    recon = model(noisy, training=False)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    row_labels = ["Original", "Noisy", "Reconstructed"]
    for row, imgs in enumerate([orig, noisy, recon]):
        for col in range(n):
            axes[row, col].imshow(imgs[col].numpy().squeeze(), cmap="gray")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=8)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(WEIGHTS_DIR, save_name), dpi=150)
    plt.show()


def show_generated(vae, n=10, latent_dim=16, save_name=None):
    """Generate new images by sampling z ~ N(0, I)."""
    z   = tf.random.normal((n, latent_dim))
    gen = vae.decode(z).numpy()
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    fig.suptitle("VAE — Generated Samples (z ~ N(0,I))", fontsize=13, fontweight="bold")
    for i in range(n):
        axes[i].imshow(gen[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(WEIGHTS_DIR, save_name), dpi=150)
    plt.show()


def compute_test_mse(model, dataset, n_batches=10):
    """Compute average MSE on the test set."""
    mse_list = []
    for i, batch in enumerate(dataset):
        if i >= n_batches:
            break
        if isinstance(batch, tuple):
            batch = batch[0]
        recon = model(batch, training=False)
        mse_list.append(float(tf.reduce_mean(tf.square(batch - recon))))
    return float(np.mean(mse_list))


# AUTOENCODER (AE)

print("\n" + "=" * 55)
print("  PART 1 — Training Autoencoder (AE)")
print("=" * 55)

ae_model, ae_history = train_ae(
    train_ds   = train_ds,
    val_ds     = val_ds,
    latent_dim = LATENT_DIM,
    epochs     = EPOCHS,
    lr         = LR,
)

# AE loss curve
hist = ae_history.history
plt.figure(figsize=(7, 4))
plt.plot(hist["reconstruction_loss"],     label="Train")
plt.plot(hist["val_reconstruction_loss"], label="Val", linestyle="--")
plt.title("AE — Reconstruction Loss", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(WEIGHTS_DIR, "ae_loss.png"), dpi=150)
plt.show()

# AE visualisations
show_reconstructions(ae_model, test_ds,
                     title="AE — Original vs Reconstructed",
                     save_name="ae_reconstructions.png")

show_latent_space(ae_model, labeled_ds, class_names,
                  title="AE — Latent Space",
                  save_name="ae_latent_space.png")

show_denoising(ae_model, test_ds,
               title="AE — Denoising",
               save_name="ae_denoising.png")

# Save AE weights
ae_model.save_weights(os.path.join(WEIGHTS_DIR, "ae_weights.weights.h5"))
print("AE weights saved.")


#VARIATIONAL AUTOENCODER (VAE)

print("\n" + "=" * 55)
print("  PART 2 — Training Variational Autoencoder (VAE)")
print("=" * 55)

vae_model, vae_history = train_vae(
    train_ds   = train_ds,
    val_ds     = val_ds,
    latent_dim = LATENT_DIM,
    beta       = BETA,
    epochs     = EPOCHS,
    lr         = LR,
)

# VAE loss curves (Total + Reconstruction + KL)
hist = vae_history.history
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("VAE — Training Loss Curves", fontsize=13, fontweight="bold")
for ax, key, label, color in zip(
    axes,
    ["total_loss",  "reconstruction_loss", "kl_loss"],
    ["Total Loss",  "Reconstruction Loss", "KL Divergence"],
    ["#2196F3",     "#4CAF50",             "#FF5722"],
):
    ax.plot(hist[key],              color=color, label="Train")
    ax.plot(hist[f"val_{key}"],     color=color, linestyle="--", label="Val")
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(WEIGHTS_DIR, "vae_loss.png"), dpi=150)
plt.show()

# VAE visualisations
show_reconstructions(vae_model, test_ds,
                     title="VAE — Original vs Reconstructed",
                     save_name="vae_reconstructions.png")

show_latent_space(vae_model, labeled_ds, class_names,
                  title="VAE — Latent Space",
                  save_name="vae_latent_space.png")

show_generated(vae_model, latent_dim=LATENT_DIM,
               save_name="vae_generated.png")

show_denoising(vae_model, test_ds,
               title="VAE — Denoising",
               save_name="vae_denoising.png")

# Save VAE weights
vae_model.save_weights(os.path.join(WEIGHTS_DIR, "vae_weights.weights.h5"))
print("VAE weights saved.")


# AE vs VAE COMPARISON
print("\n" + "=" * 55)
print("  PART 3 — AE vs VAE Comparison")
print("=" * 55)

ae_mse  = compute_test_mse(ae_model,  test_ds)
vae_mse = compute_test_mse(vae_model, test_ds)

print(f"\n{'='*40}")
print(f"{'Model':<10} {'Test MSE':>12}")
print(f"{'-'*40}")
print(f"{'AE':<10} {ae_mse:>12.6f}")
print(f"{'VAE':<10} {vae_mse:>12.6f}")
print(f"{'='*40}\n")

# VAE 2D latent grid (only works when LATENT_DIM == 2)
if LATENT_DIM == 2:
    n      = 15
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    canvas = np.zeros((28 * n, 28 * n))
    for i, yi in enumerate(grid_x[::-1]):
        for j, xi in enumerate(grid_y):
            z   = tf.constant([[xi, yi]], dtype=tf.float32)
            img = vae_model.decode(z).numpy()[0].squeeze()
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.title("VAE Latent Space 2D Grid", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(WEIGHTS_DIR, "vae_latent_grid.png"), dpi=150)
    plt.show()

print("\n All plots saved to the models/ folder.")
print(" Experiment complete.")