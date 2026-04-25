import tensorflow as tf
from tensorflow.keras import layers, Model

# AUTOENCODER (AE) 

class Encoder(layers.Layer):
    def __init__(self, latent_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        self.conv1   = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2   = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense   = layers.Dense(latent_dim, activation="relu")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dense        = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape      = layers.Reshape((7, 7, 64))
        self.deconv1      = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.deconv2      = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.output_layer = layers.Conv2DTranspose(1,  3, activation="sigmoid", padding="same")

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.output_layer(x)


class Autoencoder(Model):
    def __init__(self, latent_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(latent_dim=latent_dim, name="ae_encoder")
        self.decoder = Decoder(name="ae_decoder")

        self._loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self._loss_tracker]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat = self(x, training=True)
            loss  = tf.reduce_mean(tf.square(x - x_hat))       # MSE

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss)
        return {"reconstruction_loss": self._loss_tracker.result()}

    def test_step(self, data):
        x    = data
        x_hat = self(x, training=False)
        loss  = tf.reduce_mean(tf.square(x - x_hat))
        self._loss_tracker.update_state(loss)
        return {"reconstruction_loss": self._loss_tracker.result()}



# VARIATIONAL AUTOENCODER (VAE)

class Sampling(layers.Layer):
    def call(self, inputs: tuple) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAEEncoder(layers.Layer):
    def __init__(self, latent_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        self.conv1     = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2     = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten   = layers.Flatten()
        self.dense     = layers.Dense(128, activation="relu")
        self.z_mean    = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling  = Sampling(name="z_sampling")

    def call(self, x: tf.Tensor, training: bool = False) -> tuple:
        x         = self.conv1(x)
        x         = self.conv2(x)
        x         = self.flatten(x)
        x         = self.dense(x)
        z_mean    = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z         = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


# VAEDecoder is architecturally identical to Decoder
VAEDecoder = Decoder


class VAE(Model):
    def __init__(self, latent_dim: int = 16, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VAEEncoder(latent_dim=latent_dim, name="vae_encoder")
        self.decoder = VAEDecoder(name="vae_decoder")
        self.beta    = beta

        self._total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self._recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self._kl_loss_tracker    = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._recon_loss_tracker,
            self._kl_loss_tracker,
        ]

    # forward pass
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x: tf.Tensor) -> tuple:
        """Return (z_mean, z_log_var, z) for visualisation."""
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode a latent vector z into an image."""
        return self.decoder(z, training=False)

    # losses 
    @staticmethod
    def _reconstruction_loss(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        """Sum MSE over spatial dims, then average over batch."""
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2, 3])
        )

    @staticmethod
    def _kl_loss(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        """KL( q(z|x) || N(0,I) ) — closed-form solution."""
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

    # train / test steps 
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_hat                = self.decoder(z, training=True)
            recon_loss           = self._reconstruction_loss(x, x_hat)
            kl_loss              = self._kl_loss(z_mean, z_log_var)
            total_loss           = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self._total_loss_tracker.update_state(total_loss)
        self._recon_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss":         self._total_loss_tracker.result(),
            "reconstruction_loss": self._recon_loss_tracker.result(),
            "kl_loss":            self._kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        x_hat                = self.decoder(z, training=False)
        recon_loss           = self._reconstruction_loss(x, x_hat)
        kl_loss              = self._kl_loss(z_mean, z_log_var)
        total_loss           = recon_loss + self.beta * kl_loss

        self._total_loss_tracker.update_state(total_loss)
        self._recon_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss":         self._total_loss_tracker.result(),
            "reconstruction_loss": self._recon_loss_tracker.result(),
            "kl_loss":            self._kl_loss_tracker.result(),
        }
