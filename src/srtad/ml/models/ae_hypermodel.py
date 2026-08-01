"""
Keras Tuner HyperModel for the convolutional autoencoder.

FIX 2: input shape changed from (6*H, W, 1) to (H, W, 6).
The six panels are now treated as channels, so convolutions see them
jointly at each spatial location. This lets the model learn ON/OFF
contrast directly instead of having to recover it from a stacked image.

Searched hyperparameters:
    n_conv_layers : {1, 2}
    filters_1     : {16, 32}
    filters_2     : {32, 64}   (only used when n_conv_layers == 2)
    latent_dim    : {32, 64, 128}
    learning_rate : {1e-3, 5e-4, 1e-4}

NOTE: unchanged from the previous version. Included here only because it
is a dependency of the rewritten autoencoder.py — no fix was needed on
this file, the memory/speed problem was entirely in data collection and
training-loop plumbing in autoencoder.py.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


class AEHyperModel(kt.HyperModel):
    """
    Builds a convolutional autoencoder for a given set of hyperparameters.

    Parameters
    ----------
    h : int
        Height of a single panel (NOT total — fix 2).
    w : int
        Width of a single panel (frequency bins).
    n_panels : int
        Number of panels stacked as channels (default 6).
    random_state : int
        Random seed for TensorFlow.
    """

    def __init__(self, h: int, w: int, n_panels: int = 6, random_state: int = 42) -> None:
        self.h            = h
        self.w            = w
        self.n_panels     = n_panels
        self.random_state = random_state

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        tf.random.set_seed(self.random_state)

        n_conv     = hp.Int("n_conv_layers", min_value=1, max_value=2, step=1)
        filters_1  = hp.Choice("filters_1",  values=[16, 32])
        filters_2  = hp.Choice("filters_2",  values=[32, 64])
        latent_dim = hp.Choice("latent_dim", values=[32, 64, 128])
        lr         = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])

        # Spatial dims after each MaxPooling2D(2x2) — needed for decoder reshape
        h2, w2 = self.h // 2, self.w // 2
        h4, w4 = self.h // 4, self.w // 4

        # Encoder — input has 6 channels (one per panel)
        inputs = keras.Input(shape=(self.h, self.w, self.n_panels), name="encoder_input")
        x = layers.Conv2D(filters_1, (3, 3), padding="same", activation="relu", name="enc_conv1")(inputs)
        x = layers.MaxPooling2D((2, 2), name="enc_pool1")(x)

        if n_conv == 2:
            x = layers.Conv2D(filters_2, (3, 3), padding="same", activation="relu", name="enc_conv2")(x)
            x = layers.MaxPooling2D((2, 2), name="enc_pool2")(x)
            flat_filters, flat_h, flat_w = filters_2, h4, w4
        else:
            flat_filters, flat_h, flat_w = filters_1, h2, w2

        flat_dim = flat_filters * flat_h * flat_w
        x      = layers.Flatten(name="flatten")(x)
        latent = layers.Dense(latent_dim, name="latent")(x)

        # Decoder
        x = layers.Dense(flat_dim, activation="relu", name="dec_dense")(latent)
        x = layers.Reshape((flat_h, flat_w, flat_filters), name="dec_reshape")(x)

        if n_conv == 2:
            x = layers.Conv2DTranspose(filters_2, (2, 2), strides=2, padding="valid", activation="relu", name="dec_deconv2")(x)
            x = layers.Conv2DTranspose(filters_1, (2, 2), strides=2, padding="valid", activation="relu", name="dec_deconv1")(x)
        else:
            x = layers.Conv2DTranspose(filters_1, (2, 2), strides=2, padding="valid", activation="relu", name="dec_deconv1")(x)

        # Output has 6 channels (reconstruct all panels)
        outputs = layers.Conv2D(self.n_panels, (3, 3), padding="same",
                                activation="sigmoid", name="decoder_output")(x)

        model = keras.Model(inputs, outputs, name="conv_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return model