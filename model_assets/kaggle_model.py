"""
kaggle_model.py
----------------
CNN + spatial-attention classifier for the real Kaggle "Sign Language MNIST"
dataset (28x28 grayscale, static ASL alphabet hand poses, A-Y excluding J
which requires motion).

Why this architecture (and not the CNN+LSTM+Attention model in model_arch.py):
Sign Language MNIST is a STATIC image dataset -- one photo per sign, no video
/ temporal information. The paper's reference literature review explicitly
covers this exact scenario too (Section II cites Herdegen 2023: "applying
deep learning models to identification of ASL, using the MNIST dataset...
ResNet50, LeNet, AlexNet, VGG16, DenseNet121"). Running an RNN/LSTM over a
single static frame would be meaningless (nothing changes over "time"), so
here we use a plain CNN for spatial feature extraction -- exactly the paper's
"CNN ... extract spatial characteristics from individual video frames" -- and
add a lightweight spatial-attention block so the model still gets the
"attention mechanism ... discriminate crucial features within the hand
region" called out in Section III.A, just applied over image regions instead
of over time.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class SpatialAttention(layers.Layer):
    """
    Learns a per-pixel importance map over the CNN's feature maps and
    reweights them accordingly, so the classifier focuses on the most
    discriminative parts of the hand shape (finger tips, knuckle bends, etc.)
    rather than background/uniform regions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = layers.Conv2D(1, kernel_size=1, activation="sigmoid")

    def build(self, input_shape):
        self.attn_conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        attention_map = self.attn_conv(inputs)   # (batch, H, W, 1), values in [0,1]
        return inputs * attention_map, attention_map

    def get_config(self):
        return super().get_config()


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
    """
    Note on normalization: an earlier version of this model used
    BatchNormalization, but with this dataset/batch size its running
    mean/variance estimates (used at inference time) diverged badly from the
    true statistics -- verified empirically: >90% accuracy when evaluated
    with batch statistics vs ~8% with BatchNorm's running statistics on the
    exact same weights. GroupNormalization computes its statistics per
    example (independent of batch size, identical at train and inference
    time), which eliminates that train/eval mismatch entirely.
    """
    inputs = layers.Input(shape=(img_size, img_size, 1), name="hand_crop")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GroupNormalization(groups=16)(x)

    x, _attention_map = SpatialAttention(name="spatial_attention")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="kaggle_asl_cnn_attention")
    return model
