"""
Rebuilds the trained Keras-3 (.keras) model using the legacy tf_keras (Keras 2)
API so that tensorflowjs's h5->graph_model converter (which loads h5 files via
tf_keras internally) can read the resulting file. Keras 3's InputLayer config
uses `batch_shape`, which tf_keras's loader does not understand
(`Unrecognized keyword arguments: ['batch_shape']`) -- so we don't try to
re-load the saved h5 with tf_keras; we build a fresh tf_keras model with the
identical architecture and copy weights over by layer order.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import layers as k2layers

from kaggle_model import SpatialAttention as SpatialAttentionV3

# --- Load the original Keras-3 trained model to pull out weights ---
model_v3 = tf.keras.models.load_model(
    'kaggle_asl_model.keras', custom_objects={'SpatialAttention': SpatialAttentionV3}
)


@tf_keras.utils.register_keras_serializable(package='Custom')
class SpatialAttentionV2(k2layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = k2layers.Conv2D(1, kernel_size=1, activation='sigmoid')

    def build(self, input_shape):
        self.attn_conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        attention_map = self.attn_conv(inputs)
        return inputs * attention_map, attention_map

    def get_config(self):
        return super().get_config()


def build_v2(img_size, num_classes):
    inputs = k2layers.Input(shape=(img_size, img_size, 1), name='hand_crop')
    x = k2layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = k2layers.GroupNormalization(groups=8)(x)
    x = k2layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = k2layers.GroupNormalization(groups=8)(x)
    x = k2layers.MaxPooling2D()(x)
    x = k2layers.Dropout(0.25)(x)

    x = k2layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = k2layers.GroupNormalization(groups=16)(x)
    x = k2layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = k2layers.GroupNormalization(groups=16)(x)
    x = k2layers.MaxPooling2D()(x)
    x = k2layers.Dropout(0.25)(x)

    x = k2layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = k2layers.GroupNormalization(groups=16)(x)

    x, _attention_map = SpatialAttentionV2(name='spatial_attention')(x)

    x = k2layers.GlobalAveragePooling2D()(x)
    x = k2layers.Dense(128, activation='relu')(x)
    x = k2layers.Dropout(0.4)(x)
    outputs = k2layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return tf_keras.Model(inputs=inputs, outputs=outputs, name='kaggle_asl_cnn_attention')


model_v2 = build_v2(28, 24)

# Sanity check: layer counts/shapes must line up 1:1 in order.
assert len(model_v2.layers) == len(model_v3.layers), (len(model_v2.layers), len(model_v3.layers))
for l2, l3 in zip(model_v2.layers, model_v3.layers):
    w2 = l2.get_weights()
    w3 = l3.get_weights()
    assert len(w2) == len(w3), (l2.name, l3.name, len(w2), len(w3))
    for a, b in zip(w2, w3):
        assert a.shape == b.shape, (l2.name, l3.name, a.shape, b.shape)
    if w3:
        l2.set_weights(w3)

# Verify numerical equivalence on random input before trusting this.
rng = np.random.default_rng(0)
sample = rng.random((4, 28, 28, 1), dtype=np.float32)
out_v3 = model_v3.predict(sample, verbose=0)
out_v2 = model_v2.predict(sample, verbose=0)
max_diff = np.max(np.abs(out_v3 - out_v2))
print('max abs diff between v3 and rebuilt v2 model outputs:', max_diff)
assert max_diff < 1e-4, 'Weight copy mismatch -- outputs diverge!'

tf_keras.models.save_model(model_v2, 'kaggle_asl_model_v2.h5')
print('Saved kaggle_asl_model_v2.h5 (tf_keras/Keras-2 format, verified equivalent)')

# Convert directly from the in-memory model to a TF.js graph model. Using the
# Python API (rather than shelling out to the `tensorflowjs_converter` CLI)
# avoids needing to reload the h5 file with the custom SpatialAttention layer
# registered in a *different* process -- the CLI's internal loader has no way
# to see our register_keras_serializable() call from this script.
from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_keras_model_to_graph_model

OUTPUT_DIR = '../webapp/public/tfjs_model_asl'
convert_keras_model_to_graph_model(model_v2, OUTPUT_DIR)
print('Converted to TF.js graph model at', OUTPUT_DIR)
