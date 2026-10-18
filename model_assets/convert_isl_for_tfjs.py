"""
convert_isl_for_tfjs.py
------------------------
Same rebuild-in-tf_keras-then-convert-in-memory approach as
rebuild_for_tfjs.py, applied to the trained ISL model (64x64x3 input, 26
classes, produced by train_isl_model.py). See rebuild_for_tfjs.py for why this
two-step dance is needed on Windows: tensorflowjs's h5-based converter path
reloads the h5 file with the legacy tf_keras (Keras 2) API in a way that can't
see custom layers registered in a different process, so we rebuild the
architecture natively in tf_keras, copy weights over, verify numerical
equivalence, and convert straight from the in-memory model (which sidesteps
that reload entirely).
"""
import json

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import layers as k2layers

from train_isl_model import SpatialAttention as SpatialAttentionV3, IMG_SIZE

model_v3 = tf.keras.models.load_model(
    'kaggle_isl_model.keras', custom_objects={'SpatialAttention': SpatialAttentionV3}
)
with open('kaggle_isl_labels.json') as f:
    labels = json.load(f)
num_classes = len(labels)


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
    inputs = k2layers.Input(shape=(img_size, img_size, 3), name='hand_crop')
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
    x = k2layers.MaxPooling2D()(x)
    x = k2layers.Dropout(0.25)(x)

    x = k2layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = k2layers.GroupNormalization(groups=16)(x)

    x, _attention_map = SpatialAttentionV2(name='spatial_attention')(x)

    x = k2layers.GlobalAveragePooling2D()(x)
    x = k2layers.Dense(256, activation='relu')(x)
    x = k2layers.Dropout(0.4)(x)
    outputs = k2layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return tf_keras.Model(inputs=inputs, outputs=outputs, name='isl_cnn_attention')


model_v2 = build_v2(IMG_SIZE, num_classes)

assert len(model_v2.layers) == len(model_v3.layers), (len(model_v2.layers), len(model_v3.layers))
for l2, l3 in zip(model_v2.layers, model_v3.layers):
    w2 = l2.get_weights()
    w3 = l3.get_weights()
    assert len(w2) == len(w3), (l2.name, l3.name, len(w2), len(w3))
    for a, b in zip(w2, w3):
        assert a.shape == b.shape, (l2.name, l3.name, a.shape, b.shape)
    if w3:
        l2.set_weights(w3)

rng = np.random.default_rng(0)
sample = rng.random((4, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
out_v3 = model_v3.predict(sample, verbose=0)
out_v2 = model_v2.predict(sample, verbose=0)
max_diff = np.max(np.abs(out_v3 - out_v2))
print('max abs diff between v3 and rebuilt v2 ISL model outputs:', max_diff)
assert max_diff < 1e-4, 'Weight copy mismatch -- outputs diverge!'

from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_keras_model_to_graph_model

OUTPUT_DIR = '../webapp/public/tfjs_model_isl'
convert_keras_model_to_graph_model(model_v2, OUTPUT_DIR)
print('Converted ISL model to TF.js graph model at', OUTPUT_DIR)

import shutil
shutil.copyfile('kaggle_isl_labels.json', f'{OUTPUT_DIR}/labels.json')
print('Copied labels.json into', OUTPUT_DIR)
