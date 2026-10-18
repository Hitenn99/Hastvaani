"""
convert_asl2_for_tfjs.py
--------------------------
Converts the retrained ASL model (train_asl2_model.py: MobileNetV2 backbone +
SpatialAttention head, 96x96x3 RGB, 26 classes A-Z) to a TF.js graph model.

Same underlying problem as rebuild_for_tfjs.py / convert_isl_for_tfjs.py
(tensorflowjs's h5 loader can't see a custom layer registered in a different
process) but MobileNetV2 has ~150 layers with skip connections, too many to
safely rebuild by *position* like the earlier from-scratch CNNs did -- one
misaligned layer would silently copy the wrong weights into the wrong slot.
Instead this matches layers by NAME: tf_keras.applications.MobileNetV2 uses
the same layer names as tf.keras's (both are ports of the same reference
architecture), so weights transfer correctly by name lookup regardless of
internal ordering differences between the two Keras implementations.
"""
import json

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras import layers as k2layers

from train_asl2_model import SpatialAttention as SpatialAttentionV3, IMG_SIZE

model_v3 = tf.keras.models.load_model(
    'kaggle_asl2_model.keras', custom_objects={'SpatialAttention': SpatialAttentionV3}
)
with open('kaggle_asl2_labels.json') as f:
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
    x = k2layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    base_model = tf_keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=None
    )
    x = base_model(x, training=False)
    x, _attention_map = SpatialAttentionV2(name='spatial_attention')(x)
    x = k2layers.GlobalAveragePooling2D()(x)
    x = k2layers.Dense(256, activation='relu')(x)
    x = k2layers.Dropout(0.4)(x)
    outputs = k2layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return tf_keras.Model(inputs=inputs, outputs=outputs, name='asl2_mobilenet_attention'), base_model


model_v2, base_model_v2 = build_v2(IMG_SIZE, num_classes)

# Find the corresponding nested MobileNetV2 submodel + top-level layers in
# the v3 (Keras-3) model, then copy weights by NAME (robust to internal
# ordering differences between tf_keras and keras-3's MobileNetV2 ports).
base_model_v3 = next(l for l in model_v3.layers if isinstance(l, tf.keras.Model))

v3_by_name = {l.name: l for l in base_model_v3.layers}
copied, missing = 0, []
for l2 in base_model_v2.layers:
    if isinstance(l2, (k2layers.InputLayer,)):
        continue  # no weights to copy; input layer naming can differ between keras3/tf_keras
    l3 = v3_by_name.get(l2.name)
    if l3 is None:
        missing.append(l2.name)
        continue
    w2, w3 = l2.get_weights(), l3.get_weights()
    if len(w2) != len(w3):
        missing.append(l2.name)
        continue
    l2.set_weights(w3)
    copied += 1
print(f'MobileNetV2 backbone: copied {copied}/{len(base_model_v2.layers)} layers by name; missing: {missing}')
assert not missing, 'Some backbone layers could not be matched by name -- aborting rather than silently using random weights.'

# Copy the custom head (spatial_attention, dense, predictions) by name too.
v3_top_by_name = {l.name: l for l in model_v3.layers}
for l2 in model_v2.layers:
    if l2 is base_model_v2 or isinstance(l2, (k2layers.InputLayer, k2layers.Rescaling)):
        continue
    l3 = v3_top_by_name.get(l2.name)
    if l3 is not None and l2.get_weights():
        assert len(l2.get_weights()) == len(l3.get_weights()), l2.name
        l2.set_weights(l3.get_weights())

# Verify numerical equivalence before trusting this.
rng = np.random.default_rng(0)
sample = rng.random((4, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
out_v3 = model_v3.predict(sample, verbose=0)
out_v2 = model_v2.predict(sample, verbose=0)
max_diff = np.max(np.abs(out_v3 - out_v2))
print('max abs diff between v3 and rebuilt v2 ASL2 model outputs:', max_diff)
assert max_diff < 1e-3, 'Weight copy mismatch -- outputs diverge!'

from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_keras_model_to_graph_model

OUTPUT_DIR = 'tfjs_model_asl_new'
convert_keras_model_to_graph_model(model_v2, OUTPUT_DIR)
print('Converted ASL2 model to TF.js graph model at', OUTPUT_DIR)

import shutil
shutil.copyfile('kaggle_asl2_labels.json', f'{OUTPUT_DIR}/labels.json')
print('Copied labels.json into', OUTPUT_DIR)
