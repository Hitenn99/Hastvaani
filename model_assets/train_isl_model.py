"""
train_isl_model.py
-------------------
Trains the ISL (Indian Sign Language) alphabet classifier, following the same
proven pattern as kaggle_model.py (real Kaggle-sourced data, CNN + GroupNorm +
SpatialAttention, genuine held-out test accuracy -- never BatchNormalization,
see the pitfall note in kaggle_model.py for why).

Dataset: mirrored from the Kaggle "Indian Sign Language (ISL)" dataset family
(kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset and
equivalents) via its public Hugging Face parquet mirror
(Hemg/Indian_sign_language_dataset, no login required) since kaggle.com itself
required an account/API token this session did not have. 42,745 real photos
across 35 classes (digits 1-9 + letters A-Z); digits were dropped (out of
scope -- alphabet only) leaving 31,945 images across 26 letter classes,
extracted to isl_data/by_class/<LETTER>/*.jpg by extract.py.

Images are two-handed ISL static hand poses, 128x128 RGB, photographed against
a dark background (see isl_data/sample_*.png). We downsize to 64x64 for
practical CPU training time in this environment (no GPU available on this
machine) while keeping the images large enough to preserve hand-shape detail.
"""

import pathlib
import random

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 64
BATCH_SIZE = 64
SEED = 1337
DATA_DIR = pathlib.Path('isl_data/by_class')
LABELS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(LABELS)
print('Classes:', LABELS, 'count:', NUM_CLASSES)


class SpatialAttention(layers.Layer):
    """Same spatial-attention block as kaggle_model.py: a 1x1 Conv2D sigmoid
    gate that reweights the CNN's feature maps before pooling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = layers.Conv2D(1, kernel_size=1, activation='sigmoid')

    def build(self, input_shape):
        self.attn_conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        attention_map = self.attn_conv(inputs)
        return inputs * attention_map, attention_map

    def get_config(self):
        return super().get_config()


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3), name='hand_crop')

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GroupNormalization(groups=16)(x)

    x, _attention_map = SpatialAttention(name='spatial_attention')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='isl_cnn_attention')


def make_split_file_lists():
    """80/10/10 train/val/test split, stratified per class, fixed seed."""
    rng = random.Random(SEED)
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for label_idx, label in enumerate(LABELS):
        files = sorted((DATA_DIR / label).glob('*.jpg'))
        files = [str(f) for f in files]
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_files += files[:n_train]
        val_files += files[n_train:n_train + n_val]
        test_files += files[n_train + n_val:]
        train_labels += [label_idx] * n_train
        val_labels += [label_idx] * (len(files[n_train:n_train + n_val]))
        test_labels += [label_idx] * (len(files[n_train + n_val:]))
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def make_dataset(files, labels, training):
    path_ds = tf.data.Dataset.from_tensor_slices((files, labels))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        return img, label

    ds = path_ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(4096, seed=SEED)
        ds = ds.map(
            lambda img, label: (tf.image.random_flip_left_right(img), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = make_split_file_lists()
    print(f'train={len(train_f)} val={len(val_f)} test={len(test_f)}')

    train_ds = make_dataset(train_f, train_l, training=True)
    val_ds = make_dataset(val_f, val_l, training=False)
    test_ds = make_dataset(test_f, test_l, training=False)

    model = build_model(IMG_SIZE, NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f'FINAL HELD-OUT TEST ACCURACY: {test_acc:.4f}')
    print(f'FINAL HELD-OUT TEST LOSS: {test_loss:.4f}')

    with open('isl_test_accuracy.txt', 'w') as f:
        f.write(f'{test_acc:.4f}\n')

    model.save('kaggle_isl_model.keras')
    with open('kaggle_isl_labels.json', 'w') as f:
        import json
        json.dump(LABELS, f, indent=2)
    print('Saved kaggle_isl_model.keras and kaggle_isl_labels.json')
