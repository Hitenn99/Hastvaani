"""
train_asl2_model.py
---------------------
Retrains the ASL alphabet classifier on a more realistic, webcam-like dataset
after the originally-provided Kaggle "Sign Language MNIST" model performed
poorly on live webcam input (see README.md for the full story).

Dataset: "asl_sign_languages_alphabets_v03" (Hugging Face mirror of a Kaggle
ASL-alphabet-style dataset, no login needed) PLUS EitanG98/asl_letters'
*train* split (a second, independent photo source) -- both pre-cropped to
the hand region by crop_asl2_hands.py / extract_crop_eitan.py (same
MediaPipe bbox logic the deployed app uses) and merged into
asl_data/by_class_cropped/.

Why transfer learning, not a from-scratch CNN: extensive debugging (6+
isolated ablations -- different learning rates, no SpatialAttention, no
GroupNorm, no dropout, a raw numpy/Keras loop with zero custom layers, and
even after fixing the hand-cropping issue described above) all showed the
same failure: loss stuck at/near ln(26) (uniform-prediction chance level),
never breaking out. With only ~330 images/class and real (not studio) visual
diversity, an ImageNet-pretrained MobileNetV2 backbone is used instead of
training conv filters from scratch.

Why TWO datasets, not one: the first version of this model (trained on
Marxulia alone) hit 88.9% on its OWN held-out split but only **27.2%** when
tested against EitanG98's completely separate validation split -- clear
evidence it had memorized Marxulia's specific lighting/camera/background
signature rather than learning actual hand-shape features (the confusion
matrix showed near-random scattering, not sensible letter mix-ups). Combining
two independent sources forces the model to find features that hold across
cameras; heavier color/geometric augmentation (below) does the same. Real
generalization is checked against EitanG98's *validation* split, which is
never trained on -- see crossval_asl2.py.
"""

import os
import pathlib
import random
import json

# Keep temp files inside the project folder (system drive may be low on space).
_tmp = pathlib.Path('tmp').resolve()
_tmp.mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(_tmp)

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 96  # MobileNetV2's smallest well-supported input size
BATCH_SIZE = 32
SEED = 1337
DATA_DIR = pathlib.Path('asl_data/by_class_cropped')
LABELS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(LABELS)
print('Classes:', LABELS, 'count:', NUM_CLASSES)


class SpatialAttention(layers.Layer):
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

    # Inputs arrive normalized to [0, 1] (matching the rest of this project's
    # preprocessing, and the live webapp's preprocess.ts) -- rescale to the
    # [-1, 1] range MobileNetV2 was trained on *inside* the graph, so the JS
    # side never needs to know this model's preprocessing differs from the
    # ASL-MNIST/ISL models.
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    x, _attention_map = SpatialAttention(name='spatial_attention')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='asl2_mobilenet_attention')


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
        # Stronger augmentation than the first attempt (which only flipped):
        # random color/lighting jitter and small geometric perturbations, so
        # the model can't just key off one dataset's specific white balance,
        # exposure, or camera framing -- it has to find features that survive
        # these perturbations, which are the same features that survive a
        # move to a *different* camera/dataset entirely (see the module
        # docstring for why this matters: 88.9% same-source vs 27.2%
        # cross-source before this was added).
        def augment(img, label):
            img = tf.image.random_brightness(img, 0.25)
            img = tf.image.random_contrast(img, 0.7, 1.3)
            img = tf.image.random_saturation(img, 0.7, 1.3)
            img = tf.image.random_hue(img, 0.05)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        # Geometric aug is applied per *batch* (after .batch below), not per
        # image inside a parallel map -- Keras random layers misbehave when
        # invoked per-element across map threads.
    ds = ds.batch(BATCH_SIZE)
    if training:
        geometric_aug = tf.keras.Sequential([
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
        ])
        ds = ds.map(
            lambda imgs, labels: (geometric_aug(imgs, training=True), labels),
            num_parallel_calls=1,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


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
            monitor='val_accuracy', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'asl2_checkpoint.keras', monitor='val_accuracy',
            save_best_only=True,
        ),
    ]

    print('--- Phase 1: training head with frozen MobileNetV2 backbone ---')
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    print('--- Phase 2: gently fine-tuning top of MobileNetV2 backbone at low LR ---')
    # More conservative than the first attempt (30 layers unfroze, 15 epochs
    # budget): only the last ~10 layers, fewer epochs. The first attempt's
    # fine-tuning phase *dropped* same-source validation accuracy (93.5% ->
    # 88.9%) while presumably making cross-source overfitting worse too --
    # unfreezing too much of a backbone this small a dataset can drift it
    # away from the robust, generic ImageNet features that transfer well, in
    # favor of dataset-specific shortcuts.
    base_model = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f'FINAL HELD-OUT TEST ACCURACY: {test_acc:.4f}')
    print(f'FINAL HELD-OUT TEST LOSS: {test_loss:.4f}')

    with open('asl2_test_accuracy.txt', 'w') as f:
        f.write(f'{test_acc:.4f}\n')

    model.save('kaggle_asl2_model.keras')
    with open('kaggle_asl2_labels.json', 'w') as f:
        json.dump(LABELS, f, indent=2)
    print('Saved kaggle_asl2_model.keras and kaggle_asl2_labels.json')
