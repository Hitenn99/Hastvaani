"""
train_asl_landmarks.py
------------------------
Trains a small MLP on normalized MediaPipe hand landmarks (see
landmarks_asl.py). Honest generalization check = EitanG98 validation split
(never trained on). Saves model_assets/asl_landmarks_model.keras.

Feature normalization (MUST match webapp/src/landmarkFeatures.ts):
  pts = landmarks (21,3) with x,y already in a square pixel-like space
  pts -= pts[0]                       # wrist at origin
  pts /= max(|pts[:, :2]|_2 over points) + 1e-6   # scale by hand size
  flatten -> 63 floats
Run from the repo root.
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 1337
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)


def normalize(P):
    """P: (N,21,3) -> (N,63)."""
    P = P - P[:, :1, :]
    scale = np.linalg.norm(P[:, :, :2], axis=2).max(axis=1)[:, None, None] + 1e-6
    return (P / scale).reshape(len(P), -1).astype(np.float32)


def augment(P):
    """Random mirror, rotation (in-plane), anisotropic scale and jitter on raw landmarks."""
    P = P.copy()
    flip = rng.random(len(P)) < 0.5
    P[flip, :, 0] = 1.0 - P[flip, :, 0]
    c = P[:, :1, :2].copy()
    th = rng.normal(0, 0.15, len(P))
    R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(-1, 2, 2)
    xy = P[:, :, :2] - c
    xy = np.einsum('nij,npj->npi', R, xy)
    xy *= rng.normal(1, 0.08, (len(P), 1, 2))
    P[:, :, :2] = xy + c
    P += rng.normal(0, 0.004, P.shape).astype(np.float32)
    return P


d = np.load('model_assets/landmarks_asl.npz')
X, y, Xv, yv = d['X'], d['y'], d['Xv'], d['yv']
print('train', X.shape, 'crossval', Xv.shape)

idx = rng.permutation(len(X))
n_hold = len(X) // 10
hold, tr = idx[:n_hold], idx[n_hold:]
Xtr, ytr = X[tr], y[tr]
Xho, yho = normalize(X[hold]), y[hold]
Xcv = normalize(Xv)

# Pre-generate augmented copies (originals + 7 augmented).
Xa = np.concatenate([Xtr] + [augment(Xtr) for _ in range(7)])
ya = np.concatenate([ytr] * 8)
Xa = normalize(Xa)

model = models.Sequential([
    layers.Input((63,)),
    layers.Dense(256), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
    layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
    layers.Dense(26, activation='softmax'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    Xa, ya, validation_data=(Xho, yho), epochs=60, batch_size=128, verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ],
)
print('same-source holdout acc:', model.evaluate(Xho, yho, verbose=0)[1])
cv_acc = model.evaluate(Xcv, yv, verbose=0)[1]
print(f'CROSS-SOURCE (EitanG98 val, hand detected) acc: {cv_acc:.4f} on {len(Xcv)} images')
model.save('model_assets/asl_landmarks_model.keras')
json.dump({'crossval_acc': float(cv_acc)}, open('model_assets/asl_landmarks_result.json', 'w'))
