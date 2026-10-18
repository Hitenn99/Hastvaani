"""
train_isl_landmarks.py
------------------------
MLP on two-hand MediaPipe landmarks for ISL (see landmarks_isl.py).

Honest evaluation: the ISL photos are frames of one continuous recording per
letter, so a random split leaks. Here each class is split TEMPORALLY:
train = first 70% of frames, val = 72-80%, test = last 18%. Test frames are
from a different part of the recording (different hand pose drift, position,
etc.) than anything trained on. It's still one signer/session, so treat it as
an optimistic estimate.

Feature normalization (MUST match webapp/src/landmarkFeatures in model.ts):
  hands sorted by wrist x; centroid of all present points subtracted;
  divided by max xy distance from centroid; features =
  [hand0 (63), hand1 (63), present0, present1] = 128 floats (absent hand = 0).
Exports webapp/public/isl_landmarks/model.json with BatchNorm folded in.
Run from the repo root.
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 1337
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)


def sort_hands(P, M):
    """Sort the two hand slots by wrist x; absent hands go last."""
    P, M = P.copy(), M.copy()
    key = np.where(M > 0, P[:, :, 0, 0], 9.0)
    order = np.argsort(key, axis=1)
    P = np.take_along_axis(P, order[:, :, None, None], 1)
    M = np.take_along_axis(M, order, 1)
    return P, M


def normalize(P, M):
    """P (N,2,21,3), M (N,2) -> (N,128)."""
    Pm = P * M[:, :, None, None]
    cnt = M.sum(1)[:, None] * 21.0
    c = Pm.sum((1, 2)) / cnt  # (N,3)
    Q = (P - c[:, None, None, :]) * M[:, :, None, None]
    d = np.linalg.norm(Q[..., :2], axis=3)  # (N,2,21)
    scale = d.max((1, 2))[:, None, None, None] + 1e-6
    Q = Q / scale
    return np.concatenate([Q.reshape(len(P), -1), M], 1).astype(np.float32)


def augment(P, M):
    P = P.copy()
    n = len(P)
    flip = rng.random(n) < 0.5
    P[flip, :, :, 0] = 1.0 - P[flip, :, :, 0]
    P, M2 = sort_hands(P, M)
    c = (P[..., :2] * M2[:, :, None, None]).sum((1, 2)) / (M2.sum(1)[:, None] * 21.0)
    th = rng.normal(0, 0.15, n)
    R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(-1, 2, 2)
    xy = P[..., :2] - c[:, None, None, :]
    xy = np.einsum('nij,nhpj->nhpi', R, xy)
    xy *= rng.normal(1, 0.08, (n, 1, 1, 2))
    P[..., :2] = xy + c[:, None, None, :]
    P += rng.normal(0, 0.004, P.shape).astype(np.float32)
    return P, M2


d = np.load('model_assets/landmarks_isl.npz')
P, M, y, seq = d['P'], d['M'], d['y'], d['seq']
P, M = sort_hands(P, M)
print('samples', len(P), 'two-hand fraction', float((M.sum(1) == 2).mean()))

# Temporal split per class.
frac = np.zeros(len(y))
for c in np.unique(y):
    ix = np.where(y == c)[0]
    frac[ix] = (seq[ix] - seq[ix].min()) / max(1, (seq[ix].max() - seq[ix].min()))
tr = frac < 0.70
va = (frac >= 0.72) & (frac < 0.80)
te = frac >= 0.82
print('train/val/test', tr.sum(), va.sum(), te.sum())


def build():
    m = models.Sequential([
        layers.Input((128,)),
        layers.Dense(384), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(192), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(26, activation='softmax'),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def make_train(Ptr, Mtr, ytr, copies=6):
    Ps, Ms = [Ptr], [Mtr]
    for _ in range(copies):
        a, b = augment(Ptr, Mtr)
        Ps.append(a); Ms.append(b)
    return normalize(np.concatenate(Ps), np.concatenate(Ms)), np.concatenate([ytr] * (copies + 1))


Xtr, ytr = make_train(P[tr], M[tr], y[tr])
Xva, yva = normalize(P[va], M[va]), y[va]
Xte, yte = normalize(P[te], M[te]), y[te]

model = build()
hist = model.fit(
    Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=128, verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ],
)
best_epochs = int(np.argmax(hist.history['val_accuracy']) + 1)
test_acc = model.evaluate(Xte, yte, verbose=0)[1]
print(f'TEMPORAL-SPLIT TEST ACC: {test_acc:.4f} on {len(Xte)} frames (best epochs {best_epochs})')

# Confusions, for the README.
pred = model.predict(Xte, verbose=0).argmax(1)
labels = [chr(65 + i) for i in range(26)]
worst = sorted(((labels[a], labels[b], int(((yte == a) & (pred == b)).sum()))
                for a in range(26) for b in range(26) if a != b), key=lambda t: -t[2])[:8]
print('top confusions (true, pred, n):', worst)

# Refit on ALL frames (more of the session's variety) for the shipped model.
Xall, yall = make_train(P, M, y)
final = build()
final.fit(Xall, yall, epochs=max(best_epochs, 15), batch_size=128, verbose=0)
final.save('model_assets/isl_landmarks_model.keras')

dense = [l for l in final.layers if isinstance(l, layers.Dense)]
bns = [l for l in final.layers if isinstance(l, layers.BatchNormalization)]
out = []
for i, dl in enumerate(dense):
    W, b = dl.get_weights()
    if i < len(bns):
        g, beta, mean, var = bns[i].get_weights()
        s = g / np.sqrt(var + bns[i].epsilon)
        W, b, act = W * s, (b - mean) * s + beta, 'relu'
    else:
        act = 'softmax'
    out.append({'W': W.T.round(6).tolist(), 'b': b.round(6).tolist(), 'act': act})
os.makedirs('webapp/public/isl_landmarks', exist_ok=True)
json.dump({'layers': out}, open('webapp/public/isl_landmarks/model.json', 'w'))
json.dump({'test_acc': float(test_acc)}, open('model_assets/isl_landmarks_result.json', 'w'))
print('exported')
