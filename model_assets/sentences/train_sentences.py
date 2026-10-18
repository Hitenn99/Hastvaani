"""
train_sentences.py
---------------------
Trains a small GRU sequence classifier on the two-hand landmark sequences extracted by
extract_landmarks.py: it classifies a signed video against a FIXED CATALOG of the ~101
ISL-CSLTR sentences (isolated-sentence recognition, not open-vocabulary translation --
see README for why). Separate from every other model in this project.

Honest evaluation: each sentence has only ~1-7 recorded videos (mean 4.9), so one video
per sentence (the alphabetically-last file) is held out for validation and NEVER
augmented or trained on -- this is a genuine, if small, generalization check across
recording instances of the same two signers, not a random-frame split (which would leak,
same problem as the original ISL alphabet dataset).

Feature normalization (MUST match webapp/src/sentences/features.ts): per frame, up to two
hands sorted by wrist x, all present points centered on their joint centroid, divided by
the max xy distance from that centroid; features = [hand0 (63), hand1 (63), present0,
present1] = 128 floats per frame (this is IDENTICAL to the per-frame ISL letter feature in
train_isl_landmarks.py, just applied to every frame of a sequence instead of one photo).

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/train_sentences.py
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 1337
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)

DATA = 'model_assets/sentences/sequences.npz'
MAX_LEN = 90
AUG_COPIES = 14  # heavy: ~5 real videos/class is very little data


def sort_hands(P, M):
    """P (...,2,21,3), M (...,2) -> hands sorted by wrist x along the hand axis. The hand
    axis is always axis -3 of P / axis -1 of M, whatever batch/time dims come before it."""
    P, M = P.copy(), M.copy()
    key = np.where(M > 0, P[..., 0, 0], 9.0)  # (...,2)
    order = np.argsort(key, axis=-1)
    P = np.take_along_axis(P, order[..., None, None], axis=-3)
    M = np.take_along_axis(M, order, axis=-1)
    return P, M


def normalize(P, M):
    """P (N,T,2,21,3), M (N,T,2) -> (N,T,128). Padded (all-zero) frames stay all-zero."""
    present = M.sum(-1, keepdims=True) > 0  # (N,T,1)
    Pm = P * M[..., None, None]
    cnt = np.maximum(M.sum(-1), 1e-6)[..., None] * 21.0
    c = Pm.sum((-3, -2)) / cnt  # (N,T,3)
    Q = (P - c[..., None, None, :]) * M[..., None, None]
    d = np.linalg.norm(Q[..., :2], axis=-1)  # (N,T,2,21)
    scale = np.maximum(d.max((-2, -1)), 1e-6)[..., None, None, None]
    Q = Q / scale
    feat = np.concatenate([Q.reshape(*Q.shape[:2], -1), M], -1).astype(np.float32)
    return feat * present  # zero out fully-padded timesteps exactly


def augment(P, M, n_copies):
    """Per-sequence random mirror / rotation / scale / jitter (same transform across all
    frames of one copy, so motion stays coherent) + light temporal resampling."""
    N, T = P.shape[0], P.shape[1]
    outP, outM = [], []
    for _ in range(n_copies):
        Pc, Mc = P.copy(), M.copy()
        flip = rng.random(N) < 0.5
        Pc[flip, ..., 0] = 1.0 - Pc[flip, ..., 0]
        Pc, Mc = sort_hands(Pc, Mc)
        c = (Pc[..., :2] * Mc[..., None, None]).sum((-3, -2)) / (np.maximum(Mc.sum(-1), 1e-6)[..., None] * 21.0)
        th = rng.normal(0, 0.12, N)
        R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(-1, 2, 2)
        xy = Pc[..., :2] - c[:, :, None, None, :]
        xy = np.einsum('nij,nthpj->nthpi', R, xy)
        xy *= rng.normal(1, 0.07, (N, 1, 1, 1, 2))
        Pc[..., :2] = xy + c[:, :, None, None, :]
        Pc += rng.normal(0, 0.004, Pc.shape).astype(np.float32)
        Pc *= Mc[..., None, None]  # keep absent-hand slots exactly zero
        outP.append(Pc)
        outM.append(Mc)
    return np.concatenate(outP), np.concatenate(outM)


def time_resample(P, M, lengths, target=MAX_LEN, jitter=0.15):
    """Randomly stretches/compresses each real (unpadded) sequence, then re-pads."""
    N = len(P)
    outP = np.zeros_like(P)
    outM = np.zeros_like(M)
    for i in range(N):
        n = lengths[i]
        factor = 1 + rng.uniform(-jitter, jitter)
        new_n = int(np.clip(round(n * factor), 4, target))
        idx = np.linspace(0, n - 1, new_n)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        frac = (idx - lo)[:, None, None, None]
        outP[i, :new_n] = P[i, lo] * (1 - frac) + P[i, hi] * frac
        outM[i, :new_n] = np.where(frac[:, 0, 0] > 0.5, M[i, hi], M[i, lo])
    return outP, outM


def main():
    d = np.load(DATA, allow_pickle=True)
    P, M, length, y = d['P'], d['M'], d['length'], d['y']
    sentences = [str(s) for s in d['sentences']]
    source = [str(s) for s in d['source']]
    n_classes = len(sentences)
    print('sequences', len(P), 'classes', n_classes)

    # ---- honest split: last video (by filename) per sentence held out ----
    val_mask = np.zeros(len(y), bool)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue  # can't hold out a video for a class with only one recording
        order = idx[np.argsort([source[i] for i in idx])]
        val_mask[order[-1]] = True
    tr_mask = ~val_mask
    print('train videos', tr_mask.sum(), 'val videos', val_mask.sum(),
          'classes with only 1 video (train-only, no val check):',
          sum(1 for c in range(n_classes) if (y == c).sum() < 2))

    Ptr, Mtr, Ltr, ytr = P[tr_mask], M[tr_mask], length[tr_mask], y[tr_mask]
    Pva, Mva, yva = P[val_mask], M[val_mask], y[val_mask]

    Paug, Maug = time_resample(np.repeat(Ptr, AUG_COPIES, axis=0), np.repeat(Mtr, AUG_COPIES, axis=0),
                                np.repeat(Ltr, AUG_COPIES, axis=0))
    Paug, Maug = augment(Paug, Maug, 1)
    yaug = np.repeat(ytr, AUG_COPIES)
    # include the un-augmented originals too
    Xtr = normalize(np.concatenate([Ptr, Paug]), np.concatenate([Mtr, Maug]))
    Ytr = np.concatenate([ytr, yaug])
    Xva = normalize(Pva, Mva)
    print('train samples (incl. augmented)', len(Xtr), 'val samples', len(Xva))

    model = models.Sequential([
        layers.Input((MAX_LEN, 128)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.GRU(96, return_sequences=True)),
        layers.Dropout(0.35),
        layers.Bidirectional(layers.GRU(64)),
        layers.Dropout(0.35),
        layers.Dense(96, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint('model_assets/sentences/checkpoint.keras',
                                            monitor='val_accuracy', save_best_only=True),
    ]
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, yva), epochs=120, batch_size=32,
                      callbacks=callbacks, verbose=2)

    val_acc = max(hist.history['val_accuracy'])
    pred = model.predict(Xva, verbose=0)
    top3 = np.mean([yva[i] in np.argsort(-pred[i])[:3] for i in range(len(yva))])
    top5 = np.mean([yva[i] in np.argsort(-pred[i])[:5] for i in range(len(yva))])
    print(f'BEST VAL ACCURACY (held-out video per sentence): top-1 {val_acc:.4f}  top-3 {top3:.4f}  top-5 {top5:.4f}')

    model.save('model_assets/sentences/sentence_model.keras')
    with open('model_assets/sentences/labels.json', 'w') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    with open('model_assets/sentences/result.json', 'w') as f:
        json.dump({'val_top1': float(val_acc), 'val_top3': float(top3), 'val_top5': float(top5),
                    'n_classes': n_classes, 'n_train': len(Xtr), 'n_val': len(Xva)}, f, indent=2)
    print('saved model_assets/sentences/sentence_model.keras + labels.json + result.json')


if __name__ == '__main__':
    main()
