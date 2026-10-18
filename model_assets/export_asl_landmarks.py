"""Exports asl_landmarks_model.keras to webapp/public/asl_landmarks/model.json
(BatchNorm folded into the Dense layers, so the app runs it as plain matmuls)."""
import json
import numpy as np
import tensorflow as tf

m = tf.keras.models.load_model('model_assets/asl_landmarks_model.keras')
layers = []
dense = [l for l in m.layers if isinstance(l, tf.keras.layers.Dense)]
bns = [l for l in m.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
for i, d in enumerate(dense):
    W, b = d.get_weights()
    if i < len(bns):
        g, beta, mean, var = bns[i].get_weights()
        s = g / np.sqrt(var + bns[i].epsilon)
        W = W * s
        b = (b - mean) * s + beta
        act = 'relu'
    else:
        act = 'softmax'
    layers.append({'W': W.T.round(6).tolist(), 'b': b.round(6).tolist(), 'act': act})  # W: (out,in)
import os
os.makedirs('webapp/public/asl_landmarks', exist_ok=True)
json.dump({'layers': layers}, open('webapp/public/asl_landmarks/model.json', 'w'))
# sanity: compare to keras on random input
x = np.random.rand(3, 63).astype(np.float32)
h = x
for L in layers:
    h = h @ np.array(L['W']).T + np.array(L['b'])
    h = np.maximum(h, 0) if L['act'] == 'relu' else np.exp(h - h.max(1, keepdims=True)) / np.exp(h - h.max(1, keepdims=True)).sum(1, keepdims=True)
print('max abs diff vs keras:', np.abs(h - m.predict(x, verbose=0)).max())
