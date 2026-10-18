"""
export_sentence_model.py
---------------------------
Exports a trained sentence_model.keras (Masking -> Bidirectional(GRU) x N -> Dense x N,
Dropout layers are skipped since they're a no-op at inference) to plain JSON that
webapp/src/sentences/model.ts runs as hand-written matrix math -- no TF.js.

Weight layout note: Keras stores a GRU's kernel as (input_dim, 3*units) and recurrent_kernel
as (units, 3*units) -- i.e. output-unit is the LAST axis. model.ts indexes weights as
kernel[output_unit][input_index], so everything is transposed here.

Run from the repo root, after train_sentences.py:
  .venv_tfjs/Scripts/python.exe model_assets/sentences/export_sentence_model.py
"""
import argparse
import json

import numpy as np
import tensorflow as tf


def export_gru(layer) -> dict:
    """One Bidirectional(GRU) Keras layer -> {forward, backward, returnSequences}."""
    fwd, bwd = layer.forward_layer, layer.backward_layer
    units = fwd.units

    def one(cell_weights):
        kernel, rk, bias = cell_weights  # (in,3u), (u,3u), (2,3u)
        return {
            'units': units,
            'kernel': kernel.T.round(6).tolist(),  # (3u, in)
            'recurrentKernel': rk.T.round(6).tolist(),  # (3u, u)
            'inputBias': bias[0].round(6).tolist(),
            'recurrentBias': bias[1].round(6).tolist(),
        }

    return {
        'forward': one(fwd.get_weights()),
        'backward': one(bwd.get_weights()),
        'returnSequences': bool(fwd.return_sequences),
    }


def export_dense(layer) -> dict:
    W, b = layer.get_weights()
    act = layer.get_config()['activation']
    assert act in ('relu', 'softmax'), f'unsupported activation {act}'
    return {'W': W.T.round(6).tolist(), 'b': b.round(6).tolist(), 'act': act}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='model_assets/sentences/sentence_model.keras')
    ap.add_argument('--labels', default='model_assets/sentences/labels.json')
    ap.add_argument('--out-dir', default='webapp/public/sentence_model')
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    max_len, feature_len = model.input_shape[1], model.input_shape[2]

    gru_layers, dense_layers = [], []
    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls == 'Bidirectional':
            gru_layers.append(export_gru(layer))
        elif cls == 'Dense':
            dense_layers.append(export_dense(layer))
        elif cls in ('Masking', 'Dropout', 'InputLayer'):
            continue
        else:
            raise ValueError(f'unexpected layer type in sentence_model.keras: {cls}')

    assert gru_layers, 'no Bidirectional GRU layers found'
    assert dense_layers, 'no Dense layers found'
    assert gru_layers[-1]['returnSequences'] is False, (
        'the last GRU layer must have return_sequences=False (a single vector feeds the Dense head)')

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.out_dir}/model.json', 'w') as f:
        json.dump({
            'maxLen': int(max_len),
            'featureLen': int(feature_len),
            'gru': gru_layers,
            'dense': dense_layers,
        }, f)
    with open(args.labels) as f:
        labels = json.load(f)
    with open(f'{args.out_dir}/labels.json', 'w') as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f'exported {args.out_dir}/model.json ({len(gru_layers)} GRU layers, '
          f'{len(dense_layers)} dense layers) + labels.json ({len(labels)} sentences)')


if __name__ == '__main__':
    main()
