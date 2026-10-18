import itertools, sys
sys.path.insert(0, 'model_assets/sentences')
import dtw_trim as T  # noqa: E402  (runs its own small sweep on import; cheap)
from dtw_lab import distance_matrix, score  # noqa: E402
res = []
for thr, margin, wh, wl, wa in itertools.product((1.0, 0.8, 0.6), (0, 1, 2), (0.5, 1.0), (1, 2), (1, 2)):
    kw = dict(w_hand=wh, w_loc=wl, w_arm=wa, frames='any')
    s, _ = T.trimmed(T.D0, thr, margin, **kw); sm, _ = T.trimmed(T.MIRR, thr, margin, **kw)
    r = score(distance_matrix(s, sm, 0.2), T.D0['y'], verbose=False)
    res.append((r, thr, margin, kw))
for key, name in ((0, 'top-1'), (1, 'top-3')):
    print('BEST by', name); [print(r.round(4), t, m, kw) for r, t, m, kw in sorted(res, key=lambda t: -t[0][key])[:5]]
