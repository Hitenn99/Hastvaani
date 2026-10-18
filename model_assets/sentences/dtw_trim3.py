import itertools, sys
sys.path.insert(0, 'model_assets/sentences')
import dtw_trim as T  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402
for thr, wa, band in itertools.product((0.6, 0.5, 0.4, 0.3), (2, 3), (0.2, 0.3)):
    kw = dict(w_hand=1.0, w_loc=1, w_arm=wa, frames='any')
    s, _ = T.trimmed(T.D0, thr, 0, **kw); sm, _ = T.trimmed(T.MIRR, thr, 0, **kw)
    D = distance_matrix(s, sm, band)
    print(thr, wa, band, score(D, T.D0['y'], verbose=False).round(4), 'mean2', score(D, T.D0['y'], 'mean', 2, verbose=False).round(4), flush=True)
