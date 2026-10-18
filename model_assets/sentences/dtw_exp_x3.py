import sys, itertools
sys.path.insert(0, 'model_assets/sentences')
import dtw_exp_x as X  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
X.D0 = load_pose()
res = []
for ws, wa, wl, trim in itertools.product((1.0, 1.5, 2.0), (4, 5, 6), (1, 2), (0.3, 0.4, 0.5)):
    a = X.run(f'desc ws{ws} wa{wa} wl{wl} trim{trim}', metric=0, shape='desc', ws=ws, wa=wa, wl=wl, trim=trim)
    res.append((a, ws, wa, wl, trim))
print('BEST:'); [print(r[0].round(3), r[1:]) for r in sorted(res, key=lambda r: -(r[0][1] + r[0][0]))[:8]]
