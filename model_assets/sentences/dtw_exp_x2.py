import sys, itertools
sys.path.insert(0, 'model_assets/sentences')
import dtw_exp_x as X  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
X.D0 = load_pose()
res = []
for shape, ws, wa, trim, metric in itertools.product(('none', 'desc'), (0.5, 1.0), (2, 3, 4), (0.4, 0.6), (0, 1)):
    if shape == 'none' and ws != 1.0: continue
    a = X.run(f'{shape} ws{ws} wa{wa} trim{trim} m{metric}', metric=metric, shape=shape, ws=ws, wa=wa, trim=trim)
    res.append((a, shape, ws, wa, trim, metric))
print('BEST top-3:'); [print(r[0].round(3), r[1:]) for r in sorted(res, key=lambda r: -(r[0][1] + r[0][0]))[:6]]
