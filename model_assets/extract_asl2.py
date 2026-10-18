"""
extract_asl2.py
-----------------
Extracts the "asl_sign_languages_alphabets_v03" parquet (mirrored on Hugging
Face, no Kaggle login needed) into isl_data-style per-class folders, same
pattern as extract.py used for the ISL dataset. This is the *replacement* ASL
dataset: unlike the originally-provided Kaggle "Sign Language MNIST" model
(clinical, tightly-cropped, grayscale, low background/lighting diversity,
which a real user found performed poorly on live webcam input), this dataset
has real photographic variety -- different backgrounds, lighting, skin tones,
hand framing -- much closer to what an actual webcam feed looks like.
"""
import pyarrow.parquet as pq
import os

LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
OUT_DIR = 'asl_data/by_class'

tbl = pq.read_table('asl_data/train.parquet', columns=['image', 'label'])
df = tbl.to_pandas()
print('total rows:', len(df))

for name in LABEL_NAMES:
    os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)

counts = {}
for i, row in df.iterrows():
    label_idx = int(row['label'])
    name = LABEL_NAMES[label_idx]
    img_bytes = row['image']['bytes']
    n = counts.get(name, 0)
    out_path = os.path.join(OUT_DIR, name, f'{n:05d}.jpg')
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    counts[name] = n + 1
    if i % 2000 == 0:
        print('progress', i, '/', len(df))

print('done. counts:', counts)
