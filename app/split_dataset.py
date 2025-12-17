import pandas as pd
import numpy as np
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT = os.path.join(BASE, 'vehicle_price_prediction.csv')
OUT_DIR = os.path.join(BASE, 'parts')
SHARED_DIR = os.path.join(BASE, 'shared')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)

if not os.path.exists(INPUT):
    print(f"Input dataset not found at {INPUT}. Place your vehicle_price_prediction.csv at project root.")
    raise SystemExit(1)

print("Loading dataset...")
df = pd.read_csv(INPUT)

n = 3
parts = np.array_split(df, n)

for i, part in enumerate(parts, start=1):
    csv_out = os.path.join(OUT_DIR, f"part_node{i}.csv")
    part.to_csv(csv_out, index=False)
    print(f"Wrote CSV {csv_out} rows={len(part)}")

print("Splitting complete.")
