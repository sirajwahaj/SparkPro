import pandas as pd
import numpy as np
import os

INPUT = os.path.join(os.path.dirname(__file__), '..', 'vehicle_price_dataset.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'parts')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(INPUT):
    print(f"Input dataset not found at {INPUT}. Place your vehicle_price_dataset.csv at project root.")
    raise SystemExit(1)

print("Loading dataset...")
df = pd.read_csv(INPUT)

n = 3
parts = np.array_split(df, n)
for i, part in enumerate(parts, start=1):
    out = os.path.join(OUT_DIR, f"part_node{i}.csv")
    part.to_csv(out, index=False)
    print(f"Wrote {out} rows={len(part)}")

print("Splitting complete.")
