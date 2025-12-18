# scripts/split_dataset.py
import pandas as pd
import numpy as np
df = pd.read_csv("vehicle_price_prediction.csv")  # original dataset
n = 3
parts = np.array_split(df, n)
for i, part in enumerate(parts, start=1):
    part.to_csv(f"parts/part_node{i}.csv", index=False)