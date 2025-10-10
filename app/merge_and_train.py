import sys
from pyspark.sql import SparkSession

if len(sys.argv) < 2:
    print("Usage: merge_and_train.py <shared_dir>")
    raise SystemExit(1)

shared = sys.argv[1]
parts = [f"{shared}/processed_node1.parquet",
         f"{shared}/processed_node2.parquet",
         f"{shared}/processed_node3.parquet"]

spark = SparkSession.builder.appName("MergeAndTrain").getOrCreate()
print("Reading processed parts:")
for p in parts:
    print(" -", p)

# Read all parts that exist
from os import path
existing = [p for p in parts if path.exists(p)]
if not existing:
    print("No processed parts found in shared dir. Make sure per-node jobs wrote outputs.")
    raise SystemExit(1)

df = spark.read.parquet(*existing)
print("Combined rows:", df.count())
df.printSchema()

# Placeholder: perform a simple split
train, test = df.randomSplit([0.8, 0.2], seed=42)
print("Train count:", train.count(), "Test count:", test.count())

# Save merged dataset for later training
out_all = f"{shared}/combined.parquet"
df.write.mode('overwrite').parquet(out_all)
print("Wrote combined dataset to", out_all)

spark.stop()
