import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

if len(sys.argv) < 3:
    print("Usage: process_local_part.py <input_csv_path> <output_parquet_path>")
    raise SystemExit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]

spark = SparkSession.builder.appName("ProcessLocalPart").getOrCreate()
print(f"Reading local part from {in_path}")
df = spark.read.option("header", True).csv(in_path)

# Simple preprocessing: cast numeric fields if present and drop NA for target
# Adjust column names to your dataset
if 'price' in df.columns:
    df = df.withColumn('price', col('price').cast('double'))

# Example: drop records where price is null
df = df.na.drop(subset=['price'])

print(f"Writing processed part to {out_path}")
df.write.mode('overwrite').parquet(out_path)

spark.stop()
print("Done") 
