import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

vehicle_df = spark.read.parquet("/shared/repartitioned.parquet")
vehicle_df.show(5)
vehicle_df.printSchema()
df = vehicle_df.select("price").dropna()
null_price_df = df.filter((F.col("price").isNull()) | (F.trim(F.col("price")) == ""))
count = null_price_df.count()
print(f"Found {count} rows with null/empty price")
df.show(5)