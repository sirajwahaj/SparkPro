import pandas as pd
import numpy as np
import os
import sys
import glob
import shutil
from typing import Optional
from pyspark.sql import SparkSession
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_1 = os.path.join(BASE, 'parts/part_node1.csv')
INPUT_2 = os.path.join(BASE, 'parts/part_node2.csv')
INPUT_3 = os.path.join(BASE, 'parts/part_node3.csv')
# Write combined output to the shared mount inside the container
OUT = '/shared/combined.csv'

DEFAULT_CSV = '/shared/combined.csv'
DEFAULT_PARQUET = '/shared/combined.parquet'


def csv_to_parquet(csv_path: str = DEFAULT_CSV,
                   parquet_path: str = DEFAULT_PARQUET,
                   infer_schema: bool = True,
                   spark: Optional[SparkSession] = None,
                   single_file: bool = False):
    """Convert a CSV file to Parquet using Spark.

    Parameters:
    - csv_path: path to the CSV file (default /shared/combined.csv)
    - parquet_path: output parquet directory (default /shared/combined.parquet)
    - infer_schema: whether to infer CSV schema
    - spark: optional SparkSession. If None, one will be created and stopped.

    Returns the written DataFrame.
    """
    created = False
    if spark is None:
        spark = SparkSession.builder.appName('csv_to_parquet').getOrCreate()
        created = True

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV input not found: {csv_path}')

    reader = spark.read.option('header', 'true')
    if infer_schema:
        reader = reader.option('inferSchema', 'true')

    df = reader.csv(csv_path)

    # ensure output parent exists
    out_parent = os.path.dirname(parquet_path)
    try:
        os.makedirs(out_parent, exist_ok=True)
    except Exception:
        pass

    if single_file:
        tmp_dir = parquet_path + "_tmp"
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        df.coalesce(1).write.mode('overwrite').parquet(tmp_dir)

        # find the generated part file
        part_files = glob.glob(os.path.join(tmp_dir, 'part-*.parquet'))
        if not part_files:
            raise RuntimeError(f"No part file found in temporary parquet output: {tmp_dir}")

        part_file = part_files[0]

        # remove existing target file if present
        try:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
        except Exception:
            pass

        shutil.move(part_file, parquet_path)

        # cleanup tmp_dir
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
    else:
        df.write.mode('overwrite').parquet(parquet_path)

    if created:
        spark.stop()

    return df


df1 = pd.read_csv(INPUT_1)
df2 = pd.read_csv(INPUT_2)
df3 = pd.read_csv(INPUT_3)

df = pd.concat([df1, df2, df3], ignore_index=True)
# Ensure output directory exists (the /shared mount should exist in containers)
out_dir = os.path.dirname(OUT)
try:
    os.makedirs(out_dir, exist_ok=True)
except Exception:
    pass

df.to_csv(OUT, index=False)

csv_to_parquet(single_file=True)
print(f"Wrote combined CSV to {OUT} and single Parquet to {DEFAULT_PARQUET} with {len(df)} rows.")