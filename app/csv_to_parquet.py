from pyspark.sql import SparkSession
import os
import glob
import shutil
from typing import Optional


DEFAULT_CSV = '/data/part2.csv'
DEFAULT_PARQUET = '/shared/part_node_2.parquet'


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
        # Write to a temporary directory then move the single part file to parquet_path
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

        # ensure destination parent exists
        out_parent = os.path.dirname(parquet_path)
        os.makedirs(out_parent, exist_ok=True)

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


if __name__ == '__main__':
    print('Converting CSV to Parquet:')
    print('CSV ->', DEFAULT_CSV)
    print('Parquet ->', DEFAULT_PARQUET)
    csv_to_parquet()
