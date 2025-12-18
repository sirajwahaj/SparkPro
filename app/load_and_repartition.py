import os
from typing import Optional
from pyspark.sql import SparkSession, DataFrame


# Default static paths inside container
DEFAULT_INPUT_PATH = "/shared/combined.parquet"
DEFAULT_OUTPUT_PATH = "/shared/repartitioned.parquet"


def load_and_repartition(input_path: str = DEFAULT_INPUT_PATH,
                         output_path: str = DEFAULT_OUTPUT_PATH,
                         num_partitions: Optional[int] = None,
                         spark: Optional[SparkSession] = None) -> DataFrame:
    """Load combined Parquet via Spark SQL, repartition, and write to output path.

    Parameters:
    - input_path: path to combined parquet (default: /shared/combined.parquet)
    - output_path: path to write repartitioned parquet (default: /shared/repartitioned.parquet)
    - num_partitions: number of partitions to repartition to. If None, uses sparkContext.defaultParallelism.
    - spark: optional SparkSession. If None, a new session will be created and stopped by this function.

    Returns:
    - The repartitioned Spark DataFrame (also written to `output_path`).
    """
    created_session = False
    if spark is None:
        spark = SparkSession.builder.appName("LoadAndRepartition").getOrCreate()
        created_session = True

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Load using Spark and register as SQL view
    df = spark.read.parquet(input_path)
    df.createOrReplaceTempView("combined")
    df_sql = spark.sql("SELECT * FROM combined")

    # Determine number of partitions
    if num_partitions is None:
        num_parts = spark.sparkContext.defaultParallelism
    else:
        num_parts = int(num_partitions)

    df_repart = df_sql.repartition(num_parts)

    # Ensure output parent exists (useful when running outside container)
    out_parent = os.path.dirname(output_path)
    try:
        os.makedirs(out_parent, exist_ok=True)
    except Exception:
        pass

    df_repart.write.mode("overwrite").parquet(output_path)

    if created_session:
        spark.stop()

    return df_repart


if __name__ == "__main__":
    # simple example run when executed directly
    load_and_repartition(num_partitions=2)
