import os
from typing import Optional
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DEFAULT_PARQUET = "/shared/combined.parquet"
DEFAULT_CSV = "/shared/combined.csv"
OUT_PARQUET = "/shared/null_price_rows.parquet"
OUT_CSV = "/shared/null_price_rows.csv"


def find_null_price(input_parquet: str = DEFAULT_PARQUET,
                    input_csv: str = DEFAULT_CSV,
                    out_parquet: str = OUT_PARQUET,
                    out_csv: str = OUT_CSV,
                    spark: Optional[SparkSession] = None):
    """Find rows where `price` is null or empty string and write them out.

    Returns the DataFrame of null-price rows.
    """
    created = False
    if spark is None:
        spark = SparkSession.builder.appName("find_null_price").getOrCreate()
        created = True

    df = None
    if os.path.exists(input_parquet):
        print(f"Loading from parquet: {input_parquet}")
        df = spark.read.parquet(input_parquet)
    elif os.path.exists(input_csv):
        print(f"Parquet not found; loading CSV: {input_csv}")
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv)
    else:
        raise FileNotFoundError(f"Neither {input_parquet} nor {input_csv} exists")

    # Normalize column name to 'price' if different case
    cols = [c for c in df.columns]
    price_col = None
    for c in cols:
        if c.lower() == "price":
            price_col = c
            break
    if price_col is None:
        raise KeyError("No 'price' column found in the dataset")

    # Filter rows where price is null or empty
    null_price_df = df.filter((F.col(price_col).isNull()) | (F.trim(F.col(price_col)) == ""))

    count = null_price_df.count()
    print(f"Found {count} rows with null/empty price")

    if count > 0:
        # write parquet and csv outputs
        out_parent = os.path.dirname(out_parquet)
        try:
            os.makedirs(out_parent, exist_ok=True)
        except Exception:
            pass

        null_price_df.write.mode("overwrite").parquet(out_parquet)

        # also write CSV for quick inspection (coalesce to 1)
        tmp_csv_dir = out_csv + "_tmp"
        try:
            if os.path.exists(tmp_csv_dir):
                import shutil

                shutil.rmtree(tmp_csv_dir)
        except Exception:
            pass

        null_price_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_csv_dir)

        # move the single part CSV to the out_csv path
        try:
            import glob
            import shutil

            part_files = glob.glob(os.path.join(tmp_csv_dir, "part-*.csv"))
            if part_files:
                # remove existing file if any
                try:
                    if os.path.exists(out_csv):
                        os.remove(out_csv)
                except Exception:
                    pass
                shutil.move(part_files[0], out_csv)
            # cleanup tmp dir
            try:
                shutil.rmtree(tmp_csv_dir)
            except Exception:
                pass
        except Exception as e:
            print("Failed to write single CSV file:", e)

    if created:
        spark.stop()

    return null_price_df


if __name__ == "__main__":
    find_null_price()
