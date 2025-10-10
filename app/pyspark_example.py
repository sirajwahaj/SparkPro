# app/pi.py - simple Spark job (Monte Carlo Pi)
from random import random
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PiExample").getOrCreate()
sc = spark.sparkContext

NUM_SAMPLES = 1000000

def inside(_):
    x = random()*2 - 1
    y = random()*2 - 1
    return 1 if x*x + y*y <= 1 else 0

count = sc.parallelize(range(NUM_SAMPLES)).map(inside).sum()
pi = 4.0 * count / NUM_SAMPLES
print("****************************** Pi is roughly %f *********************************" % pi)

spark.stop()