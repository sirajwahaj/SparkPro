import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.regression import GBTRegressor

spark = SparkSession.builder \
    .appName("xgboost_pipeline_part1") \
    .config("spark.executor.instances", "1") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

df = spark.read.parquet("/shared/repartitioned.parquet")
# Spark splits tasks by partitions; do processing with Spark normally

# Shuffle and rebalance partitions for parallel training
#df = df.withColumn("shuffle_key", rand())
#df = df.repartition(8, "shuffle_key").drop("shuffle_key").cache()

makeIndexer = StringIndexer(inputCol="make", outputCol="makeIndex")
modelIndexer = StringIndexer(inputCol="model", outputCol="modelIndex")
transmissionIndexer = StringIndexer(inputCol="transmission", outputCol="transmissionIndex")
#fuelTypeIndexer = StringIndexer(inputCol="fuel_type", outputCol="fuelTypeIndex")
#driveTrainIndexer = StringIndexer(inputCol="drivetrain", outputCol="driveTrainIndex")
#bodyTypeIndexer = StringIndexer(inputCol="body_type", outputCol="bodyTypeIndex")
#exteriorColorIndexer = StringIndexer(inputCol="exterior_color", outputCol="exteriorColorIndex")
#interiorColorIndexer = StringIndexer(inputCol="interior_color", outputCol="interiorColorIndex")
#accidentHistoryIndexer = StringIndexer(inputCol="accident_history", outputCol="accidentHistoryIndex")
#sellerTypeIndexer = StringIndexer(inputCol="seller_type", outputCol="sellerTypeIndex")
#conditionIndexer = StringIndexer(inputCol="condition", outputCol="conditionIndex")
#trimIndexer = StringIndexer(inputCol="trim", outputCol="trimIndex")


makeIndexerEncoder = OneHotEncoder(inputCols=["makeIndex"], outputCols=["makeVec"])
modelIndexerEncoder = OneHotEncoder(inputCols=["modelIndex"], outputCols=["modelVec"])
transmissionIndexerEncoder = OneHotEncoder(inputCols=["transmissionIndex"], outputCols=["transmissionVec"])
#fuelTypeIndexerEncoder = OneHotEncoder(inputCols=["fuelTypeIndex"], outputCols=["fuelTypeVec"])
#driveTrainIndexerEncoder = OneHotEncoder(inputCols=["driveTrainIndex"], outputCols=["driveTrainVec"])
#bodyTypeIndexerEncoder = OneHotEncoder(inputCols=["bodyTypeIndex"], outputCols=["bodyTypeVec"])
#exteriorColorIndexerEncoder = OneHotEncoder(inputCols=["exteriorColorIndex"], outputCols=["exteriorColorVec"])
#interiorColorIndexerEncoder = OneHotEncoder(inputCols=["interiorColorIndex"], outputCols=["interiorColorVec"])
#accidentHistoryIndexerEncoder = OneHotEncoder(inputCols=["accident  HistoryIndex"], outputCols=["accidentHistoryVec"])
#sellerTypeIndexerEncoder = OneHotEncoder(inputCols=["sellerTypeIndex"], outputCols=["sellerTypeVec"])
#conditionIndexerEncoder = OneHotEncoder(inputCols=["conditionIndex"], outputCols=["conditionVec"])
#trimIndexerEncoder = OneHotEncoder(inputCols=["trimIndex"], outputCols=["trimVec"])

assembler = VectorAssembler(
    inputCols=["year", "mileage", "engine_hp",  "vehicle_age", "mileage_per_year", "brand_popularity", "makeVec", "modelVec", "transmissionVec"],
    outputCol="features"
)

gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=30, maxDepth=10, subsamplingRate=0.8, featureSubsetStrategy="auto" )

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    makeIndexer, modelIndexer, transmissionIndexer, 
    makeIndexerEncoder, modelIndexerEncoder, transmissionIndexerEncoder, assembler, gbt
])

# Clean data (drop nulls, rename if needed)
df = df.dropna()
df = df.withColumnRenamed("price", "label")  # If needed, but assuming 'price' is the label

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Fit pipeline
start_time = time.time()
model = pipeline.fit(train_df)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# Predictions 
predictions = model.transform(test_df)
predictions.select("year", "mileage", "label", "prediction").show(10)

# Evaluation
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Feature Importance
feature_importances = model.stages[-1].featureImportances
feature_names = assembler.getInputCols()
importance_dict = dict(zip(feature_names, feature_importances.toArray()))
print("Feature Importances (sorted):")
for name, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")
print(f"\nTraining time: {end_time - start_time:.2f} seconds")

# ===== WRITE RESULTS TO FILE =====
results_file = "/shared/xgb_results.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("GRADIENT BOOSTED TREES REGRESSOR - VEHICLE PRICE PREDICTION\n")
    f.write("=" * 70 + "\n\n")
    
    # Dataset Info
    f.write("DATASET INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total samples (after cleaning): {df.count()}\n")
    f.write(f"Training samples: {train_df.count()}\n")
    f.write(f"Test samples: {test_df.count()}\n")
    f.write(f"Number of features: {len(feature_names)}\n")
    f.write(f"Partitions: {df.rdd.getNumPartitions()}\n\n")
    
    # Model Configuration
    f.write("MODEL CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    gbt_stage = model.stages[-1]
    f.write(f"Algorithm: Gradient Boosted Trees (GBT)\n")
    f.write(f"Max Iterations: {gbt_stage.getMaxIter()}\n")
    f.write(f"Max Depth: {gbt_stage.getMaxDepth()}\n")
    f.write(f"Subsampling Rate: {gbt_stage.getSubsamplingRate()}\n")
    f.write(f"Feature Subset Strategy: {gbt_stage.getFeatureSubsetStrategy()}\n")
    f.write(f"Training Time: {end_time - start_time:.2f} seconds\n\n")
    
    # Performance Metrics - Training Set
    train_predictions = model.transform(train_df)
    train_rmse = evaluator.evaluate(train_predictions)
    train_mae = evaluator.evaluate(train_predictions, {evaluator.metricName: "mae"})
    train_r2 = evaluator.evaluate(train_predictions, {evaluator.metricName: "r2"})
    
    f.write("PERFORMANCE METRICS (TRAINING SET)\n")
    f.write("-" * 70 + "\n")
    f.write(f"RMSE (Root Mean Square Error): {train_rmse:.2f}\n")
    f.write(f"MAE (Mean Absolute Error):     {train_mae:.2f}\n")
    f.write(f"R² (R-squared):                {train_r2:.4f}\n\n")
    
    # Performance Metrics - Test Set
    f.write("PERFORMANCE METRICS (TEST SET)\n")
    f.write("-" * 70 + "\n")
    f.write(f"RMSE (Root Mean Square Error): {rmse:.2f}\n")
    f.write(f"MAE (Mean Absolute Error):     {mae:.2f}\n")
    f.write(f"R² (R-squared):                {r2:.4f}\n\n")
    
    # Feature Importance
    f.write("FEATURE IMPORTANCE (SORTED)\n")
    f.write("-" * 70 + "\n")
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, imp) in enumerate(sorted_features, 1):
        f.write(f"{rank:2d}. {name:<25} {imp:.4f}\n")
    f.write("\n")
    
    # Input Features List
    f.write("INPUT FEATURES\n")
    f.write("-" * 70 + "\n")
    for i, feature in enumerate(feature_names, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")
    
    # Footer
    f.write("=" * 70 + "\n")
    f.write("End of Report\n")
    f.write("=" * 70 + "\n")

print(f"\n✓ Results written to: {results_file}")

spark.stop()