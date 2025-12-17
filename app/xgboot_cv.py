import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.regression import GBTRegressor

spark = SparkSession.builder \
    .appName("xgboost_pipeline") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

df = spark.read.parquet("/shared/repartitioned.parquet")
# Spark splits tasks by partitions; do processing with Spark normally

# Shuffle and rebalance partitions for parallel training (reduced for CV memory constraints)
df = df.withColumn("shuffle_key", rand())
df = df.repartition(4, "shuffle_key").drop("shuffle_key")  # Removed .cache() to reduce memory pressure

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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

pipeline = Pipeline(stages=[
    makeIndexer, modelIndexer, transmissionIndexer, 
    makeIndexerEncoder, modelIndexerEncoder, transmissionIndexerEncoder, assembler, gbt
])

# Clean data (drop nulls, rename if needed)
df = df.dropna()
df = df.withColumnRenamed("price", "label")  # If needed, but assuming 'price' is the label

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ===== CROSS-VALIDATION SETUP =====
print("=" * 60)
print("Setting up Cross-Validation for Hyperparameter Tuning")
print("=" * 60)

# Build parameter grid for GBT (reduced combinations to prevent OOM)
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [20, 30]) \
    .addGrid(gbt.maxDepth, [10, 15]) \
    .addGrid(gbt.subsamplingRate, [0.7, 0.8]) \
    .build()

print(f"Total parameter combinations to test: {len(paramGrid)}")
print(f"Total training runs: {len(paramGrid)} combinations × 3 folds = {len(paramGrid) * 3} models\n")

# Create evaluator
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# Create CrossValidator
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,  # 3-fold cross-validation
    parallelism=2,  # Train 2 folds in parallel (matches executor instances)
    seed=42
)

# Fit with cross-validation
print("\nStarting Cross-Validation (this may take several minutes)...")
start_time = time.time()
cvModel = cv.fit(train_df)
end_time = time.time()
print(f"\nCross-Validation completed in {end_time - start_time:.2f} seconds")

# Get best model
model = cvModel.bestModel

# Display best parameters
best_params = {
    "maxIter": model.stages[-1].getMaxIter(),
    "maxDepth": model.stages[-1].getMaxDepth(),
    "subsamplingRate": model.stages[-1].getSubsamplingRate()
}
print("\n" + "=" * 60)
print("Best Hyperparameters Found:")
print("=" * 60)
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Show average metrics from CV
avg_metrics = cvModel.avgMetrics
print(f"\nTop 5 Average RMSE from Cross-Validation:")
sorted_metrics = sorted(enumerate(avg_metrics), key=lambda x: x[1])
for idx, (config_idx, metric) in enumerate(sorted_metrics[:5]):
    print(f"  Rank {idx+1} - Config {config_idx+1}: RMSE = {metric:.2f}")

# ===== COMMENTED OUT: Original single model training =====
# # Fit pipeline
# start_time = time.time()
# model = pipeline.fit(train_df)
# end_time = time.time()
# print(f"Training time: {end_time - start_time} seconds")

# Predictions 
predictions = model.transform(test_df)
predictions.select("year", "mileage", "label", "prediction").show(10)

 # Evaluation on Test Set
print("\n" + "=" * 60)
print("Evaluating Best Model on Test Set")
print("=" * 60)
rmse = evaluator.evaluate(predictions)
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R²: {r2:.4f}")

# Feature Importance
print("\n" + "=" * 60)
print("Feature Importance Analysis")
print("=" * 60)
feature_importances = model.stages[-1].featureImportances
feature_names = assembler.getInputCols()
importance_dict = dict(zip(feature_names, feature_importances.toArray()))
print("Feature Importances (sorted):")
for name, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")

# ===== COMMENTED OUT: Duplicate training time print =====
# print(f"Training time: {end_time - start_time} seconds")

spark.stop()