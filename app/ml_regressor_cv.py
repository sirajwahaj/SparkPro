import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, percentile_approx
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder \
    .appName("ml_regressor_cv") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

df = spark.read.parquet("/shared/repartitioned.parquet")
# Remove rows with nulls to keep indexers/encoders stable
df = df.dropna()

# Shuffle and rebalance partitions for parallel training
df = df.withColumn("shuffle_key", rand())
df = df.repartition(4, "shuffle_key").drop("shuffle_key")

# String Indexers for categorical columns
makeIndexer = StringIndexer(inputCol="make", outputCol="makeIndex")
modelIndexer = StringIndexer(inputCol="model", outputCol="modelIndex")
transmissionIndexer = StringIndexer(inputCol="transmission", outputCol="transmissionIndex")
fuelTypeIndexer = StringIndexer(inputCol="fuel_type", outputCol="fuelTypeIndex")
driveTrainIndexer = StringIndexer(inputCol="drivetrain", outputCol="driveTrainIndex")
bodyTypeIndexer = StringIndexer(inputCol="body_type", outputCol="bodyTypeIndex")
exteriorColorIndexer = StringIndexer(inputCol="exterior_color", outputCol="exteriorColorIndex")
interiorColorIndexer = StringIndexer(inputCol="interior_color", outputCol="interiorColorIndex")
accidentHistoryIndexer = StringIndexer(inputCol="accident_history", outputCol="accidentHistoryIndex")
sellerTypeIndexer = StringIndexer(inputCol="seller_type", outputCol="sellerTypeIndex")
conditionIndexer = StringIndexer(inputCol="condition", outputCol="conditionIndex")
trimIndexer = StringIndexer(inputCol="trim", outputCol="trimIndex")

# One-Hot Encoders
makeIndexerEncoder = OneHotEncoder(inputCols=["makeIndex"], outputCols=["makeVec"])
modelIndexerEncoder = OneHotEncoder(inputCols=["modelIndex"], outputCols=["modelVec"])
transmissionIndexerEncoder = OneHotEncoder(inputCols=["transmissionIndex"], outputCols=["transmissionVec"])
fuelTypeIndexerEncoder = OneHotEncoder(inputCols=["fuelTypeIndex"], outputCols=["fuelTypeVec"])
driveTrainIndexerEncoder = OneHotEncoder(inputCols=["driveTrainIndex"], outputCols=["driveTrainVec"])
bodyTypeIndexerEncoder = OneHotEncoder(inputCols=["bodyTypeIndex"], outputCols=["bodyTypeVec"])
exteriorColorIndexerEncoder = OneHotEncoder(inputCols=["exteriorColorIndex"], outputCols=["exteriorColorVec"])
interiorColorIndexerEncoder = OneHotEncoder(inputCols=["interiorColorIndex"], outputCols=["interiorColorVec"])
accidentHistoryIndexerEncoder = OneHotEncoder(inputCols=["accidentHistoryIndex"], outputCols=["accidentHistoryVec"])
sellerTypeIndexerEncoder = OneHotEncoder(inputCols=["sellerTypeIndex"], outputCols=["sellerTypeVec"])
conditionIndexerEncoder = OneHotEncoder(inputCols=["conditionIndex"], outputCols=["conditionVec"])
trimIndexerEncoder = OneHotEncoder(inputCols=["trimIndex"], outputCols=["trimVec"])

# Vector Assembler
assembler = VectorAssembler(
    inputCols=["year", "mileage", "engine_hp", "vehicle_age", "mileage_per_year", "brand_popularity", 
               "makeVec", "modelVec", "transmissionVec", "fuelTypeVec", "driveTrainVec", "bodyTypeVec", 
               "exteriorColorVec", "interiorColorVec", "accidentHistoryVec", "sellerTypeVec", "conditionVec", "trimVec"],
    outputCol="features"
)

# Common preprocessing stages (reused for all MLPs)
preprocess_stages = [
    makeIndexer, modelIndexer, transmissionIndexer,
    makeIndexerEncoder, modelIndexerEncoder, transmissionIndexerEncoder,
    fuelTypeIndexer, driveTrainIndexer, bodyTypeIndexer, exteriorColorIndexer, interiorColorIndexer, 
    accidentHistoryIndexer, sellerTypeIndexer, conditionIndexer, trimIndexer,
    fuelTypeIndexerEncoder, driveTrainIndexerEncoder, bodyTypeIndexerEncoder, exteriorColorIndexerEncoder, 
    interiorColorIndexerEncoder, accidentHistoryIndexerEncoder, sellerTypeIndexerEncoder, conditionIndexerEncoder, trimIndexerEncoder,
    assembler
]

# Fit preprocessing once to discover the true feature vector size after one-hot encoding
print("=" * 60)
print("Computing Feature Vector Size")
print("=" * 60)
preprocess_model = Pipeline(stages=preprocess_stages).fit(df)
sample_features = preprocess_model.transform(df.limit(1)).select("features").head()[0]
input_features = sample_features.size
num_classes = 3  # Price categories: 0=Low, 1=Medium, 2=High
print(f"Assembled feature vector size: {input_features}")
print(f"Number of output classes: {num_classes}\n")

# Bin price into categories (Low, Medium, High) for classification
price_percentiles = df.select(
    percentile_approx("price", 0.33).alias("p33"),
    percentile_approx("price", 0.67).alias("p67")
).collect()[0]

low_threshold = price_percentiles[0]
high_threshold = price_percentiles[1]

print("=" * 60)
print("Price Binning Thresholds")
print("=" * 60)
print(f"  Low:    < ${low_threshold:.0f}")
print(f"  Medium: ${low_threshold:.0f} - ${high_threshold:.0f}")
print(f"  High:   > ${high_threshold:.0f}\n")

# Create bucketizer for price categories
bucketizer = Bucketizer(
    splits=[0, low_threshold, high_threshold, float('inf')],
    inputCol="price",
    outputCol="label"
)
df = bucketizer.transform(df).drop("price")

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Multilayer Perceptron Classifier with correct input size
mlp = MultilayerPerceptronClassifier(
    featuresCol="features", labelCol="label",
    layers=[input_features, 128, 64, 32, num_classes],
    blockSize=128, maxIter=100, solver="l-bfgs", seed=42
)

# Pipeline with all stages
pipeline = Pipeline(stages=preprocess_stages + [mlp])

# ===== CROSS-VALIDATION SETUP =====
print("=" * 60)
print("Setting up Cross-Validation for Hyperparameter Tuning")
print("=" * 60)

# Build parameter grid for MLP (reduced combinations to prevent OOM)
paramGrid = ParamGridBuilder() \
    .addGrid(mlp.maxIter, [50, 100, 150]) \
    .addGrid(mlp.blockSize, [64, 128]) \
    .build()

print(f"Total parameter combinations to test: {len(paramGrid)}")
print(f"Total training runs: {len(paramGrid)} combinations × 3 folds = {len(paramGrid) * 3} models\n")

# Create evaluator for classification
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

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
print("Starting Cross-Validation (this may take several minutes)...")
start_time = time.time()
cvModel = cv.fit(train_df)
end_time = time.time()
cv_duration = end_time - start_time
print(f"\nCross-Validation completed in {cv_duration:.2f} seconds")

# Get best model
best_model = cvModel.bestModel

# Display best parameters
best_mlp_stage = best_model.stages[-1]
best_params = {
    "maxIter": best_mlp_stage.getMaxIter(),
    "blockSize": best_mlp_stage.getBlockSize(),
    "solver": best_mlp_stage.getSolver(),
    "layers": best_mlp_stage.getLayers()
}
print("\n" + "=" * 60)
print("Best Hyperparameters Found:")
print("=" * 60)
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Show average metrics from CV
avg_metrics = cvModel.avgMetrics
print(f"\nTop {min(5, len(avg_metrics))} Average Accuracy from Cross-Validation:")
sorted_metrics = sorted(enumerate(avg_metrics), key=lambda x: x[1], reverse=True)  # Higher accuracy is better
for idx, (config_idx, metric) in enumerate(sorted_metrics[:5]):
    print(f"  Rank {idx+1} - Config {config_idx+1}: Accuracy = {metric:.4f}")

# ===== EVALUATION ON TEST SET =====
print("\n" + "=" * 60)
print("Evaluating Best Model on Test Set")
print("=" * 60)

predictions = best_model.transform(test_df)
predictions.select("year", "mileage", "label", "prediction").show(10)

# Evaluation metrics
accuracy = evaluator.evaluate(predictions)
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-Score:  {f1:.4f}")

# ===== WRITE RESULTS TO FILE =====
results_file = "/shared/cv_results.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("MULTILAYER PERCEPTRON CLASSIFIER - CROSS-VALIDATION RESULTS\n")
    f.write("=" * 70 + "\n\n")
    
    # Dataset Info
    f.write("DATASET INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total samples: {df.count()}\n")
    f.write(f"Training samples: {train_df.count()}\n")
    f.write(f"Test samples: {test_df.count()}\n")
    f.write(f"Raw input columns assembled: {len(assembler.getInputCols())}\n")
    f.write(f"Expanded feature vector size after one-hot: {input_features}\n")
    f.write(f"Number of classes: 3 (Low, Medium, High Price)\n\n")
    
    # Price Binning Info
    f.write("PRICE BINNING THRESHOLDS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Low Price:    < ${low_threshold:.2f}\n")
    f.write(f"Medium Price: ${low_threshold:.2f} - ${high_threshold:.2f}\n")
    f.write(f"High Price:   > ${high_threshold:.2f}\n\n")
    
    # Cross-Validation Setup
    f.write("CROSS-VALIDATION CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Number of folds: 3\n")
    f.write(f"Parallelism: 2\n")
    f.write(f"Total parameter combinations: {len(paramGrid)}\n")
    f.write(f"Total models trained: {len(paramGrid) * 3}\n")
    f.write(f"Cross-validation time: {cv_duration:.2f} seconds\n\n")
    
    # Parameter Grid
    f.write("PARAMETER GRID EXPLORED\n")
    f.write("-" * 70 + "\n")
    f.write("maxIter: [50, 100, 150]\n")
    f.write("blockSize: [64, 128]\n\n")
    
    # Best Hyperparameters
    f.write("BEST HYPERPARAMETERS\n")
    f.write("-" * 70 + "\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    f.write("\n")
    
    # CV Results Summary
    f.write("CROSS-VALIDATION RESULTS (Top 5 Configurations)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Rank':<6} {'Config':<10} {'Avg Accuracy':<15}\n")
    f.write("-" * 70 + "\n")
    for idx, (config_idx, metric) in enumerate(sorted_metrics[:5]):
        f.write(f"{idx+1:<6} {config_idx+1:<10} {metric:<15.4f}\n")
    f.write("\n")
    
    # Test Set Performance
    f.write("TEST SET PERFORMANCE (BEST MODEL)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n\n")
    
    # Feature Information
    f.write("INPUT FEATURES\n")
    f.write("-" * 70 + "\n")
    feature_names = assembler.getInputCols()
    for i, feature in enumerate(feature_names, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")
    
    # Footer
    f.write("=" * 70 + "\n")
    f.write("End of Cross-Validation Report\n")
    f.write("=" * 70 + "\n")

print(f"\n✓ Cross-validation results written to: {results_file}")

spark.stop()
