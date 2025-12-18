import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("ml_regressor_part2") \
    .config("spark.executor.instances", "1") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

df = spark.read.parquet("/shared/repartitioned.parquet")
# Remove rows with nulls to keep indexers/encoders stable
df = df.dropna()

# Keep original partitioning for local file access (no reshuffling)
# df = df.withColumn("shuffle_key", rand())
# df = df.repartition(4, "shuffle_key").drop("shuffle_key")

# String Indexers for categorical columns
makeIndexer = StringIndexer(inputCol="make", outputCol="makeIndex")
modelIndexer = StringIndexer(inputCol="model", outputCol="modelIndex")
transmissionIndexer = StringIndexer(inputCol="transmission", outputCol="transmissionIndex")
#Added Indexers for other categorical features
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
               "makeVec", "modelVec", "transmissionVec", "fuelTypeVec", "driveTrainVec", "bodyTypeVec", "exteriorColorVec", "interiorColorVec", "accidentHistoryVec", "sellerTypeVec", "conditionVec", "trimVec"],
    outputCol="features"
)

# Common preprocessing stages (reused for all MLPs)
preprocess_stages = [
    makeIndexer, modelIndexer, transmissionIndexer,
    makeIndexerEncoder, modelIndexerEncoder, transmissionIndexerEncoder,
    fuelTypeIndexer, driveTrainIndexer, bodyTypeIndexer, exteriorColorIndexer, interiorColorIndexer, accidentHistoryIndexer, sellerTypeIndexer, conditionIndexer, trimIndexer,
    fuelTypeIndexerEncoder, driveTrainIndexerEncoder, bodyTypeIndexerEncoder, exteriorColorIndexerEncoder, interiorColorIndexerEncoder, accidentHistoryIndexerEncoder, sellerTypeIndexerEncoder, conditionIndexerEncoder, trimIndexerEncoder,
    assembler
]

# Fit preprocessing once to discover the true feature vector size after one-hot encoding
preprocess_model = Pipeline(stages=preprocess_stages).fit(df)
sample_features = preprocess_model.transform(df.limit(1)).select("features").head()[0]
input_features = sample_features.size
num_classes = 3  # Price categories: 0=Low, 1=Medium, 2=High

# Multilayer Perceptron Classifier definitions with correct input size
# mlp_small = MultilayerPerceptronClassifier(
#     featuresCol="features", labelCol="label",
#     layers=[input_features, 64, 32, num_classes],
#     blockSize=128, maxIter=100, solver="l-bfgs", seed=42
# )

# mlp_medium = MultilayerPerceptronClassifier(
#     featuresCol="features", labelCol="label",
#     layers=[input_features, 128, 64, 32, num_classes],
#     blockSize=128, maxIter=150, solver="l-bfgs", seed=42
# )

mlp_large = MultilayerPerceptronClassifier(
    featuresCol="features", labelCol="label",
    layers=[input_features, 256, 128, 64, num_classes],
    blockSize=128, maxIter=200, solver="l-bfgs", seed=42
)

# Pipelines reuse the same preprocessing steps
# mlp_small_pipeline = Pipeline(stages=preprocess_stages + [mlp_small])
# mlp_medium_pipeline = Pipeline(stages=preprocess_stages + [mlp_medium])
mlp_large_pipeline = Pipeline(stages=preprocess_stages + [mlp_large])

# Bin price into categories (Low, Medium, High) for classification
from pyspark.sql.functions import percentile_approx

price_percentiles = df.select(
    percentile_approx("price", 0.33).alias("p33"),
    percentile_approx("price", 0.67).alias("p67")
).collect()[0]

low_threshold = price_percentiles[0]
high_threshold = price_percentiles[1]

print(f"Price Binning Thresholds:")
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

# Evaluator for classification
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
feature_names = assembler.getInputCols()

# Store results
results = {}

# Train MLP Small Network
# print("=" * 50)
# print("Training MLP Small Network (64-32)")
# print("=" * 50)
# mlp_small_start = time.time()
# mlp_small_model = mlp_small_pipeline.fit(train_df)
# mlp_small_time = time.time() - mlp_small_start
# print(f"Training time: {mlp_small_time:.2f} seconds")

# Predictions
# mlp_small_predictions = mlp_small_model.transform(test_df)
# mlp_small_predictions.select("year", "mileage", "label", "prediction").show(10)

# Evaluation
# mlp_small_accuracy = evaluator.evaluate(mlp_small_predictions)
# mlp_small_precision = evaluator.evaluate(mlp_small_predictions, {evaluator.metricName: "weightedPrecision"})
# mlp_small_f1 = evaluator.evaluate(mlp_small_predictions, {evaluator.metricName: "f1"})
# print(f"Accuracy: {mlp_small_accuracy:.4f}")
# print(f"Precision: {mlp_small_precision:.4f}")
# print(f"F1-Score: {mlp_small_f1:.4f}")
# results["MLP Small"] = {"accuracy": mlp_small_accuracy, "precision": mlp_small_precision, "f1": mlp_small_f1, "time": mlp_small_time}

# Train MLP Medium Network
# print("\n" + "=" * 50)
# print("Training MLP Medium Network (128-64-32)")
# print("=" * 50)
# mlp_medium_start = time.time()
# mlp_medium_model = mlp_medium_pipeline.fit(train_df)
# mlp_medium_time = time.time() - mlp_medium_start
# print(f"Training time: {mlp_medium_time:.2f} seconds")

# Predictions
# mlp_medium_predictions = mlp_medium_model.transform(test_df)
# mlp_medium_predictions.select("year", "mileage", "label", "prediction").show(10)

# Evaluation
# mlp_medium_accuracy = evaluator.evaluate(mlp_medium_predictions)
# mlp_medium_precision = evaluator.evaluate(mlp_medium_predictions, {evaluator.metricName: "weightedPrecision"})
# mlp_medium_f1 = evaluator.evaluate(mlp_medium_predictions, {evaluator.metricName: "f1"})
# print(f"Accuracy: {mlp_medium_accuracy:.4f}")
# print(f"Precision: {mlp_medium_precision:.4f}")
# print(f"F1-Score: {mlp_medium_f1:.4f}")
# results["MLP Medium"] = {"accuracy": mlp_medium_accuracy, "precision": mlp_medium_precision, "f1": mlp_medium_f1, "time": mlp_medium_time}

# Train MLP Large Network
print("\n" + "=" * 50)
print("Training MLP Large Network (256-128-64)")
print("=" * 50)
mlp_large_start = time.time()
mlp_large_model = mlp_large_pipeline.fit(train_df)
mlp_large_time = time.time() - mlp_large_start
print(f"Training time: {mlp_large_time:.2f} seconds")

# Predictions
mlp_large_predictions = mlp_large_model.transform(test_df)
mlp_large_predictions.select("year", "mileage", "label", "prediction").show(10)

# Evaluation
mlp_large_accuracy = evaluator.evaluate(mlp_large_predictions)
mlp_large_precision = evaluator.evaluate(mlp_large_predictions, {evaluator.metricName: "weightedPrecision"})
mlp_large_f1 = evaluator.evaluate(mlp_large_predictions, {evaluator.metricName: "f1"})
print(f"Accuracy: {mlp_large_accuracy:.4f}")
print(f"Precision: {mlp_large_precision:.4f}")
print(f"F1-Score: {mlp_large_f1:.4f}")
results["MLP Large"] = {"accuracy": mlp_large_accuracy, "precision": mlp_large_precision, "f1": mlp_large_f1, "time": mlp_large_time}

# Classification Info
print("\n" + "=" * 50)
print("Neural Network Classification Info")
print("=" * 50)
print("Price Categories (Classes):")
print("  0 = Low Price (< ${:.0f})".format(low_threshold))
print("  1 = Medium Price (${:.0f} - ${:.0f})".format(low_threshold, high_threshold))
print("  2 = High Price (> ${:.0f})".format(high_threshold))
print("\nNetwork Architectures (Input={} features):".format(input_features))
# print(f"- Small:  Input({input_features}) -> 64 -> 32 -> Output(3)")
# print(f"- Medium: Input({input_features}) -> 128 -> 64 -> 32 -> Output(3)")
print(f"- Large:  Input({input_features}) -> 256 -> 128 -> 64 -> Output(3)")

# Model Comparison
print("\n" + "=" * 50)
print("Model Comparison")
print("=" * 50)
for model_name, metrics in results.items():
    print(f"{model_name:<20} - Accuracy: {metrics['accuracy']:>7.4f}, Precision: {metrics['precision']:>7.4f}, F1: {metrics['f1']:>7.4f}, Time: {metrics['time']:>7.2f}s")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
print(f"\n✓ Best Model: {best_model[0]} with Accuracy: {best_model[1]['accuracy']:.4f}")

# ===== WRITE RESULTS TO FILE =====
results_file = "/shared/results.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("MULTILAYER PERCEPTRON CLASSIFIER - VEHICLE PRICE PREDICTION\n")
    f.write("=" * 70 + "\n\n")
    
    # Dataset Info
    f.write("DATASET INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total samples: {df.count()}\n")
    f.write(f"Training samples: {train_df.count()}\n")
    f.write(f"Test samples: {test_df.count()}\n")
    f.write(f"Raw input columns assembled: {len(feature_names)}\n")
    f.write(f"Expanded feature vector size after one-hot: {input_features}\n")
    f.write(f"Number of classes: 3 (Low, Medium, High Price)\n\n")
    
    # Price Binning Info
    f.write("PRICE BINNING THRESHOLDS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Low Price:    < ${low_threshold:.2f}\n")
    f.write(f"Medium Price: ${low_threshold:.2f} - ${high_threshold:.2f}\n")
    f.write(f"High Price:   > ${high_threshold:.2f}\n\n")
    
    # Feature Information
    f.write("INPUT FEATURES\n")
    f.write("-" * 70 + "\n")
    for i, feature in enumerate(feature_names, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")
    
    # Network Architecture
    f.write("NEURAL NETWORK ARCHITECTURES\n")
    f.write("-" * 70 + "\n")
    # f.write(f"Small Network:   Input({input_features}) -> 64 -> 32 -> Output(3)\n")
    # f.write("                 maxIter=100, solver=l-bfgs, blockSize=128\n\n")
    # f.write(f"Medium Network:  Input({input_features}) -> 128 -> 64 -> 32 -> Output(3)\n")
    # f.write("                 maxIter=150, solver=l-bfgs, blockSize=128\n\n")
    f.write(f"Large Network:   Input({input_features}) -> 256 -> 128 -> 64 -> Output(3)\n")
    f.write("                 maxIter=200, solver=l-bfgs, blockSize=128\n\n")
    
    # Model Performance Results
    f.write("MODEL PERFORMANCE RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15} {'Time (s)':<15}\n")
    f.write("-" * 70 + "\n")
    for model_name, metrics in results.items():
        f.write(f"{model_name:<20} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} {metrics['f1']:<15.4f} {metrics['time']:<15.2f}\n")
    f.write("\n")
    
    # Best Model Summary
    f.write("BEST MODEL SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Model Name:   {best_model[0]}\n")
    f.write(f"Accuracy:     {best_model[1]['accuracy']:.4f}\n")
    f.write(f"Precision:    {best_model[1]['precision']:.4f}\n")
    f.write(f"F1-Score:     {best_model[1]['f1']:.4f}\n")
    f.write(f"Training Time: {best_model[1]['time']:.2f} seconds\n\n")
    
    # Footer
    f.write("=" * 70 + "\n")
    f.write("End of Report\n")
    f.write("=" * 70 + "\n")

print(f"\n✓ Results written to: {results_file}")

spark.stop()
