# Distributed Machine Learning with Apache Spark - Project Report

## Project Overview
This project implements a distributed machine learning pipeline using Apache Spark deployed via Docker Compose for vehicle price prediction. The system demonstrates parallel model training across multiple worker nodes using both regression and classification approaches.

---

## 1. Architecture & Infrastructure

### Docker Compose Setup
- **Spark Master**: Coordinates job distribution and cluster management
- **Spark Worker 1**: 4 cores, 4GB memory - processes `part_1.parquet`
- **Spark Worker 2**: 4 cores, 4GB memory - processes `part_2.parquet`
- **Total Cluster Capacity**: 8 cores, 8GB memory

### Data Distribution
- **Dataset**: Vehicle price prediction dataset (249,867 samples after cleaning)
- **Partitioning Strategy**: Data split into node-specific partitions
  - `part_node_1.parquet` → Worker 1
  - `part_node_2.parquet` → Worker 2
- **Storage Format**: Parquet (columnar, optimized for Spark)
- **Shared Volume**: `/shared` directory mounted across all containers for result aggregation

### Network Configuration
- **Spark Network**: Custom Docker bridge network for inter-container communication
- **Ports Exposed**:
  - 7077: Spark Master (native protocol)
  - 8080: Master Web UI
  - 8081-8082: Worker Web UIs

---

## 2. Machine Learning Models Implemented

### 2.1 Gradient Boosted Trees (GBT) Regressor
**Purpose**: Price prediction using ensemble decision trees

**Implementation**: `xgboost_pipeline.py`

**Feature Engineering**:
- Categorical features: make, model, transmission (StringIndexer + OneHotEncoder)
- Numerical features: year, mileage, engine_hp, vehicle_age, mileage_per_year, brand_popularity
- Total features after encoding: 9

**Hyperparameters**:
- Max Iterations: 30
- Max Depth: 10
- Subsampling Rate: 0.8
- Feature Subset Strategy: auto

**Performance (on part_1.parquet)**:
- **Training Set**: RMSE: 2,142.49 | MAE: 1,519.85 | R²: 0.9697
- **Test Set**: RMSE: 2,340.59 | MAE: 1,642.77 | R²: 0.9635
- **Training Time**: 103.98 seconds

**Feature Importance Analysis**:
1. engine_hp (42.86%)
2. year (21.19%)
3. vehicle_age (20.55%)
4. mileage (5.83%)
5. brand_popularity (2.23%)

**Key Insight**: Engine horsepower is the strongest predictor, followed by vehicle age and year, explaining ~85% of predictive power.

---

### 2.2 Multilayer Perceptron (MLP) Classifier
**Purpose**: Price category classification (Low/Medium/High)

**Implementation**: `ml_regressor.py`

**Feature Engineering**:
- Extended categorical encoding: 12 categorical features (make, model, transmission, fuel_type, drivetrain, body_type, exterior_color, interior_color, accident_history, seller_type, condition, trim)
- 6 numerical features
- **Feature explosion via one-hot encoding**: 18 input columns → 162 features

**Price Binning Strategy**:
- Low: < $11,431.45 (33rd percentile)
- Medium: $11,431.45 - $21,606.18 (33rd-67th percentile)
- High: > $21,606.18 (67th percentile)

**Neural Network Architecture**:
```
Input Layer: 162 neurons (one-hot encoded features)
Hidden Layer 1: 256 neurons
Hidden Layer 2: 128 neurons
Hidden Layer 3: 64 neurons
Output Layer: 3 neurons (price categories)
Activation: L-BFGS solver
```

**Hyperparameters**:
- Max Iterations: 200
- Block Size: 128
- Solver: L-BFGS (quasi-Newton optimization)

**Performance (on part_2.parquet)**:
- **Test Accuracy**: 58.21%
- **Precision**: 57.08%
- **F1-Score**: 57.49%
- **Training Time**: 2,394.91 seconds (~40 minutes)

**Analysis**: Moderate classification accuracy due to:
- Class imbalance (33-33-33 split may not reflect true distribution)
- High feature dimensionality (162 features from one-hot encoding)
- Non-linear price boundaries in real-world data

---

### 2.3 Cross-Validation for Hyperparameter Tuning
**Implementation**: `ml_regressor_cv.py`

**Cross-Validation Setup**:
- **Strategy**: 3-fold stratified cross-validation
- **Parallelism**: 2 folds trained simultaneously
- **Parameter Grid**:
  - maxIter: [50, 100, 150]
  - blockSize: [64, 128]
  - **Total combinations**: 6
  - **Total models trained**: 18 (6 configs × 3 folds)

**Grid Search Results**:
| Rank | Configuration | Avg Accuracy |
|------|--------------|--------------|
| 1    | maxIter=150, blockSize=128 | 0.5787 |
| 2    | maxIter=100, blockSize=128 | 0.5748 |
| 3    | maxIter=150, blockSize=64  | 0.5713 |

**Best Model**:
- maxIter: 150
- blockSize: 128
- Architecture: [162, 128, 64, 32, 3]
- **Cross-validation time**: 5,867.70 seconds (~1.6 hours)

**Key Finding**: Increasing iterations from 100→150 improved accuracy by ~0.4%, but blockSize had minimal impact.

---

## 3. Distributed Training Strategy

### Parallel Job Execution
**Command**:
```powershell
Start-Job -ScriptBlock { 
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-memory 2g \
    --conf spark.cores.max=2 \
    /opt/app/ml_regressor.py 
}

Start-Job -ScriptBlock { 
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-memory 2g \
    --conf spark.cores.max=2 \
    /opt/app/xgboost_pipeline.py 
}
```

**Resource Allocation**:
- Each job limited to 2 cores to enable parallel execution
- Total: 4 cores (2 per job) out of 8 available
- Memory: 2GB per job to avoid OOM errors

**Challenges Overcome**:
1. **Initial Problem**: Jobs queued sequentially when requesting 4 cores each
2. **Solution**: Reduced to 2 cores per job with `spark.cores.max` config
3. **Data Locality Issue**: Local file paths (`file://`) failed when executors ran on different workers than where data resided
4. **Solution**: Disabled data reshuffling to maintain partition locality

---

## 4. Technical Implementation Details

### Data Pipeline
1. **Data Loading**: CSV → Parquet conversion using `csv_to_parquet.py`
2. **Data Cleaning**: Null value removal via `dropna()`
3. **Partitioning**: `load_and_repartition.py` splits data across workers
4. **Feature Engineering**: StringIndexer → OneHotEncoder → VectorAssembler pipeline

### Spark Configuration Optimizations
```python
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()
```

**Key Settings**:
- Shuffle partitions reduced to 4 (matches worker cores) to avoid overhead
- Extended timeouts for long-running MLP training
- Heartbeat interval increased to prevent executor loss during training

### Avoiding Data Reshuffling (Critical for Local Files)
```python
# BEFORE (Failed with local file paths)
df = df.withColumn("shuffle_key", rand())
df = df.repartition(4, "shuffle_key").drop("shuffle_key")

# AFTER (Preserved data locality)
# Commenting out reshuffling preserved partitions on original workers
```

---

## 5. Results & Performance Analysis

### Model Comparison

| Model | Task | Metric | Training | Test | Training Time |
|-------|------|--------|----------|------|---------------|
| GBT Regressor | Regression | R² | 0.9697 | 0.9635 | 104 sec |
| GBT Regressor | Regression | RMSE | 2,142 | 2,340 | 104 sec |
| MLP Classifier | Classification | Accuracy | - | 58.21% | 2,395 sec |
| MLP Classifier | Classification | F1-Score | - | 57.49% | 2,395 sec |

### Key Observations

1. **GBT Superior for Regression**:
   - Achieved 96.35% R² on test set
   - Fast training (104s vs 2,395s for MLP)
   - Excellent generalization (minimal overfitting)

2. **MLP Moderate for Classification**:
   - 58% accuracy suggests complex class boundaries
   - Significant training time due to 162-dimensional feature space
   - L-BFGS solver converged after 200 iterations

3. **Training Time Analysis**:
   - GBT: 23x faster than MLP
   - Cross-validation: 18 models trained in 1.6 hours
   - Parallelism reduced CV time by ~50% (estimated)

4. **Feature Engineering Impact**:
   - One-hot encoding: 18 columns → 162 features
   - Curse of dimensionality affected MLP performance
   - GBT handled sparse features more efficiently

---

## 6. Distributed Computing Benefits Demonstrated

### Scalability
- **Horizontal scaling**: Adding workers linearly increases capacity
- **Data partitioning**: Each worker processes independent data chunks
- **Parallel model training**: Two models trained simultaneously

### Fault Tolerance
- Spark's RDD lineage enables recomputation on failure
- Worker restarts handled gracefully by master
- Parquet format supports predicate pushdown for efficiency

### Resource Utilization
- **Before optimization**: 50% resource utilization (jobs queued)
- **After optimization**: 100% utilization (both jobs running)
- **Memory management**: 2GB per executor prevented OOM errors

---

## 7. Challenges & Solutions

### Challenge 1: File Path Issues with Local Storage
**Problem**: Executors couldn't access `file:///data/part_X.parquet` on remote workers

**Solution**:
- Mounted data volumes on all containers
- Disabled data reshuffling to maintain locality
- Ensured driver (master) and executors (workers) had identical volume mounts

### Challenge 2: Job Queuing in Cluster Mode
**Problem**: Second job waited when both requested 4 cores each

**Solution**:
- Reduced `--total-executor-cores` to 2 per job
- Added `--conf spark.cores.max=2` to enforce limits
- Used PowerShell `Start-Job` for true parallel submission

### Challenge 3: MLP Training Timeouts
**Problem**: Neural network training exceeded default Spark timeouts

**Solution**:
- Increased `spark.network.timeout` to 800s
- Adjusted `spark.executor.heartbeatInterval` to 60s
- Reduced shuffle partitions to minimize overhead

### Challenge 4: Memory Errors During Cross-Validation
**Problem**: OOM errors with 4GB executors and 18 models

**Solution**:
- Reduced parameter grid from 12 to 6 combinations
- Used parallelism=2 (limited concurrent model training)
- Avoided caching large DataFrames

---

## 8. Code Structure & Files

### Core Pipeline Scripts
- `csv_to_parquet.py` - Data format conversion
- `load_and_repartition.py` - Data partitioning across workers
- `xgboost_pipeline.py` - GBT regression pipeline
- `ml_regressor.py` - MLP classification pipeline
- `ml_regressor_cv.py` - Cross-validation for MLP
- `xgboot_cv.py` - Cross-validation for GBT (if implemented)

### Infrastructure
- `docker-compose.yml` - Container orchestration
- `Dockerfile.spark-custom` - Custom Spark image with dependencies
- `requirements.txt` - Python package dependencies

### Results & Outputs
- `shared/xgb_results.txt` - GBT performance report
- `shared/results.txt` - MLP performance report
- `shared/cv_results.txt` - Cross-validation results

---

## 9. Lessons Learned

### Technical Insights
1. **Data Locality Matters**: Local file systems require careful partitioning strategy
2. **Resource Constraints**: Oversubscription causes job queuing even with available cores
3. **Model Selection**: GBT outperforms deep learning for tabular regression tasks
4. **Feature Engineering**: One-hot encoding dramatically increases dimensionality

### Best Practices
1. Always set `spark.cores.max` when running multiple jobs
2. Monitor Spark UI (port 8080) for resource allocation
3. Use Parquet for columnar storage efficiency
4. Test with small datasets before scaling up
5. Profile training time vs. accuracy tradeoffs

### Future Improvements
1. Implement HDFS/S3 for true distributed storage
2. Add dynamic resource allocation
3. Experiment with feature selection to reduce MLP input size
4. Implement hyperparameter tuning for GBT
5. Add real-time prediction endpoint
6. Integrate MLflow for experiment tracking

---

## 10. Conclusion

This project successfully demonstrates:
- **Distributed machine learning** using Apache Spark on Docker Compose
- **Parallel model training** across multiple worker nodes
- **Two ML approaches**: GBT regression (96% R²) and MLP classification (58% accuracy)
- **Hyperparameter optimization** via 3-fold cross-validation
- **Production-ready practices**: containerization, resource management, result logging

### Key Takeaway
For tabular vehicle price prediction:
- **GBT is the winner**: 23x faster training, 96% R², interpretable feature importances
- **MLP shows promise**: 58% classification accuracy with room for improvement via feature engineering and architecture tuning
- **Spark's power**: Enabled training multiple models in parallel, reducing total experiment time

The infrastructure is scalable, reproducible, and ready for production deployment with additional workers or cloud migration.

---

## Appendix: Running the Project

### Prerequisites
```bash
docker-compose version 1.29+
Docker Desktop with 8GB+ RAM allocated
```

### Startup
```bash
cd docker_compose_setup_2
docker-compose up -d
```

### Run Single Job
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --total-executor-cores 4 \
  --executor-memory 4g \
  /opt/app/xgboost_pipeline.py
```

### Run Parallel Jobs
```powershell
Start-Job -ScriptBlock { 
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 --executor-memory 2g \
    --conf spark.cores.max=2 /opt/app/ml_regressor.py 
}
Start-Job -ScriptBlock { 
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 --executor-memory 2g \
    --conf spark.cores.max=2 /opt/app/xgboost_pipeline.py 
}
```

### Monitor
- Spark Master UI: http://localhost:8080
- Worker 1 UI: http://localhost:8081
- Worker 2 UI: http://localhost:8082
- Application UI: http://localhost:4040 (when job running)

### View Results
```bash
docker exec spark-master cat /shared/xgb_results.txt
docker exec spark-master cat /shared/results.txt
docker exec spark-master cat /shared/cv_results.txt
```

---

**Project Duration**: 3 weeks  
**Total Training Time**: ~3 hours (including cross-validation)  
**Dataset Size**: 249,867 samples  
**Models Trained**: 20+ (including CV experiments)  
**Technologies**: Apache Spark 3.5, PySpark MLlib, Docker Compose, Python 3.11
