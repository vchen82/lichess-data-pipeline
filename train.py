from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ChessFeatureEngineeringModeling") \
    .getOrCreate()
# Load cleaned data
input_path = "gs://my-bigdata-project-vc/cleaned/lichess_blitz_2013_01_combined_cleaned.parquet"
df = spark.read.parquet(input_path)
# Label: 1 if White won, 0 otherwise
df = df.withColumn("label", when(df["result"] == "1-0", 1).otherwise(0))
# Feature Engineering
# Categorical features
categorical_cols = ["eco", "opening", "timecontrol", "termination"]
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
    for col in categorical_cols
]
encoders = [
    OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
    for col in categorical_cols
]
# Numeric features
numeric_cols = ["whiteelo", "blackelo", "nummoves"]
# Assemble all features
assembler_inputs = [f"{col}_vec" for col in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_features")
# Standardize
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")
# Model Setup
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    seed=42
)
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
# CrossValidator setup
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)
# Full pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, cv])
# Train/Test Split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
# Fit pipeline
model = pipeline.fit(train_df)
#  Evaluation
predictions = model.transform(test_df)
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.4f}")
# Save Outputs
with open("/tmp/evaluation_results.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n")
# Save engineered data to trusted zone
trusted_path = "gs://my-bigdata-project-vc/trusted/chess_trusted_data.parquet"
transformed_df = model.transform(df)
transformed_df.write.mode("overwrite").parquet(trusted_path)
# overwrite previous files and save the model
import subprocess
subprocess.run([
    "gsutil", "-m", "rm", "-r", "gs://my-bigdata-project-vc/models/chess_rf_model"
])
model_path = "gs://my-bigdata-project-vc/models/chess_rf_model"
model.save(model_path)
# Upload evaluation result to GCS
import subprocess
subprocess.run([
    "gsutil", "cp", "/tmp/evaluation_results.txt", "gs://my-bigdata-project-vc/models/evaluation_results.txt"
])
spark.stop()
