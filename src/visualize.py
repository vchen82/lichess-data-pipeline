from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("ChessDataVisualization") \
    .getOrCreate()
trusted_path = "gs://my-bigdata-project-vc/trusted/chess_trusted_data.parquet"
df = spark.read.parquet(trusted_path)
# If label/prediction are not present, rerun model.transform(df), but assume it's already there
predictions = df.select("whiteelo", "blackelo", "label", "prediction")
#  Convert to Pandas
predictions_pd = predictions.toPandas()
#  ELO Distribution Plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(10, 5))
sns.histplot(predictions_pd["whiteelo"], color="blue", label="White Elo", kde=True)
sns.histplot(predictions_pd["blackelo"], color="red", label="Black Elo", kde=True)
plt.title("Distribution of ELO Ratings (White vs Black)")
plt.legend()
plt.xlabel("ELO Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/tmp/viz1_elo_distribution.png")
# Result Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=predictions_pd)
plt.title("Distribution of Game Results (0 = Not White Win, 1 = White Win)")
plt.xlabel("Game Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("/tmp/viz2_result_distribution.png")
# Elo Difference vs. Outcome Plot 
predictions_pd["elo_diff"] = predictions_pd["whiteelo"] - predictions_pd["blackelo"]
plt.figure(figsize=(8, 5))
sns.boxplot(x="label", y="elo_diff", data=predictions_pd)
plt.title("ELO Difference vs. Game Outcome")
plt.xlabel("White Win? (1=Yes, 0=No)")
plt.ylabel("WhiteElo - BlackElo")
plt.tight_layout()
plt.savefig("/tmp/viz3_elo_diff_vs_outcome.png")
# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(predictions_pd["label"], predictions_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not White Win", "White Win"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix of Model Predictions")
plt.tight_layout()
plt.savefig("/tmp/viz4_confusion_matrix.png")
# Load your model if needed
from pyspark.ml import PipelineModel
model = PipelineModel.load("gs://my-bigdata-project-vc/models/chess_rf_model")
# Extract the best RandomForest model
rf_model = model.stages[-1].bestModel
# Get feature importances (last 3 numeric features: whiteelo, blackelo, nummoves)
importances = rf_model.featureImportances.toArray()
numeric_feats = ["whiteelo", "blackelo", "nummoves"]
feat_df = pd.DataFrame({
    "Feature": numeric_feats,
    "Importance": importances[-3:]
}).sort_values("Importance", ascending=False)
# Plot feature importances
plt.figure(figsize=(6, 4))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance (Numeric Only)")
plt.tight_layout()
plt.savefig("/tmp/viz5_feature_importance.png")
