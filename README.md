# Lichess Blitz Chess Analytics and Outcome Prediction

## Project Overview

This project explores factors associated with the outcomes of online blitz chess games, with the goal of predicting whether the White player wins. It combines data acquisition, exploratory analysis, data cleaning, feature engineering, machine learning, and visualization using Google Cloud Platform, Python, and PySpark.

**Author:** Victor Chen  
**Course:** CIS 9760 — Big Data Technologies  
**Date:** May 16, 2025

## Dataset

The project uses the [Lichess Blitz Subsample dataset on Kaggle](https://www.kaggle.com/datasets/naddleman/lichess-blitz-subsample).

Games are stored in Portable Game Notation (PGN) format and include player ratings, game results, opening information, time controls, termination details, and moves.

The prediction target is:

- **1:** White wins.
- **0:** White does not win, including Black wins and draws.

## Technologies

- **Cloud infrastructure:** Google Cloud Platform, Google Cloud Storage, Google Cloud Dataproc
- **Data processing:** Python, pandas, python-chess, PySpark
- **Machine learning:** Spark MLlib
- **Visualization:** Matplotlib, Seaborn, scikit-learn
- **Data formats:** PGN and Parquet

## Project Workflow

### 1. Data Acquisition

The dataset was downloaded from Kaggle into a Google Cloud virtual machine, extracted, and copied into the landing folder of a Cloud Storage bucket.

The bucket was organized into five folders:

| Folder | Purpose |
| --- | --- |
| `landing/` | Raw PGN files |
| `cleaned/` | Cleaned data |
| `trusted/` | Data with engineered features |
| `code/` | Project code |
| `models/` | Saved models and evaluation results |

### 2. Exploratory Data Analysis and Cleaning

Python’s chess library and pandas were used to parse PGN files into structured game records. Exploratory analysis examined numeric summaries, missing values, categorical distributions, player ratings, game outcomes, and common openings.

The cleaning code standardized column names, removed rows with missing values, converted data types, and prepared the data for Parquet storage.

The following columns were selected for removal:

- `White`
- `Black`
- `UTCDate`
- `UTCTime`

These fields contain player usernames and timestamps that were not needed for the selected modeling approach. The report documents difficulties saving the cleaned data to Parquet during this milestone.

### 3. Feature Engineering

PySpark code processed the data using a Dataproc cluster.

| Feature Type | Columns | Processing |
| --- | --- | --- |
| Numeric | `whiteelo`, `blackelo`, `nummoves` | Included in the feature vector and standardized |
| Categorical | `eco`, `opening`, `timecontrol`, `termination` | StringIndexer and OneHotEncoder |

VectorAssembler combined the encoded categorical features and numeric columns into a single vector. StandardScaler then produced the model’s feature vector.

### 4. Model Training

A Random Forest classifier was trained to distinguish White wins from other outcomes.

The modeling configuration included:

- **Training/test split:** 80% training and 20% testing
- **Random seed:** 42
- **Cross-validation:** Three folds
- **Number of trees tested:** 20 and 50
- **Maximum depths tested:** 5 and 10
- **Evaluation metric:** Accuracy

The modeling code saved transformed data to the `trusted/` folder and the trained model and evaluation results to the `models/` folder.

## Results and Visualizations

The report states a classification accuracy of **57.58%**, summarized as approximately **57.6%** in its conclusion.

The visualizations examined:

- **Player Elo distributions:** White and Black ratings followed similar distributions, with White players slightly higher rated on average.
- **Game outcomes:** The dataset showed a near-even split between White wins and other outcomes.
- **Elo difference by outcome:** White wins generally corresponded to a greater rating advantage for White.
- **Confusion matrix:** The chart displayed correct predictions and misclassifications for both outcome classes.
- **Numeric feature importance:** BlackElo ranked highest, followed by WhiteElo and the number of moves.

## Future Improvements

The report identifies several opportunities for further exploration:

- Incorporating move-level data
- Analyzing time usage patterns
- Experimenting with deeper models
- Adding player behavior metrics

## Acknowledgments

PGN parsing and exploratory analysis were adapted from the sample notebooks `python_eda_functions_20250331` and `Amazon_Reviews_EDA_Python_20250331`, along with chess PGN code shared by Professor Holowczak.

Feature engineering and modeling were adapted from the sample notebook `Amazon_Reviews_Feature_Engineering_20250506`, also shared by Professor Holowczak.

## Project Report

[Read the full project report](docs/original_project_report.pdf)
