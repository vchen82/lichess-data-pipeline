# Lichess Blitz Chess Analytics and Outcome Prediction

**CIS 9760 — Big Data Technologies | May 2025**

## Project overview

This project explores historical blitz chess games from the [Lichess Blitz Subsample on Kaggle](https://www.kaggle.com/datasets/naddleman/lichess-blitz-subsample). Its goal is to examine factors associated with game outcomes and classify whether White won (`1`) or did not win (`0`). The project uses Google Cloud Platform for data storage and processing, Python for PGN parsing and exploratory data analysis, and PySpark MLlib for feature engineering and Random Forest classification.

## Technologies

- **Cloud:** Google Cloud Storage, Google Cloud Dataproc
- **Data processing:** Python, pandas, `python-chess`, PySpark, Apache Spark
- **Machine learning:** Spark MLlib (Random Forest, cross-validation)
- **Visualization:** Matplotlib, Seaborn, scikit-learn
- **Data formats:** PGN, Parquet

## Project workflow

```text
Kaggle Lichess PGN files
         |
         v
Google Cloud Storage: landing/     Raw PGN input
         |
         v
Python + pandas                    Parse chess games and perform EDA
         |
         v
Data cleaning / Parquet
         |
         v
Google Cloud Storage: cleaned/     Structured training input
         |
         v
Dataproc + PySpark MLlib           Encode, assemble and scale features
         |                         Train and tune Random Forest
         +-----------------------> GCS trusted/  Engineered data
         +-----------------------> GCS models/   Saved model and evaluation
         |
         v
Visualizations                     Elo ratings, game outcomes,
                                    confusion matrix, feature importance
```

### Data preparation and exploratory analysis

PGN files contain game metadata, including player Elo ratings, opening names, game results, time controls, and termination details. Python and `python-chess` are used to parse the games into tabular records. pandas is used for exploratory summaries, cleaning, and preparation of structured data.

### Feature engineering and model training

The Spark pipeline uses the following features:

| Feature type | Fields | Transformation |
| --- | --- | --- |
| Numeric | `whiteelo`, `blackelo`, `nummoves` | Vector assembly and standard scaling |
| Categorical | `eco`, `opening`, `timecontrol`, `termination` | StringIndexer and OneHotEncoder |
| Label | Game result | `1` for White win, `0` otherwise |

A Random Forest classifier is fitted using an 80/20 training/test split (`seed=42`), with three-fold cross-validation over 20/50 trees and depths of 5/10.

> **Model interpretation:** Because `nummoves` and `termination` are known after a game ends, this model is best interpreted as retrospective outcome classification rather than a pregame forecasting model.

## Results

The May 2025 project report gives an accuracy of **57.58%** for the classifier. The confusion matrix shown in the same report contains 190 and 185 correct predictions and 37 and 50 incorrect predictions, equivalent to approximately **81.17%** accuracy for the cases in that chart. These figures are not directly consistent; the report does not establish that the metric and chart use the same evaluation sample. They are presented here as documented, without claiming a newly verified benchmark.

The project also includes visualizations of Elo rating distributions, game outcomes, Elo difference by outcome, confusion matrix, and numeric feature importance. See the [original project report](docs/original_project_report.docx) for charts, explanations, and milestone details.

## Repository structure

```text
lichess-chess-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── setup_gcp.sh
│   └── create_dataproc_cluster.sh
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   └── visualize.py
└── docs/
    ├── original_project_report.docx
    └── report_embedded_images/
```

## Code and setup notes

The scripts correspond to the original project's GCP setup, exploratory analysis, cleaning, training, and visualization stages. They retain original project-specific Cloud Storage paths and resource names; the preprocessing file also depends on an existing `df` DataFrame. **The repository is not supplied as a tested, one-command runnable deployment.** Review and adapt resource references and preprocessing inputs before attempting to run it. Creating GCP resources can incur charges.

Raw PGN data and Spark model binaries are not included. The dataset is available from the [Kaggle source](https://www.kaggle.com/datasets/naddleman/lichess-blitz-subsample). Dependency versions were not specified in the project report, so `requirements.txt` lists relevant libraries without reproducing an exact historical environment.

## Acknowledgments

PGN parsing/EDA and feature engineering code drew on course-provided examples and instructional notebooks, as credited in the [May 2025 project report](docs/original_project_report.docx).
