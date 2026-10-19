from google.cloud import storage
from io import StringIO
import pandas as pd
import chess
import chess.pgn
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style
sns.set(style="whitegrid")
def perform_EDA_numeric(df, filename=None):
    numeric_df = df.select_dtypes(include='number')
    summary_df = numeric_df.describe().T
    summary_df['missing_%'] = 100 * numeric_df.isna().sum() / len(numeric_df)
    if filename:
        summary_df.to_csv(f"numeric_summary_{filename}.csv")
    return summary_df
def perform_EDA_categorical(df, filename=None, include_columns=[]):
    categorical_df = df.select_dtypes(include='object')
    if include_columns:
        categorical_df = categorical_df[include_columns]
    summary = {}
    for col in categorical_df.columns:
        value_counts = categorical_df[col].value_counts(dropna=False)
        summary[col] = {
            "num_missing": categorical_df[col].isna().sum(),
            "cardinality": categorical_df[col].nunique(),
            "top_values": value_counts.head(5).to_dict()
        }
    summary_df = pd.DataFrame(summary).T
    if filename:
        summary_df.to_csv(f"categorical_summary_{filename}.csv")
    return summary_df
def games_to_dataframe(games):
    data = []
    for game in games:
        headers = game.headers
        data.append({
            "White": headers.get("White"),
            "Black": headers.get("Black"),
            "WhiteElo": int(headers.get("WhiteElo", 0)) if headers.get("WhiteElo") else None,
            "BlackElo": int(headers.get("BlackElo", 0)) if headers.get("BlackElo") else None,
            "Result": headers.get("Result"),
            "ECO": headers.get("ECO"),
            "Opening": headers.get("Opening"),
            "TimeControl": headers.get("TimeControl"),
            "UTCDate": headers.get("UTCDate"),
            "UTCTime": headers.get("UTCTime"),
            "Termination": headers.get("Termination"),
            "NumMoves": len(list(game.mainline_moves()))
        })
    return pd.DataFrame(data)
# Visualization functions
def plot_distributions(df):
    # WhiteElo histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(df['WhiteElo'].dropna(), bins=30, kde=True)
    plt.title("Distribution of White Elo")
    plt.xlabel("White Elo")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    # BlackElo histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(df['BlackElo'].dropna(), bins=30, kde=True)
    plt.title("Distribution of Black Elo")
    plt.xlabel("Black Elo")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    # Result count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Result', data=df, order=df['Result'].value_counts().index)
    plt.title("Game Result Distribution")
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    # Top10 openings
    plt.figure(figsize=(10, 5))
    top_openings = df['Opening'].value_counts().head(10)
    sns.barplot(x=top_openings.values, y=top_openings.index)
    plt.title("Top 10 Most Common Openings")
    plt.xlabel("Count")
    plt.ylabel("Opening")
    plt.tight_layout()
    plt.show()
    # Elo boxplot by result
    plt.figure(figsize=(8, 4))
    df_melted = df.melt(id_vars='Result', value_vars=['WhiteElo', 'BlackElo'], var_name='Player', value_name='Elo')
    sns.boxplot(x='Result', y='Elo', hue='Player', data=df_melted)
    plt.title("Elo Distribution by Game Result")
    plt.tight_layout()
    plt.show()
def main_chess():
    bucket_name = 'my-bigdata-project-vc'
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix="landing/lichess_blitz_subsample_2013-01")
    pgn_blobs = [blob for blob in blobs if blob.name.endswith('.pgn')]
    for blob in pgn_blobs:
        print(f"File: {blob.name} | Size: {blob.size} bytes | Created: {blob.time_created}")
        pgn_content = blob.download_as_string()
        pgn_text = pgn_content.decode('utf-8')
        games = []
        pgn_io = StringIO(pgn_text)
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            games.append(game)
        print(f"Parsed {len(games)} games from this file.")
        df = games_to_dataframe(games)
        print("\n--- Numeric EDA ---")
        numeric_summary_df = perform_EDA_numeric(df, filename=blob.name.replace("/", "_"))
        print(numeric_summary_df)
        print("\n--- Categorical EDA ---")
        categorical_summary_df = perform_EDA_categorical(df, filename=blob.name.replace("/", "_"))
        print(categorical_summary_df)
        print("\n--- Generating Visualizations ---")
        plot_distributions(df)
        break  # Remove to process all files
main_chess()
