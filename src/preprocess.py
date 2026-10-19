df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
    .str.lower()
)
columns_to_drop = ['utcdate', 'utctime', 'white', 'black']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df_cleaned = df.dropna()
df_cleaned['whiteelo'] = df_cleaned['whiteelo'].astype(int)
df_cleaned['blackelo'] = df_cleaned['blackelo'].astype(int)
df_cleaned['nummoves'] = df_cleaned['nummoves'].astype(int)
for col in ['result', 'eco', 'opening', 'timecontrol', 'termination']:
    df_cleaned[col] = df_cleaned[col].astype('category')
df_cleaned.to_parquet('./cleaned/chess_cleaned_data.parquet', index=False)
