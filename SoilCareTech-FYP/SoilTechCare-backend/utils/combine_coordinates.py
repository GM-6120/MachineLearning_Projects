import pandas as pd

# Load feature dataset (without coordinates)
df_features = pd.read_csv("outputs\merged_features.csv")

# Load coordinates dataset (must contain 'Latitude' & 'Longitude')
df_coords = pd.read_csv("outputs\coordinates.csv")

# Ensure both datasets have the same row count
if len(df_features) == len(df_coords):
    df_merged = pd.concat([df_coords, df_features], axis=1)  # Merge side by side
    df_merged.to_csv("outputs\merged_features.csv", index=False)
    print("✅ Coordinates added successfully!")
else:
    print("⚠️ Error: Mismatch in row counts between features & coordinates.")
