import pandas as pd

# File paths
text_features_path = "outputs\scaled_text_features.csv"
image_features_path = "outputs\image_features.csv"
merged_features_path = "outputs/merged_features.csv"

# Load text and image features
text_features = pd.read_csv(text_features_path)
image_features = pd.read_csv(image_features_path)

# Ensure column names have no spaces
text_features.columns = text_features.columns.str.strip()
image_features.columns = image_features.columns.str.strip()

# Add an "ID" column to text features (if missing)
if "ID" not in text_features.columns:
    text_features.insert(0, "ID", range(1, len(text_features) + 1))  # Assign 1, 2, 3, ...

# Ensure "ID" is integer for merging
text_features["ID"] = text_features["ID"].astype(int)
image_features["ID"] = image_features["ID"].astype(int)

# Merge on 'ID'
merged_features = pd.merge(text_features, image_features, on="ID", how="inner")

# Save merged features
merged_features.to_csv(merged_features_path, index=False)
print(f"✅ Merged features saved to {merged_features_path}")
