import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load pre-trained EfficientNetB0 model (without classification layer)
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Image dataset paths
image_folder = "dataset\images\preprocessed_images"
image_features_path = "outputs/image_features.csv"

# Extract features for each image
features_list = []
image_ids = []

for img_name in os.listdir(image_folder):
    if img_name.endswith(".png"):  # Ensure only images are processed
        img_path = os.path.join(image_folder, img_name)
        
        # Load & preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = model.predict(img_array).flatten()
        
        features_list.append(features)
        
        # Extract numeric ID using regex
        match = re.search(r'\d+', img_name)  # Extracts the first number found in filename
        if match:
            image_ids.append(int(match.group()))
        else:
            print(f"⚠ Warning: Could not extract numeric ID from {img_name}, skipping.")
            continue

# Convert to DataFrame & save
features_df = pd.DataFrame(features_list)
features_df.insert(0, "ID", image_ids)  # Add image IDs as integers

# Save CSV
features_df.to_csv(image_features_path, index=False)
print(f"✅ Image features saved to {image_features_path}")
