import os
import rasterio
import numpy as np
from PIL import Image
from skimage import exposure

# Function to read a specific band of a Sentinel-2 image
def read_band(image_path, band_number):
    with rasterio.open(image_path) as src:
        band = src.read(band_number)  # Reading the specific band (1-indexed)
        return band

# Function to preprocess the image
def preprocess_image(image_path, output_path, resize_dim=(256, 256)):
    print(f"Processing {image_path}...")

    # Read the individual bands (B2, B3, B4)
    b2 = read_band(image_path, 1)  # Blue band (B2)
    b3 = read_band(image_path, 2)  # Green band (B3)
    b4 = read_band(image_path, 3)  # Red band (B4)

    # Handle NaN values by replacing them with 0
    b2 = np.nan_to_num(b2, nan=0)
    b3 = np.nan_to_num(b3, nan=0)
    b4 = np.nan_to_num(b4, nan=0)

    # Apply histogram equalization
    b2 = exposure.equalize_hist(b2) * 255  # Scale to 0-255
    b3 = exposure.equalize_hist(b3) * 255  # Scale to 0-255
    b4 = exposure.equalize_hist(b4) * 255  # Scale to 0-255

    # Stack bands into an RGB image
    rgb_image = np.stack([b4, b3, b2], axis=-1)  # Red, Green, Blue

    # Convert the numpy array to a PIL image
    rgb_image_pil = Image.fromarray(rgb_image.astype(np.uint8))

    # Resize the image to the specified dimensions
    resized_image = rgb_image_pil.resize(resize_dim, Image.Resampling.LANCZOS)

    # Save the resized image (PNG format)
    resized_image.save(output_path, format='PNG')

# Function to process all images in a folder
def process_all_images(input_folder, output_folder, resize_dim=(256, 256)):
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # Only process .tif files
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"processed_{filename.split('.')[0]}.png")

            preprocess_image(input_image_path, output_image_path, resize_dim=resize_dim)

# Specify your input and output folders
input_folder = r'D:\projects\SoilTechCare-backend\dataset\images\raw_images'
output_folder = r'D:\projects\SoilTechCare-backend\dataset\images\pre_processed_images'

# Define the desired image size (e.g., 224x224 for deep learning models)
resize_dim = (224, 224)

# Process all images in the folder
process_all_images(input_folder, output_folder, resize_dim=resize_dim)
