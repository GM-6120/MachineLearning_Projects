import ee
import os
import requests
import pandas as pd

def initialize_gee():
    """Initialize Google Earth Engine (GEE) once."""
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except ee.EEException as e:
        print(f"GEE Initialization Error: {e}")
        raise

def fetch_image(coordinates, output_folder="dataset/images", max_cloud_cover=50):
    """Fetch satellite images for the given coordinates."""
    try:
        aoi = ee.Geometry.Point(coordinates)

        # Define the Landsat 8 image collection
        collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
            .filterBounds(aoi) \
            .filterDate("2023-01-01", "2023-12-31") \
            .sort("CLOUD_COVER") \
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))

        # Fetch the first image from the collection
        image = collection.first()

        if image is None:
            print(f"No image found for coordinates: {coordinates} (no images in collection).")
            return None

        image_info = image.getInfo()  # Check if the image has metadata
        if not image_info:
            print(f"No image metadata found for coordinates: {coordinates}.")
            return None

        # Clip the image to the region of interest (aoi)
        image = image.clip(aoi)

        # Define the URL for downloading the image (True color RGB bands)
        url = image.getThumbURL({
            "region": aoi,
            "dimensions": "256x256",  # Resolution in pixels
            "format": "png",
            "bands": ["SR_B4", "SR_B3", "SR_B2"]  # True-color visualization (Red, Green, Blue)
        })

        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        image_path = os.path.join(output_folder, f"{coordinates[0]}_{coordinates[1]}.png")

        # Make the request to download the image
        response = requests.get(url)

        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved at: {image_path}")
            return image_path
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error fetching image for {coordinates}: {e}")
        return None

def fetch_images_for_dataset(dataset_path="dataset/soil_data.csv", output_folder="dataset/images"):
    """Fetch images for each set of coordinates in the dataset."""
    try:
        # Read the dataset (soil_data.csv) into a DataFrame
        df = pd.read_csv(dataset_path)

        # Check if 'Geo-coordinates' column exists
        if 'Geo-coordinates' not in df.columns:
            print("'Geo-coordinates' column not found in the dataset.")
            return

        # Iterate over each row in the dataset to fetch images
        for index, row in df.iterrows():
            coordinates_str = row['Geo-coordinates']
            
            # Ensure the coordinates are in the form of (latitude, longitude)
            try:
                coordinates = eval(coordinates_str)  # Convert string to tuple (latitude, longitude)
                print(f"Fetching image for coordinates: {coordinates}")
                result = fetch_image(coordinates, output_folder)
                if not result:
                    print(f"No valid image downloaded for coordinates: {coordinates}")
            except Exception as e:
                print(f"Invalid coordinates at index {index}: {coordinates_str}. Error: {e}")

    except Exception as e:
        print(f"Error fetching images for dataset: {e}")

if __name__ == "__main__":
    # Initialize GEE only once before processing the dataset
    try:
        initialize_gee()
        # Fetch images for the dataset
        fetch_images_for_dataset()
    except Exception as e:
        print(f"An error occurred during initialization or fetching images: {e}")
