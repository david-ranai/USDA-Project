import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.ndimage import label
import cv2

# Define thresholds for filtering
P1, P2, P3 = 1.05, 1.1, 1.7
epsilon = 1e-5  # Small value to prevent division by zero

# HSV Ranges for each species
species_hsv = {
    1: {'hue_min': 33, 'hue_max': 80, 'sat_min': 25, 'sat_max': 211, 'val_min': 54, 'val_max': 217},
    2: {'hue_min': 28, 'hue_max': 80, 'sat_min': 13, 'sat_max': 216, 'val_min': 64, 'val_max': 215},
    3: {'hue_min': 35, 'hue_max': 58, 'sat_min': 37, 'sat_max': 115, 'val_min': 53, 'val_max': 179},
    4: {'hue_min': 18, 'hue_max': 88, 'sat_min': 30, 'sat_max': 124, 'val_min': 48, 'val_max': 128},
    5: {'hue_min': 36, 'hue_max': 64, 'sat_min': 68, 'sat_max': 236, 'val_min': 37, 'val_max': 208},
    6: {'hue_min': 25, 'hue_max': 59, 'sat_min': 42, 'sat_max': 217, 'val_min': 31, 'val_max': 238},
    7: {'hue_min': 75, 'hue_max': 107, 'sat_min': 44, 'sat_max': 147, 'val_min': 108, 'val_max': 255},
    8: {'hue_min': 57, 'hue_max': 99, 'sat_min': 13, 'sat_max': 113, 'val_min': 98, 'val_max': 249}
}

# Function to calculate the green percentage using combined Canopeo and HSV algorithms
def calculate_green_percentage(image, hsv_values):
    red_band = image[0]
    green_band = image[1]
    blue_band = image[2]

    # Canopeo mask
    canopeo_mask = (
        ((red_band / (green_band + epsilon)) < P1) &
        ((red_band / (green_band + epsilon)) < P2) &
        (2 * green_band - red_band - blue_band > P3)
    )

    # Convert image to RGB and then to HSV
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1).astype(np.uint8)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # HSV mask
    hue, sat, val = cv2.split(hsv_image)
    hsv_mask = (
        (hue >= hsv_values['hue_min']) & (hue <= hsv_values['hue_max']) &
        (sat >= hsv_values['sat_min']) & (sat <= hsv_values['sat_max']) &
        (val >= hsv_values['val_min']) & (val <= hsv_values['val_max'])
    )

    # Combine Canopeo and HSV masks
    combined_mask = canopeo_mask & hsv_mask

    # Optional: Refine the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Calculate the percentage of green pixels
    green_percentage = np.sum(refined_mask) / refined_mask.size

    return green_percentage, refined_mask, rgb_image

# Function to extract green percentage for each plot
def extract_green_percentage(dates, shapefile_base_dir, raster_base_dir):
    data = []
    images = {}
    
    for date in dates:
        shapefile_dir = os.path.join(shapefile_base_dir, f'.SHP {date}')
        shapefile_files = [f'Shapefile.shp', f'Shapefile.shx', f'Shapefile.dbf', f'Shapefile.prj']
        shapefile_paths = [os.path.join(shapefile_dir, f) for f in shapefile_files]
        shapefile_exists = all(os.path.exists(p) for p in shapefile_paths)
        
        if not shapefile_exists:
            print(f"Shapefile components missing for date: {date}")
            continue
        
        shapefile_path = os.path.join(shapefile_dir, 'Shapefile.shp')
        raster_path = os.path.join(raster_base_dir, f'{date}.tif')
        
        plots = gpd.read_file(shapefile_path)
        
        with rasterio.open(raster_path, mode='r') as ds:
            for _, row in plots.iterrows():
                plot_id = row['Treatment']
                species_id = row['Species']
                hsv_values = species_hsv.get(int(species_id), species_hsv[1])  # Default to species 1 if not found
                print(f"Processing plot {plot_id} with Species ID: {species_id} (type: {type(species_id)})")
                geom = row['geometry']
                out_img, out_transform = mask(ds, [geom], crop=True)
                
                green_percentage, mask_canopeo, rgb_image = calculate_green_percentage(out_img, hsv_values)
                
                if not np.isnan(green_percentage):
                    data.append({
                        'date': date,
                        'plot_id': plot_id,
                        'plant_id': int(species_id),
                        'green_percentage': green_percentage
                    })
                    
                    images[(plot_id, date)] = (rgb_image, mask_canopeo)
    
    df = pd.DataFrame(data)
    return df, images

# Function to visualize original and masked images
def visualize_original_vs_masked(image, mask, plot_id, date):
    output_image = image.copy()
    output_image[mask == 1] = [0, 255, 0]  # Highlight green pixels in green color
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Masked Image')
    plt.imshow(output_image)
    plt.axis('off')

    plt.suptitle(f'Plot {plot_id} on {date}')
    plt.show()

# Function to create sequences for time series prediction
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to train separate models for each species
def train_models(df, n_steps):
    species_models = {}
    
    for species_id in df['plant_id'].unique():
        species_df = df[df['plant_id'] == species_id].sort_values(by='date')
        
        green_percentages = species_df['green_percentage'].values
        X, y = create_sequences(green_percentages, n_steps)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(n_steps,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)
        
        # Plot training and validation loss
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss for Species {species_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        species_models[species_id] = model
    
    return species_models

# Function to predict and visualize results for a specific plot
def predict_for_plot(df, images, plot_id, n_steps, future_dates):
    plot_df = df[df['plot_id'] == plot_id].sort_values(by='date')
    
    if plot_df.empty:
        print(f"Plot ID {plot_id} not found in data.")
        return
    
    species_id = plot_df['plant_id'].iloc[0]
    model = species_models.get(species_id, None)
    
    if model is None:
        print(f"No model found for species ID {species_id}.")
        return
    
    green_percentages = plot_df['green_percentage'].values
    X, _ = create_sequences(green_percentages, n_steps)
    
    future_predictions = []
    last_sequence = green_percentages[-n_steps:]
    for future_date in future_dates:
        pred = model.predict(last_sequence.reshape(1, n_steps)).flatten()
        future_predictions.append(pred[0])
        last_sequence = np.append(last_sequence[1:], pred[0])
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_df['date'], green_percentages[:len(plot_df)], label='Actual')
    plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Green Percentage')
    plt.title(f'Growth Prediction for Plot {plot_id}')
    plt.legend()
    plt.show()
    
    for date in plot_df['date']:
        date_str = date.strftime('%m%d%Y')
        if (plot_id, date_str) in images:
            image, mask = images[(plot_id, date_str)]
            visualize_original_vs_masked(image, mask, plot_id, date_str)

# Define paths and dates
dates = ['05182023', '06082023', '06212023', '07052023', '07212023', '08022023', '08192023','08262023','10022023','10202023','09102023', '11072023','12212023']
shapefile_base_dir = 'C:\\Users\\User\\Desktop\\USDA\\SHP Complete\\'  
raster_base_dir = 'C:\\Users\\User\\Desktop\\USDA\\USDA PHOTO\\'  

green_percentage_df, images = extract_green_percentage(dates, shapefile_base_dir, raster_base_dir)

green_percentage_df['date'] = pd.to_datetime(green_percentage_df['date'], format='%m%d%Y')

n_steps = 3 
species_models = train_models(green_percentage_df, n_steps)

future_dates = pd.to_datetime(['2023-12-19', '2023-12-26', '2024-01-04'])  

while True:
    plot_id = input("Enter the plot ID (e.g., P1, P2, etc.) or 'quit' to exit: ")
    if plot_id.lower() == 'quit':
        break
    predict_for_plot(green_percentage_df, images, plot_id, n_steps, future_dates)
