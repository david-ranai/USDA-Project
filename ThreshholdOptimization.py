import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_hsv_histogram_and_mask(image_path, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the HSV range for the mask
    lower_bound = np.array([hue_min, sat_min, val_min])
    upper_bound = np.array([hue_max, sat_max, val_max])

    # Create the mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original and masked images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Masked Image')
    plt.imshow(masked_image)
    plt.axis('off')

    plt.show()

# Example usage
image_path = "C:\\Users\\User\\Desktop\\HSV Imagesss\\S3.png"

# Further adjusted HSV ranges for species 3 to capture green areas
hue_min, hue_max = 35, 90  # Focusing on green hues
sat_min, sat_max = 30, 170
val_min, val_max = 50, 200

calculate_hsv_histogram_and_mask(image_path, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
