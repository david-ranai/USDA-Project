import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate and apply the mask based on current HSV thresholds
def apply_mask(image, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv_image)
    hsv_mask = (
        (hue >= hue_min) & (hue <= hue_max) &
        (sat >= sat_min) & (sat <= sat_max) &
        (val >= val_min) & (val <= val_max)
    )
    return hsv_mask

# Function to update the mask and display the result with the trackbar values
def update_mask(val):
    hue_min = cv2.getTrackbarPos('Hue Min', 'Masked Image')
    hue_max = cv2.getTrackbarPos('Hue Max', 'Masked Image')
    sat_min = cv2.getTrackbarPos('Sat Min', 'Masked Image')
    sat_max = cv2.getTrackbarPos('Sat Max', 'Masked Image')
    val_min = cv2.getTrackbarPos('Val Min', 'Masked Image')
    val_max = cv2.getTrackbarPos('Val Max', 'Masked Image')

    mask = apply_mask(image_copy, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    output_image = image_copy.copy()
    output_image[mask == 1] = [0, 255, 0]  # Highlight the selected pixels in green

    cv2.imshow('Masked Image', output_image)

# Function to visualize original and masked images with HSV trackbars
def visualize_original_vs_masked(image, plot_id, date):
    global image_copy
    image_copy = image.copy()
    
    cv2.namedWindow('Masked Image')
    
    # Create trackbars for adjusting HSV values
    cv2.createTrackbar('Hue Min', 'Masked Image', 24, 179, update_mask)
    cv2.createTrackbar('Hue Max', 'Masked Image', 103, 179, update_mask)
    cv2.createTrackbar('Sat Min', 'Masked Image', 23, 255, update_mask)
    cv2.createTrackbar('Sat Max', 'Masked Image', 225, 255, update_mask)
    cv2.createTrackbar('Val Min', 'Masked Image', 0, 255, update_mask)
    cv2.createTrackbar('Val Max', 'Masked Image', 255, 255, update_mask)

    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    update_mask(0)  # Initial call to update the mask
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage in your existing workflow
# Assuming `images` dictionary is already populated
while True:
    plot_id = input("Enter the plot ID (e.g., P1, P2, etc.) or 'quit' to exit: ")
    if plot_id.lower() == 'quit':
        break
    date = '05182023'  # Example date, use appropriate date as needed
    if (plot_id, date) in images:
        image, _ = images[(plot_id, date)]
        visualize_original_vs_masked(image, plot_id, date)
