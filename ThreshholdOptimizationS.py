import cv2
import numpy as np

def nothing(x):
    pass

def apply_hsv_mask(image, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the mask
    lower_bound = np.array([hue_min, sat_min, val_min])
    upper_bound = np.array([hue_max, sat_max, val_max])

    # Create the mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Load the image
image_path = "C:\\Users\\User\\Desktop\HSV Imagesss\\S4.png"  # Use raw string for the path
print(f"Loading image from: {image_path}")  # Debug print
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for hue, saturation, and value ranges
    cv2.createTrackbar('Hue Min', 'image', 0, 179, nothing)
    cv2.createTrackbar('Hue Max', 'image', 179, 179, nothing)
    cv2.createTrackbar('Sat Min', 'image', 0, 255, nothing)
    cv2.createTrackbar('Sat Max', 'image', 255, 255, nothing)
    cv2.createTrackbar('Val Min', 'image', 0, 255, nothing)
    cv2.createTrackbar('Val Max', 'image', 255, 255, nothing)

    while(1):
        # Get current positions of trackbars
        h_min = cv2.getTrackbarPos('Hue Min', 'image')
        h_max = cv2.getTrackbarPos('Hue Max', 'image')
        s_min = cv2.getTrackbarPos('Sat Min', 'image')
        s_max = cv2.getTrackbarPos('Sat Max', 'image')
        v_min = cv2.getTrackbarPos('Val Min', 'image')
        v_max = cv2.getTrackbarPos('Val Max', 'image')

        # Apply the HSV mask
        masked_image = apply_hsv_mask(image, h_min, h_max, s_min, s_max, v_min, v_max)

        # Show the masked image
        cv2.imshow('image', masked_image)

        # Wait for the 'esc' key to be pressed to exit
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Destroy all windows
    cv2.destroyAllWindows()
