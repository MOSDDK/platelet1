import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

def is_circular(contour):
    """
    Check if a contour is approximately circular.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:  # Avoid division by zero
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 < circularity <= 1.0  # Reverting to original detection approach

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for segmentation
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define size threshold for small objects
    small_threshold = 100  # Reverting to original threshold
    
    small_circular_count = 0  # Initialize counter for small circular objects
    marked_image = image.copy()
    
    # Process each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Count only small circular objects
        if area < small_threshold and is_circular(contour):
            small_circular_count += 1  # Increment the counter
            cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 2)  # Mark detected platelets
    
    return small_circular_count, marked_image

# Streamlit UI
st.title("Platelet Counter")
st.write("Upload multiple images or take photos using your camera.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Camera input for capturing images
camera_image = st.camera_input("Take a picture")

all_images = []

if uploaded_files:
    all_images.extend(uploaded_files)

if camera_image:
    all_images.append(camera_image)

if all_images:
    total_platelet_count = 0
    image_count = len(all_images)
    marked_images = []
    
    for uploaded_file in all_images:
        # Convert the uploaded file or camera input to an OpenCV image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process the image
        platelet_count, marked_image = process_image(image)
        total_platelet_count += platelet_count
        
        # Convert marked image back to RGB for display
        marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
        marked_images.append((uploaded_file.name if hasattr(uploaded_file, 'name') else "Camera Image", marked_image, platelet_count))
    
    # Calculate average count and approximate platelet count
    average_platelet_count = total_platelet_count / image_count
    approximate_platelet_count = average_platelet_count * 15000
    
    # Display results
    for name, marked_image, count in marked_images:
        st.image(marked_image, caption=f"Detected Platelets in {name}", use_column_width=True)
        st.write(f"### Estimated Platelet Count in {name}: {count}")
    
    st.write(f"## Average Platelet Count per Image: {average_platelet_count:.2f}")
    st.write(f"## Approximate Platelet Count: {approximate_platelet_count:.0f}")
