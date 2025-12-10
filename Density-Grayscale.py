import os
import cv2
import numpy as np

def calculate_average_grayscale(image_path):
    """
    Read a single image in grayscale and return its average pixel value.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return np.mean(image)

def batch_calculate_average_grayscale(folder_path):
    """
    Process all images in a folder and compute the average grayscale value
    for each image.
    
    Supported formats: png, jpg, jpeg, bmp, tiff.
    Returns a dictionary {filename: average_grayscale_value}.
    """
    files = os.listdir(folder_path)

    # Filter image files by extension
    image_files = [f for f in files 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    results = {}

    # Compute grayscale average for each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            avg_gray = calculate_average_grayscale(image_path)
            results[image_file] = avg_gray
        except ValueError as e:
            print(e)  # Skip unreadable images
    
    return results

# Example usage
folder_path = " "  # Replace with your folder path
average_grayscale_values = batch_calculate_average_grayscale(folder_path)

# Write results to a text file
output_path = " "  # Replace with your output file path
with open(output_path, "w") as file:
    # Write average grayscale for each image
    for image_name, avg_gray in average_grayscale_values.items():
        file.write(f"{image_name}: {avg_gray}\n")
    
    # Write summary statistics
    file.write(f"Total images processed: {len(average_grayscale_values)}\n")
    file.write(f"Overall average grayscale: {np.mean(list(average_grayscale_values.values()))}\n")
