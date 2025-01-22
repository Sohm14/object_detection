import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os


# Define the source directories
source_dir = "yolo5_train\dataset_yolo\images"
train_dir = os.path.join(source_dir, "train")
test_dir = os.path.join(source_dir, "test")
val_dir = os.path.join(source_dir, "val")

# Define the destination directory
destination_dir = "yolo5_train\dataset_yolo\images"
destination_train_dir = os.path.join(destination_dir, "train")
destination_test_dir = os.path.join(destination_dir, "test")
destination_val_dir = os.path.join(destination_dir, "val")

# Create destination directories if they do not exist
os.makedirs(destination_train_dir, exist_ok=True)
os.makedirs(destination_val_dir, exist_ok=True)


def process_and_save_images(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, os.path.relpath(src_path, src_dir))

            # Ensure the destination subdirectory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            try:
                exposure = file.split("_")[5].split(".")[0]
                Sun_azimuth = file.split("_")[3].split(".")[0]
            except IndexError:
                print(f"Skipping file with unexpected format: {file}")
                continue

            # Get file extension in uppercase
            file_extension = file.split(".")[-1].upper()

            try:
                if exposure in ["0016", "0032", "0064", "0128"] or (
                    Sun_azimuth == "270"
                    and exposure in ["0016", "0032", "0064", "0128", "0256"]
                ):
                    # Process with PIL
                    with Image.open(src_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Save in the same format
                        img.save(dest_path, format=file_extension)

                    if Sun_azimuth == "270":
                        image = cv2.imread(dest_path)
                        if image is None:
                            raise ValueError("Failed to read image with OpenCV.")

                        # Convert to grayscale and equalize histogram
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        clahe_image = clahe.apply(gray)

                        # Save with OpenCV in the same format
                        if file_extension == "JPG":
                            file_extension = (
                                "JPEG"  # OpenCV uses 'JPEG' instead of 'JPG'
                            )
                        cv2.imwrite(dest_path, clahe_image)

                else:
                    # Process with OpenCV
                    image = cv2.imread(src_path)
                    if image is None:
                        raise ValueError("Failed to read image with OpenCV.")

                    # Convert to grayscale and equalize histogram
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    equalized = cv2.equalizeHist(gray)

                    # Save with OpenCV in the same format
                    if file_extension == "JPG":
                        file_extension = "JPEG"  # OpenCV uses 'JPEG' instead of 'JPG'
                    rgb_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(dest_path, rgb_image)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")


# Process train and val directories
process_and_save_images(train_dir, destination_train_dir)
process_and_save_images(test_dir, destination_test_dir)
process_and_save_images(val_dir, destination_val_dir)

print("Image conversion completed.")
