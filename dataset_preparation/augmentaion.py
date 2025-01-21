import os
import random
import cv2
import albumentations as A
from albumentations.core.composition import OneOf
import numpy as np

# Paths for images and labels
base_path = "yolo5_train\dataset_yolo"
images_path = os.path.join(base_path, "images/train")
labels_path = os.path.join(base_path, "labels/train")

# Define augmentations
augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Resize(640, 640),  # Ensure compatibility with YOLOv5 input size
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


# Function to read YOLO labels
def read_yolo_labels(label_path):
    with open(label_path, "r") as file:
        labels = [list(map(float, line.strip().split())) for line in file.readlines()]
    return labels


# Function to write YOLO labels
def write_yolo_labels(output_path, labels):
    with open(output_path, "w") as file:
        for label in labels:
            file.write(" ".join(f"{x:.6f}" for x in label) + "\n")


# Get list of all images in the training folder
all_images = os.listdir(images_path)
random.shuffle(all_images)

# Select 80% of images for augmentation
num_to_augment = int(0.8 * len(all_images))
images_to_augment = all_images[:num_to_augment]

# Process each image and corresponding label
for image_name in images_to_augment:
    # Paths for the image and corresponding label
    image_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + ".txt")

    if not os.path.exists(label_path):
        continue  # Skip if label file doesn't exist

    # Read the image and labels
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    labels = read_yolo_labels(label_path)

    # Convert YOLO labels to normalized bounding boxes
    bboxes = [[label[1], label[2], label[3], label[4]] for label in labels]
    class_labels = [int(label[0]) for label in labels]

    # Apply augmentations
    augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_class_labels = augmented["class_labels"]

    # Prepare augmented labels in YOLO format
    aug_labels = [[cls] + bbox for cls, bbox in zip(aug_class_labels, aug_bboxes)]

    # Save augmented image and labels in the same folders
    output_image_name = f"aug_{image_name}"
    output_label_name = f"aug_{os.path.splitext(image_name)[0]}.txt"

    cv2.imwrite(os.path.join(images_path, output_image_name), aug_image)
    write_yolo_labels(os.path.join(labels_path, output_label_name), aug_labels)

print("Image augmentation and label generation complete.")
