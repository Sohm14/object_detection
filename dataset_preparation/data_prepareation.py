import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your dataset
image_root = "original_data_set"
label_root = "POLAR-Sim\Labels"
output_dir = "yolo5_train\dataset_yolo"

# Output structure
output_images_train = os.path.join(output_dir, "images/train")
output_images_val = os.path.join(output_dir, "images/val")
output_images_test = os.path.join(output_dir, "images/test")
output_labels_train = os.path.join(output_dir, "labels/train")
output_labels_val = os.path.join(output_dir, "labels/val")
output_labels_test = os.path.join(output_dir, "labels/test")

# Create output directories
os.makedirs(output_images_train, exist_ok=True)
os.makedirs(output_images_val, exist_ok=True)
os.makedirs(output_images_test, exist_ok=True)
os.makedirs(output_labels_train, exist_ok=True)
os.makedirs(output_labels_val, exist_ok=True)
os.makedirs(output_labels_test, exist_ok=True)

# Collect all image-label pairs
image_label_pairs = []
for terrain_folder in os.listdir(image_root):
    image_folder = os.path.join(image_root, terrain_folder)
    label_folder = os.path.join(label_root, terrain_folder)

    if os.path.isdir(image_folder) and os.path.isdir(label_folder):
        images = sorted(os.listdir(image_folder))
        for img_name in images:
            label_name = img_name.replace(".png", ".txt")
            img_path = os.path.join(image_folder, img_name)
            label_path = os.path.join(label_folder, label_name)

            if os.path.exists(img_path) and os.path.exists(label_path):
                image_label_pairs.append((img_path, label_path))

# Split data into train and validation sets
train_pairs, val_pairs = train_test_split(
    image_label_pairs, test_size=0.2, random_state=42
)

# Split validation data into validation and test sets
val_pairs, test_pairs = train_test_split(val_pairs, test_size=0.5, random_state=42)


# Move files to train, val, and test directories
def move_files(pairs, img_dest, lbl_dest):
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, img_dest)
        shutil.copy(lbl_path, lbl_dest)


move_files(train_pairs, output_images_train, output_labels_train)
move_files(val_pairs, output_images_val, output_labels_val)
move_files(test_pairs, output_images_test, output_labels_test)

print("Dataset preparation complete!")
