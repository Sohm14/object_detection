import os
import glob
import shutil


def convert_yolo_to_faster_rcnn(data_dir, output_dir):
    """
    Convert YOLO labels to Faster R-CNN format and organize into the specified output directory.

    :param data_dir: Root directory containing 'images/train', 'images/val', 'images/test',
                     'labels/train', 'labels/val', 'labels/test'.
    :param output_dir: Root directory to store converted data in Faster R-CNN format.
    """
    # Define the structure for the output directory
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for split in splits:
        images_src_dir = os.path.join(data_dir, "images", split)
        labels_src_dir = os.path.join(data_dir, "labels", split)

        images_dest_dir = os.path.join(output_dir, "images", split)
        labels_dest_dir = os.path.join(output_dir, "labels", split)

        # Copy images
        for image_file in glob.glob(os.path.join(images_src_dir, "*.png")):
            shutil.copy(image_file, images_dest_dir)

        # Convert and copy labels
        for label_file in glob.glob(os.path.join(labels_src_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()

            image_name = os.path.basename(label_file).replace(".txt", ".png")
            image_path = os.path.join(images_src_dir, image_name)

            # Dynamically get image dimensions if necessary
            # if os.path.exists(image_path):
            #     from PIL import Image

            #     with Image.open(image_path) as img:
            #         width, height = img.size
            # else:
            #     width, height = 1920, 1080  # Default dimensions
            width, height = 640, 640

            annotations = []
            for line in lines:
                class_id, x_center, y_center, bbox_width, bbox_height = map(
                    float, line.split()
                )
                class_id += 1
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                x_max = (x_center + bbox_width / 2) * width
                y_max = (y_center + bbox_height / 2) * height

                annotations.append(f"{int(class_id)} {x_min} {y_min} {x_max} {y_max}\n")

            # Save converted labels
            output_label_path = os.path.join(
                labels_dest_dir, os.path.basename(label_file)
            )
            with open(output_label_path, "w") as f:
                f.writelines(annotations)


# Example usage
output_dir = r"rcnn_train\dataset_rcnn"
os.makedirs(output_dir, exist_ok=True)

convert_yolo_to_faster_rcnn(
    data_dir=r"yolo_train\dataset_yolo",
    output_dir=output_dir,
)
