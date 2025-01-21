import os
from PIL import Image


def resize_and_save_images_labels(
    input_images_folder, input_labels_folder, output_folder, target_size=(640, 640)
):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all subfolders (train and val) in the images and labels directories
    subfolders = ["train", "test" "val"]

    for subfolder in subfolders:
        image_subfolder = os.path.join(input_images_folder, subfolder)
        label_subfolder = os.path.join(input_labels_folder, subfolder)
        output_images_subfolder = os.path.join(output_folder, "images", subfolder)
        output_labels_subfolder = os.path.join(output_folder, "labels", subfolder)

        # Create subfolders in the output directory if they don't exist
        os.makedirs(output_images_subfolder, exist_ok=True)
        os.makedirs(output_labels_subfolder, exist_ok=True)

        # Get all image files from the image subfolder
        image_files = [
            f
            for f in os.listdir(image_subfolder)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        if not image_files:
            print(f"No image files found in the {subfolder} subfolder.")
            continue

        for image_file in image_files:
            try:
                # Define the full image and label paths
                image_path = os.path.join(image_subfolder, image_file)
                label_path = os.path.join(
                    label_subfolder, os.path.splitext(image_file)[0] + ".txt"
                )

                # Check if the label file exists
                if not os.path.exists(label_path):
                    print(f"Label file not found for {image_file}, skipping.")
                    continue

                # Open the image
                image = Image.open(image_path)

                # Get original image dimensions
                img_width, img_height = image.size

                # Resize the image to the target size
                image_resized = image.resize(target_size)
                new_width, new_height = image_resized.size

                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(image_path))[0]

                # Save the resized image
                resized_image_path = os.path.join(
                    output_images_subfolder, f"{base_filename}.png"
                )
                image_resized.save(resized_image_path)

                # Read the YOLO label file
                with open(label_path, "r") as file:
                    labels = file.readlines()

                # Prepare the updated label file content
                updated_labels = []

                # Process each label and adjust the bounding box coordinates
                for label in labels:
                    # YOLO format: class_id center_x center_y width height (normalized)
                    parts = label.strip().split()
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])

                    # Convert normalized coordinates to pixel values (original image)
                    center_x *= img_width
                    center_y *= img_height
                    width *= img_width
                    height *= img_height

                    # Resize the coordinates to match the new image dimensions
                    center_x_new = center_x * (new_width / img_width)
                    center_y_new = center_y * (new_height / img_height)
                    width_new = width * (new_width / img_width)
                    height_new = height * (new_height / img_height)

                    # Normalize the new coordinates
                    center_x_new /= new_width
                    center_y_new /= new_height
                    width_new /= new_width
                    height_new /= new_height

                    # Update label and store the new normalized labels
                    updated_labels.append(
                        f"{class_id} {center_x_new} {center_y_new} {width_new} {height_new}\n"
                    )

                # Save the updated labels to a new file
                updated_label_path = os.path.join(
                    output_labels_subfolder, f"{base_filename}.txt"
                )
                with open(updated_label_path, "w") as label_file:
                    label_file.writelines(updated_labels)

                print(
                    f"Processed {image_file} and saved to {output_images_subfolder} and {output_labels_subfolder}"
                )

            except Exception as e:
                print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    input_images_folder = r"C:\Users\Samarth\Desktop\polar3D\datasets\dataset\images"
    input_labels_folder = r"C:\Users\Samarth\Desktop\polar3D\datasets\dataset\labels"
    output_folder = r"C:\Users\Samarth\Desktop\polar3D\dataset_resized1"
    resize_and_save_images_labels(
        input_images_folder, input_labels_folder, output_folder
    )
