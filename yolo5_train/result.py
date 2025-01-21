# import cv2
# import torch

# # Load the YOLOv5 model
# model = torch.hub.load(
#     "ultralytics/yolov5",
#     "custom",
#     path="C:/Users/Samarth/Desktop/polar3D/models/exp4/exp2/weights/best.pt",
# )

# # Load the image
# image_path = r"C:\Users\Samarth\Desktop\polar3D\dataset_resized\images\train\01_C_off_30_L_0032.png"
# image = cv2.imread(image_path)
# height, width, _ = image.shape  # Get image dimensions

# # Perform object detection
# results = model(image)

# # Extract detections
# detections = results.xywhn[
#     0
# ]  # Normalized [x_center, y_center, width, height, confidence, class]
# classes = results.names  # Class names

# # Open a file to save YOLO format labels
# output_label_path = "C:/Users/Samarth/Desktop/polar3D/detected_labels1.txt"
# with open(output_label_path, "w") as file:
#     for detection in detections:
#         (
#             x_center,
#             y_center,
#             bbox_width,
#             bbox_height,
#             confidence,
#             class_id,
#         ) = detection.tolist()
#         class_id = int(class_id)

#         # Write to YOLO format output file
#         file.write(
#             f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
#         )

#         # Draw bounding box if class is 0
#         if class_id == 0:
#             # Convert normalized values to absolute pixel coordinates
#             x1 = int((x_center - bbox_width / 2) * width)
#             y1 = int((y_center - bbox_height / 2) * height)
#             x2 = int((x_center + bbox_width / 2) * width)
#             y2 = int((y_center + bbox_height / 2) * height)

#             # Draw bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f"{classes[class_id]} {confidence:.2f}"
#             cv2.putText(
#                 image,
#                 label,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )

# # Save the image with bounding boxes
# output_image_path = "C:/Users/Samarth/Desktop/polar3D/detected_image1.jpg"
# cv2.imwrite(output_image_path, image)

# # Display the image with bounding boxes
# cv2.imshow("Detected Objects", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(f"YOLO labels saved to: {output_label_path}")
# print(f"Image with bounding boxes saved to: {output_image_path}")

import cv2
import torch
from PIL import Image
import numpy as np
import random
import os

# Load the YOLOv5 model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=r"C:\Users\Samarth\Desktop\polar3D\models\exp5\exp\weights\best.pt",
)


def process_image(image_path):
    try:
        exposure = image_path.split("_")[-1].split(".")[0]
        print(exposure)
    except IndexError:
        print(f"Error: {image_path}")

    try:
        if exposure in ["0016", "0032", "0064", "0128"]:
            # Process with PIL
            try:
                with Image.open(image_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_array = np.array(img)
                    return img_array
            except Exception as e:
                raise ValueError(f"Error processing image with PIL: {e}")

        else:
            # Process with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image with OpenCV.")

            # Convert to grayscale and equalize histogram
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            return equalized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_random_images(img_dir, num_images):
    try:
        # Get all image files from the directory
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png"))]

        if len(image_files) < num_images:
            raise ValueError(
                f"Not enough images in the directory to select {num_images}. Found only {len(image_files)}."
            )

        # Select random images
        # selected_images = random.sample(image_files, num_images)

        # Process each selected image
        processed_images = []
        for image_name in image_files:
            image_path = os.path.join(img_dir, image_name)
            print(f"Processing image: {image_name}")
            processed_image = process_image(image_path)
            if processed_image is not None:
                processed_images.append((image_name, processed_image))
            else:
                print(f"Failed to process image: {image_name}")

        return processed_images

    except Exception as e:
        print(f"Error: {e}")
        return []


# Load the image
img_dir = r"C:\Users\Samarth\Desktop\polar3D\dataset_resized1_copy\images\test"
result_dir = r"C:\Users\Samarth\Desktop\polar3D\results\yolo5_nano\images"
os.makedirs(result_dir, exist_ok=True)
# image_path = r"C:\Users\Samarth\Desktop\polar3D\dataset_resized1_copy\images\test\04_B_off_180_R_0128.png"
# image = process_image(image_path)

processed_images = process_random_images(img_dir, num_images=len(img_dir))

for image_name, image in processed_images:
    # Check and resize image to 640x640 if necessary
    if image is not None:
        try:
            # Ensure the image has a 2D or 3D shape before resizing
            if image.ndim not in [2, 3]:
                raise ValueError("Processed image has an invalid number of dimensions.")

            if image.shape[:2] != (640, 640):
                image = cv2.resize(image, (640, 640))

            print("Image processed and resized successfully.")
        except Exception as e:
            print(f"Error resizing image: {e}")
    else:
        print("Image processing failed.")

    # alpha = 3  # Contrast control (1.0-3.0)
    # beta = 0  # Brightness control (0-100)
    # adjusted_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    height, width = 0, 0

    # Check the number of dimensions
    if image.ndim == 2:  # Grayscale image
        height, width = image.shape
    elif image.ndim == 3:  # Color image
        height, width, _ = image.shape
    else:
        print("Invalid image dimensions.")

    # Perform object detection
    results = model(image)

    # Extract detections
    detections = results.xywhn[
        0
    ]  # Normalized [x_center, y_center, width, height, confidence, class]
    classes = results.names  # Class names

    # Initialize counter for class 0
    class_0_count = 0

    # Open a file to save YOLO format labels
    # output_label_path = "C:/Users/Samarth/Desktop/polar3D/detected_labels1.txt"
    label_name = image_name.replace("png", "txt")
    output_label_path = os.path.join(result_dir, label_name)
    with open(output_label_path, "w") as file:
        for detection in detections:
            (
                x_center,
                y_center,
                bbox_width,
                bbox_height,
                confidence,
                class_id,
            ) = detection.tolist()
            class_id = int(class_id)

            # Write to YOLO format output file
            file.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
            )

            # Process detections for class 0
            if class_id == 0:
                class_0_count += 1

                # Convert normalized values to absolute pixel coordinates
                x1 = int((x_center - bbox_width / 2) * width)
                y1 = int((y_center - bbox_height / 2) * height)
                x2 = int((x_center + bbox_width / 2) * width)
                y2 = int((y_center + bbox_height / 2) * height)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{class_0_count}"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

    cv2.putText(
        image,
        f"Total Count: {class_0_count}",
        (10, 30),  # Position at the top-left
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # Save the image with bounding boxes and count
    # output_image_path = "C:/Users/Samarth/Desktop/polar3D/detected_image1.jpg"
    output_image_path = os.path.join(result_dir, image_name)
    cv2.imwrite(output_image_path, image)

    # # Display the image with bounding boxes
    # cv2.imshow("Detected Objects", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"YOLO labels saved to: {output_label_path}")
    print(f"Image with bounding boxes saved to: {output_image_path}")
