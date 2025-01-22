import subprocess
import sys


def train_yolov5n():
    try:
        # Define the command to run the YOLOv5 training script
        command = [
            sys.executable,  # Use the current Python executable
            r"C:\Users\Samarth\Desktop\polar3D\yolov5\train.py",  # Path to YOLOv5 train.py
            "--img",
            "640",  # Image size
            "--batch",
            "4",  # Batch size (adjust based on available resources)
            "--epochs",
            "45",  # Number of epochs
            "--data",
            r"C:\Users\Samarth\Desktop\polar3D\yolo5_train\custom_dataset.yaml",  # Path to dataset config
            "--weights",
            r"C:\Users\Samarth\Desktop\polar3D\yolov5\yolov5n.pt",  # Path to YOLOv5 Nano pre-trained weights
            "--device",
            "0",  # Device to use (e.g., GPU 0, or "cpu" for CPU)
            "--project",
            r"C:\Users\Samarth\Desktop\polar3D\yolov5\model",  # Path to save models
        ]

        # print("Starting YOLOv5 Nano training...")
        subprocess.check_call(command)
        print("Training started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while starting training: {e}")
        sys.exit(1)


# Call the function
train_yolov5n()
