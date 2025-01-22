# Object Detection on Lunar Terrain Images

This project focuses on detecting rocks and shadows from lunar terrain images using the POLAR SIM dataset. Two object detection models were explored:

- **YOLOv5 Nano**
- **Faster R-CNN (with MobileNet backbone)**

## Project Structure

### Dataset Preparation

1. Clone the [POLAR-Sim repository](https://github.com/uwsbel/POLAR-Sim.git):

   ```bash
   git clone https://github.com/uwsbel/POLAR-Sim.git
   ```

   This repository contains YOLO-compatible labels.
2. Download the [POLAR SIM dataset](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/dataset_public_release.zip) and extract it.
3. Clone this project repository:

   ```bash
   git clone https://github.com/Sohm14/object_detection.git
   ```

## YOLOv5 Nano

### Steps to Train YOLOv5 Nano

1. **Clone YOLOv5 Repository**:

   - Clone the YOLOv5 repository and install dependencies:
     ```bash
     git clone https://github.com/ultralytics/yolov5 
     cd yolov5
     pip install -r requirements.txt 
     ```
2. **Download YOLOv5 Nano Model**:

   - Use `wget` to download the YOLOv5 Nano model:
     ```bash
     wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
     ```
3. **Prepare the Dataset**:

   - Run the scripts in the dataset_preparation directory in the following order:
     - `data_preparation.py`
     - `image_resize.py`
     - `image_to_rgb.py`
     - `augmentation.py`
4. **Create the Configuration File**:

   - Create a `custom_dataset.yaml` file as shown in the `yolo_train` directory. This file should define the paths to the dataset and the number of classes.
5. **Train the Model**:

   - Run the YOLOv5 training script:
     ```bash
     python yolo5_model_train.py
     ```
   - The trained model will be saved in the `models` directory.
6. **View Predictions**:

   - Run the `result.py` script to see predictions on test images.

## Faster R-CNN

### Steps to Train Faster R-CNN

1. **Convert Labels**:

   - Run the `convert_yolo_labels.py` script to convert YOLO labels to Faster R-CNN format.
   - The converted dataset will be saved in the `datasets` directory within the `rcnn_train` directory.
2. **Train the Model**:

   - See the training script in the jupitar notebook for Faster R-CNN (with MobileNet backbone):
     ```bash
     sol.ipynb
     ```
3. **Evaluate the Model**:

   - Use the confusion matrix to evaluate the model performance .
   - using confusion_matrix.ipynb

## Results

Both models were trained and tested on the POLAR SIM dataset. The performance of the models was evaluated based on accuracy and the ability to detect both rocks and shadows under varying lunar terrain conditions.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- YOLOv5 dependencies
- Faster R-CNN dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Acknowledgements

- [POLAR-Sim Dataset](https://github.com/uwsbel/POLAR-Sim)
