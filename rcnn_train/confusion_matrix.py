import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def load_labels(file_path):
    """Load labels from a text file."""
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append(
                (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            )
    return labels


def iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compute_confusion_matrix(
    ground_truth_dir, prediction_dir, labels, iou_threshold=0.5
):
    """
    Compute confusion matrix for object detection.
    - ground_truth_dir: Directory containing ground truth label files.
    - prediction_dir: Directory containing prediction label files.
    - labels: List of class labels.
    - iou_threshold: IoU threshold for matching.
    """
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    gt_files = sorted(os.listdir(ground_truth_dir))
    pred_files = sorted(os.listdir(prediction_dir))

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_path = os.path.join(prediction_dir, pred_file)

        ground_truths = load_labels(gt_path)
        predictions = load_labels(pred_path)

        for gt in ground_truths:
            gt_matched = False
            gt_class, gt_box = gt[0], gt[1:]

            for pred in predictions:
                pred_class, pred_box = pred[0], pred[1:]
                if iou(gt_box, pred_box) >= iou_threshold:
                    if pred_class == gt_class:
                        confusion_matrix[gt_class, pred_class] += 1  # True Positive
                    else:
                        confusion_matrix[gt_class, pred_class] += 1  # Misclassified
                    gt_matched = True
                    predictions.remove(pred)  # Remove matched prediction
                    break

            if not gt_matched:
                confusion_matrix[gt_class, gt_class] += 1  # False Negative

        # Remaining predictions are False Positives
        for pred in predictions:
            pred_class = pred[0]
            confusion_matrix[
                0, pred_class
            ] += 1  # Assume unmatched predictions are background FP

    return confusion_matrix


def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, and accuracy from the confusion matrix."""
    tp = np.diag(confusion_matrix)
    precision = np.divide(
        tp,
        np.sum(confusion_matrix, axis=0),
        out=np.zeros_like(tp, dtype=float),
        where=np.sum(confusion_matrix, axis=0) != 0,
    )
    recall = np.divide(
        tp,
        np.sum(confusion_matrix, axis=1),
        out=np.zeros_like(tp, dtype=float),
        where=np.sum(confusion_matrix, axis=1) != 0,
    )
    accuracy = (
        np.sum(tp) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0
    )
    f1_score = np.divide(
        2 * (precision * recall),
        (precision + recall),
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    return precision, recall, f1_score, accuracy


def visualize_confusion_matrix(confusion_matrix, labels, output_path):
    """Visualize and save the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=labels
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()


def visualize_metrics(metrics, labels, output_dir):
    """Visualize precision, recall, and accuracy."""
    precision, recall, f1_score, accuracy = metrics
    metrics_dict = {"Precision": precision, "Recall": recall, "F1-Score": f1_score}

    for metric_name, metric_values in metrics_dict.items():
        plt.figure(figsize=(8, 6))
        plt.bar(labels, metric_values, color="skyblue")
        plt.title(f"{metric_name} by Class")
        plt.xlabel("Class Labels")
        plt.ylabel(metric_name)
        for i, value in enumerate(metric_values):
            plt.text(i, value + 0.01, f"{value:.2f}", ha="center")
        output_path = os.path.join(output_dir, f"{metric_name.lower()}_by_class.jpeg")
        plt.savefig(output_path)
        plt.close()

    # Visualize overall accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(["Overall Accuracy"], [accuracy], color="lightcoral")
    plt.title("Overall Accuracy")
    plt.ylabel("Accuracy")
    plt.text(0, accuracy + 0.01, f"{accuracy:.2f}", ha="center")
    output_path = os.path.join(output_dir, "overall_accuracy.jpeg")
    plt.savefig(output_path)
    plt.close()


# Example usage
ground_truth_dir = (
    "rcnn_train\dataset_rcnn\labels\test"  # Directory containing ground truth files
)
prediction_dir = "rcnn_train\predictions"  # Directory containing prediction files
labels = ["Background", "Rock", "Shadow"]

conf_matrix = compute_confusion_matrix(ground_truth_dir, prediction_dir, labels)
metrics = calculate_metrics(conf_matrix)

output_dir = "rcnn_train\visualizations"
os.makedirs(output_dir, exist_ok=True)

visualize_confusion_matrix(
    conf_matrix, labels, os.path.join(output_dir, "confusion_matrix.jpeg")
)
visualize_metrics(metrics, labels, output_dir)

print("Visualizations saved in:", output_dir)
