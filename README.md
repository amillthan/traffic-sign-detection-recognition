# Traffic Sign Detection and Recognition System

An end-to-end deep learning system for automatically detecting and recognizing German traffic signs from road images.

The project combines two computer vision models:

1. A YOLO-based object detector to locate traffic signs in full road images.
2. A Convolutional Neural Network (CNN) to classify each detected sign into one of 43 German traffic sign classes.

A Streamlit web application provides an interactive interface where users can upload a road image and receive automatic traffic sign detections and classifications.

## Project Overview

Traffic sign recognition from a cropped sign image is only one part of a practical traffic sign recognition system. In real road scenes, the system must first determine where traffic signs are located before they can be classified.

This project therefore uses a two-stage architecture:

```text
Road Image
    |
    v
YOLO Traffic Sign Detector
    |
    v
Detected Bounding Box
    |
    v
15% Padded Traffic Sign Crop
    |
    v
CNN Traffic Sign Classifier
    |
    v
43-Class Traffic Sign Prediction
```

The detector is responsible for locating traffic signs, while the CNN is responsible for identifying the type of each detected sign.

## Main Features

- Automatic traffic sign detection from full road images
- Traffic sign classification into 43 GTSRB classes
- Multiple traffic sign detection in a single image
- Top-3 CNN predictions for each detected sign
- YOLO detection confidence display
- CNN classification confidence display
- Bounding box visualization
- 15% padded sign cropping before classification
- Interactive Streamlit web application
- Separate training and evaluation pipelines
- Support for external road-image evaluation

## Technologies

The main technologies used in this project are:

- Python
- TensorFlow
- Keras
- Ultralytics YOLO
- PyTorch
- NumPy
- Pillow
- Matplotlib
- Streamlit
- Jupyter Notebook

## Datasets

Three traffic sign datasets are used in the project.

### GTSRB

The German Traffic Sign Recognition Benchmark (GTSRB) is used to train and evaluate the CNN classifier.

The classifier recognizes 43 traffic sign classes.

### GTSDB

The German Traffic Sign Detection Benchmark (GTSDB) is used for traffic sign detector training.

The original annotations are converted into YOLO format and treated as a one-class detection problem:

```text
traffic_sign
```

### Mapillary Traffic Sign Dataset (MTSD)

MTSD is used as supplementary detection data.

A subset of compatible traffic signs is selected and combined with GTSDB to increase the diversity of the detector training data.

Large datasets are excluded from this repository.

See:

```text
data/README.md
```

for additional information about the expected dataset structure.

## CNN Traffic Sign Classifier

The final classifier is a custom Convolutional Neural Network trained using GTSRB.

### Input

```text
32 x 32 x 1
```

Each RGB traffic sign image is converted to grayscale using the arithmetic mean of the RGB channels:

```text
grayscale = (R + G + B) / 3
```

The grayscale values are then normalized using:

```text
(grayscale - 128) / 128
```

The same preprocessing procedure is used during both training and inference.

### CNN Architecture

The final classifier contains:

```text
Data Augmentation
        |
Conv2D (32 filters)
        |
Batch Normalization
        |
ReLU
        |
Conv2D (32 filters)
        |
Max Pooling
        |
Dropout
        |
Conv2D (64 filters)
        |
Batch Normalization
        |
ReLU
        |
Conv2D (64 filters)
        |
Max Pooling
        |
Dropout
        |
Flatten
        |
Dense (256)
        |
Dropout
        |
Dense (43, Softmax)
```

The model contains approximately 1.125 million parameters.

### CNN Training

The final CNN was trained using:

- Adam optimizer
- Sparse categorical cross-entropy
- Batch size: 128
- Maximum epochs: 50
- Early stopping
- Validation-loss monitoring
- Data augmentation

Training stopped after 24 epochs, with the best validation loss occurring at epoch 17.

### CNN Test Results

The final CNN achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | 97.07% |
| Test Loss | 0.1444 |
| Macro Precision | 94.98% |
| Macro Recall | 94.41% |
| Macro F1 Score | 94.36% |
| Weighted F1 Score | 97.03% |
| Correct Predictions | 12,260 |
| Incorrect Predictions | 370 |
| Test Samples | 12,630 |

These results measure the CNN classifier on the GTSRB test dataset and should not be interpreted as end-to-end road-scene performance.

## YOLO Traffic Sign Detector

The detector is trained as a one-class object detector.

```text
Class 0: traffic_sign
```

The purpose of the detector is only to locate traffic signs. The CNN performs the final 43-class recognition.

### GTSDB Preparation

The GTSDB dataset contains:

```text
900 road images
1,213 traffic sign annotations
```

The complete dataset was converted into YOLO format.

The train/validation split was:

```text
720 training images
180 validation images
```

Images without traffic signs were retained as negative training examples.

### MTSD Selection

A compatible subset of MTSD was selected for supplementary detector training.

The selected subset contains:

```text
800 images
1,266 supported traffic sign bounding boxes
```

### Combined Detection Dataset

The final detector training dataset combines GTSDB with the selected MTSD data.

```text
Total images: 1,700
Training images: 1,360
Validation images: 340
Traffic sign boxes: 2,479
```

### Final Detector Training

The final detector was trained using:

```text
Model: YOLO26n
Image size: 640
Epochs: 50
Batch size: 4
Device: CPU
Patience: 10
```

### Detector Validation Results

The final combined-data detector achieved:

| Metric | Result |
|---|---:|
| Precision | 0.820 |
| Recall | 0.671 |
| mAP@0.50 | 0.741 |
| mAP@0.50:0.95 | 0.555 |

These results were obtained on the validation split of the combined GTSDB and selected MTSD dataset.

Results from different detector training datasets or validation splits should not be directly compared as if they were measured under identical conditions.

## End-to-End Pipeline

During inference, YOLO first detects traffic signs in the full road image.

The detector uses:

```text
Image size: 640
Detection confidence threshold: 0.25
IoU threshold: 0.45
```

A 15% padding area is added around each detected bounding box before sending the crop to the CNN.

The original YOLO bounding box is retained for visualization.

```text
Full Road Image
      |
      v
YOLO Detection
      |
      v
Original Bounding Box
      |
      +----------------------+
      |                      |
      v                      v
Display Box          Add 15% Crop Padding
                             |
                             v
                     Resize to 32 x 32
                             |
                             v
                     Grayscale Conversion
                             |
                             v
                       Normalization
                             |
                             v
                       CNN Classifier
                             |
                             v
                    Top-3 Predictions
```

## External Road-Image Evaluation

A small set of 10 external road images containing 13 relevant traffic signs was used to examine how the complete system behaves on road scenes outside the primary training datasets.

The final pipeline produced:

```text
Relevant traffic signs: 13
Signs detected by YOLO: 11
Correctly detected and classified: 8
Detector misses: 2
Detected but misclassified: 3
```

This gives:

| External Evaluation Metric | Result |
|---|---:|
| Sign-Level Detector Recall | 84.62% (11/13) |
| End-to-End Correct Recognition | 61.54% (8/13) |
| Images with at least one detection | 9/10 |

The 9/10 result represents image-level detection coverage and is not classification accuracy.

The 61.54% result represents successful end-to-end detection and recognition on this small external set. It should not be confused with the CNN's 97.07% GTSRB test accuracy.

The external images were also used during development and crop-padding experiments. Therefore, this set is reported as a small generalization evaluation set rather than an untouched final benchmark.

## Streamlit Application

The project includes an interactive Streamlit application.

Users can:

- Upload a full road image
- Automatically detect traffic signs
- View bounding boxes
- View individual detected sign crops
- View YOLO detection confidence
- View CNN classification confidence
- View the top three CNN predictions
- Process images containing multiple traffic signs

The application uses the final models stored in:

```text
models/classifier/traffic_sign_cnn.keras
models/detector/best.pt
```

## Project Structure

```text
traffic-sign-recognition/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- yolo26n.pt
|
|-- .streamlit/
|
|-- models/
|   |-- classifier/
|   |   |-- traffic_sign_cnn.keras
|   |   `-- class_names.json
|   |
|   `-- detector/
|       `-- best.pt
|
|-- src/
|   |-- __init__.py
|   |-- classifier.py
|   `-- pipeline.py
|
|-- training/
|   |-- classifier/
|   |   `-- Traffic_Sign_Classification.ipynb
|   |
|   `-- detector/
|       |-- prepare_gtsdb.py
|       |-- verify_gtsdb.py
|       |-- check_gtsdb_dataset.py
|       |-- analyze_mtsd_labels.py
|       |-- select_mtsd_dataset.py
|       |-- verify_mtsd_dataset.py
|       |-- combine_datasets.py
|       |-- check_combined_dataset.py
|       |-- train_gtsdb_detector.py
|       |-- train_combined_detector.py
|       |-- traffic_sign_gtsdb.yaml
|       `-- traffic_sign_combined.yaml
|
|-- evaluation/
|   |-- __init__.py
|   |-- save_external_crops.py
|   |-- test_classifier_model.py
|   |-- test_external_detector.py
|   |-- test_external_pipeline.py
|   `-- test_padding.py
|
|-- data/
|   |-- README.md
|   |-- gtsrb/
|   |-- gtsdb/
|   |-- mtsd/
|   |-- combined_dataset/
|   `-- external_test/
|
|-- results/
|   |-- classifier/
|   |-- detector/
|   `-- external_evaluation/
|
|-- assets/
|   |-- screenshots/
|   `-- diagrams/
|
`-- docs/
    `-- project_report/
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd traffic-sign-recognition
```

### 2. Create a Python Environment

A separate Python environment is recommended.

The project was developed and verified using:

```text
Python 3.13.5
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The verified development environment includes:

```text
TensorFlow 2.21.0
Keras 3.15.1
NumPy 2.1.3
Pillow 11.1.0
Ultralytics 8.4.161
PyTorch 2.14.0
PyYAML 6.0.2
Matplotlib 3.10.0
Streamlit 1.45.1
```

## Running the Application

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local address in the terminal.

Upload a road image through the interface to run the complete detection and classification pipeline.

## Model Files

The deployment models are:

```text
models/classifier/traffic_sign_cnn.keras
models/detector/best.pt
```

The root:

```text
yolo26n.pt
```

is the generic pretrained YOLO checkpoint used as the starting point for detector training. It is not the final deployment detector.

## Training

### CNN Classifier

The classifier training notebook is located at:

```text
training/classifier/Traffic_Sign_Classification.ipynb
```

### YOLO Detector

Detector preparation and training scripts are located under:

```text
training/detector/
```

The final combined detector can be trained using:

```bash
python training/detector/train_combined_detector.py
```

The training scripts use project-relative paths so that the project is not tied to the original development directory.

## Evaluation

Evaluation utilities are available under:

```text
evaluation/
```

Examples include:

```bash
python -m evaluation.test_classifier_model
python -m evaluation.test_external_detector
python -m evaluation.test_external_pipeline
```

These scripts test different parts of the system independently.

## Limitations

The project currently has several limitations:

- The detector may miss small, distant, partially visible, or visually difficult traffic signs.
- A correctly detected sign can still be misclassified by the CNN.
- Detection errors propagate to the classification stage.
- The CNN was primarily trained on GTSRB cropped traffic signs, while real-world YOLO crops can contain additional background, perspective distortion, blur, and lighting variation.
- CNN softmax confidence should not be interpreted as a calibrated probability of correctness.
- The small external evaluation set is not an untouched benchmark because it was also used during development and crop-padding experiments.
- The detector is trained as a single `traffic_sign` class and relies on the CNN for fine-grained recognition.

## Future Improvements

Possible future improvements include:

- Training the detector with more diverse road-scene data
- Increasing representation of small and distant traffic signs
- Improving alignment between detector crops and CNN training data
- Using stronger traffic sign classification architectures
- Investigating confidence calibration
- Evaluating on a larger untouched external test dataset
- Testing the pipeline on video
- Extending the system toward real-time traffic sign recognition

## Summary

This project demonstrates a complete traffic sign recognition workflow rather than only a standalone image classifier.

```text
Road Image
   |
   v
YOLO Detector
   |
   v
Traffic Sign Localization
   |
   v
CNN Classifier
   |
   v
43-Class Recognition
   |
   v
Streamlit Interface
```

The CNN achieved 97.07% accuracy on the GTSRB test dataset, while the external road-image evaluation demonstrated the additional challenges introduced when detection and classification are combined into a complete real-world pipeline.