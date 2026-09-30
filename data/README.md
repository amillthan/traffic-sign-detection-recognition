# Dataset Information

The datasets used to train and evaluate this project are not included in the GitHub repository because of their size.

The project uses three main traffic sign datasets together with a small external evaluation set.

## 1. GTSRB

The German Traffic Sign Recognition Benchmark (GTSRB) is used to train and evaluate the CNN traffic sign classifier.

Expected project structure:

```text
data/gtsrb/
|-- train.p
|-- valid.p
`-- test.p
```

The classifier recognizes 43 traffic sign classes.

## 2. GTSDB

The German Traffic Sign Detection Benchmark (GTSDB) is used for traffic sign detection training.

The original dataset is converted into YOLO format before detector training.

Expected project structure:

```text
data/gtsdb/
|-- FullIJCNN2013/
`-- yolo_dataset/
```

The detector is trained as a one-class object detector using the class:

```text
traffic_sign
```

The detected traffic sign is later classified into one of the 43 GTSRB classes by the CNN classifier.

## 3. Mapillary Traffic Sign Dataset (MTSD)

The Mapillary Traffic Sign Dataset (MTSD) is used as supplementary detection data to improve the diversity of the traffic sign detector.

Only traffic signs compatible with the project's supported German traffic sign categories are selected for detector training.

Expected project structure:

```text
data/mtsd/
|-- raw/
`-- selected_training/
```

The selected MTSD samples are combined with the GTSDB training data to create the final detector training dataset.

## 4. Combined Detection Dataset

The combined dataset contains traffic sign detection samples prepared from GTSDB and the selected MTSD samples.

Expected project structure:

```text
data/combined_dataset/
|-- images/
|   |-- train/
|   `-- val/
`-- labels/
    |-- train/
    `-- val/
```

This dataset is used to train the final YOLO traffic sign detector.

## 5. External Evaluation Images

A small collection of road images is stored in:

```text
data/external_test/
```

These images are used to evaluate the complete detection and classification pipeline on road scenes outside the primary training datasets.

This external set was also used during development and tuning, so it should be treated as a small generalization evaluation set rather than an untouched final benchmark.

## Data Processing Pipeline

The overall dataset usage is:

```text
GTSRB -> CNN traffic sign classifier

GTSDB + selected MTSD -> YOLO traffic sign detector

YOLO detector -> detected traffic sign crop -> CNN classifier -> final traffic sign prediction
```

## Note

Large raw and processed datasets are intentionally excluded from version control through the project's `.gitignore` file.