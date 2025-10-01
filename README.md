# OpenMedicQR: End-to-End Multi-QR Code Recognition

## 1. Project Overview

OpenMedicQR is a complete, end-to-end computer vision pipeline designed to solve the real-world challenge of recognizing multiple QR codes on medicine packs. Using only open-source Python libraries, this project robustly detects and decodes QR codes from images, even when faced with visual challenges like blur, tilt, and difficult lighting.

The entire system is designed to run locally without any reliance on external APIs or paid services, ensuring it is accessible, reproducible, and compliant with the hackathon rules. The project successfully implements both the primary detection task and the bonus decoding task.

### Key Features
*   **Accurate Multi-QR Detection:** Utilizes a custom-trained YOLOv8 model to detect all QR codes in an image.
*   **Offline QR Decoding:** Employs the `pyzbar` and `OpenCV` libraries for efficient, local decoding of detected QR codes.
*   **End-to-End Reproducibility:** From setup to final output, the entire pipeline can be reproduced with a few simple commands.
*   **Compliance:** Adheres strictly to the "no external APIs" rule, using only `pip`-installable libraries.

### Core Workflow
The project follows a standard machine learning workflow:

1.  **Data Preparation:** 200 annotated images were used to create a robust training dataset.
2.  **Model Training:** A YOLOv8 model was fine-tuned on this dataset to learn how to specifically identify QR codes on medicine packs.
3.  **Inference Pipeline:** The trained model is used to process a folder of unseen test images, detect all QR codes, and decode their contents.
4.  **Output Generation:** The pipeline produces two structured JSON files containing the detection and decoding results, matching the exact format required for submission.

---

## 2. Repository Structure

The project follows the recommended repository structure for clarity and ease of evaluation.

```
multiqr-hackathon/
├── README.md                # You are here!
├── requirements.txt         # All Python dependencies
├── train.py                 # Script to train the YOLO model
├── infer.py                 # Script to run inference and generate final JSON outputs
├── data/
│   └── demo_images/         # A small set of images for quick demo runs
├── outputs/
│   ├── submission_detection_1.json   # Stage 1 output file
│   └── submission_decoding_2.json    # Stage 2 (bonus) output file
└── src/
    ├── models/
    ├── datasets/
    └── utils/
```

---

## 3. Step-by-Step Reproduction Guide

Follow these instructions to set up the environment and run the complete project pipeline.

### Step 3.1: Environment Setup

**Prerequisites:**
*   Python 3.8+
*   Git for cloning the repository

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd multiqr-hackathon
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate.bat

    # On macOS or Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install following packages manually**
    ```bash
    pip install pyyaml
    pip install ultralytics==8.0.196
    pip install scikit-learn
    ```

4.  **Install all required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 3.2: Dataset Preparation

This project was trained on a dataset of 200 annotated images and evaluated on a test set of 50 images, as provided by the hackathon.

To run the pipeline, you must place the dataset folders in the following structure:

```
multiqr-hackathon/
└── data/
    ├── train_images/     # Folder with ~200 .jpg training images
    ├── train_labels/     # Folder with ~200 .txt YOLO annotation files
    └── test_images/      # Folder with 50 .jpg test images
```
*(Note: Per hackathon rules, the actual dataset is not committed to this repository.)*

### Step 3.3: How to Run the Pipeline

The entire process is broken down into two main commands: training and inference.

#### **Part 1: Training the Model**

This script automates the process of splitting the data, configuring the model, and training it for 50 epochs.

1.  **Run the training script from the root directory:**
    ```bash
    python train.py
    ```

2.  **Output:** The training process will take some time. Upon completion, the best-performing model weights will be saved as `best.pt` in a newly created `runs/detect/.../` directory.

#### **Part 2: Running Inference**

This script uses the trained `best.pt` model to process the test images and generate the final submission files.

1.  **Run the inference script, providing the path to your trained model and the input images:**
    *(Make sure to replace the model path with the actual path generated after your training is complete.)*

    ```bash
    python infer.py --model runs/detect/yolov8_qr_detection_final2/weights/best.pt --input data/test_images --output outputs/
    ```

2.  **Output:** The script will process all images in the `test_images` folder and generate the two required submission files inside the `outputs/` directory.

---

## 4. Output Format

The inference script generates two JSON files as per the hackathon requirements.

#### **Stage 1: `submission_detection_1.json`**
This file contains the bounding box coordinates for all detected QR codes in each test image.

*   **Format:**
    ```json
    [
      {
        "image_id": "img201",
        "qrs": [
          {"bbox": [x_min, y_min, x_max, y_max]}
        ]
      }
    ]
    ```

#### **Stage 2 (Bonus): `submission_decoding_2.json`**
This file contains the bounding boxes plus the decoded string value from each QR code.

*   **Format:**
    ```json
    [
      {
        "image_id": "img201",
        "qrs": [
          {
            "bbox": [x_min, y_min, x_max, y_max],
            "value": "DecodedStringFromQR"
          }
        ]
      }
    ]
    ```