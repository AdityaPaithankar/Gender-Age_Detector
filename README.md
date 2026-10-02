# Gender & Age Detection

A real-time computer vision application that detects faces from a webcam and predicts **gender and estimated age** using a custom multi-task CNN trained with TensorFlow/Keras.

## Features

- Real-time face detection using OpenCV
- Gender classification
- Age estimation
- Multi-task CNN architecture
- Batch Normalization and ReLU activation
- Global Average Pooling
- Dropout for regularization
- Fine-tuned model for improved performance
- Real-time webcam inference

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## Model Architecture

The model takes a `128 × 128 × 3` RGB face image as input.

```text
Input Image
    ↓
Conv2D (32)
    ↓
Batch Normalization
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D (64)
    ↓
Batch Normalization
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D (128)
    ↓
Batch Normalization
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D (256)
    ↓
Batch Normalization
    ↓
ReLU
    ↓
MaxPooling
    ↓
Global Average Pooling
    ↓
Dense + Dropout
    ↓
 ┌───────────────┐
 │               │
Gender          Age
 │               │
Classification  Regression
```

The model uses two outputs:

- **Gender:** Binary classification
- **Age:** Regression

## Loss Functions

The model uses:

- Binary Crossentropy for gender prediction
- Huber Loss for age prediction

The training process also uses weighted losses to balance the two tasks.

## Project Structure

```text
Gender Detector/
│
├── app/
│   └── webcam.py
│
├── model/
│   └── final_model.keras
│
├── notebooks/
│   └── training.ipynb
│
│
├── assets/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Gender-Detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the webcam detector:

```bash
python app/webcam.py
```

The application will open the webcam and detect faces.

The predicted result will be displayed above each detected face:

```text
Female | Age: 24
```

Press **Q** to close the application.

## Training

The training notebook is available in:

```text
notebooks/training.ipynb
```

The training pipeline includes:

1. Dataset preprocessing
2. Age and gender label preparation
3. Image normalization
4. Train/validation splitting
5. CNN training
6. Multi-task learning
7. Fine-tuning
8. Model evaluation
9. Saving the final trained model

## Real-Time Pipeline

```text
Webcam
   ↓
OpenCV
   ↓
Face Detection
   ↓
Face Crop
   ↓
Resize to 128 × 128
   ↓
RGB Conversion
   ↓
Normalization
   ↓
CNN
   ↓
Gender + Age
   ↓
Display Prediction
```

## Future Improvements

- Improve age estimation accuracy
- Add face tracking for smoother predictions
- Improve performance with a stronger face detector
- Add confidence scores
- Support multiple faces with independent tracking
- Build a web interface for deployment
- Optimize the model for real-time inference

## Disclaimer

Age and gender predictions are model estimates and may not always be accurate. Performance can vary depending on lighting, image quality, face angle, and dataset characteristics.
