#  AI-Powered Plant Disease Detection & Remedy Recommendation

An AI-powered web application that detects plant diseases from leaf images and provides disease-specific remedies, symptoms, and prevention guidelines.

The project uses **transfer learning with EfficientNetB0** for image classification and **Grad-CAM** for model explainability. A **Streamlit-based interface** allows users to upload an unseen leaf image and receive a prediction with confidence information and recommended remedies.

## Features

- Plant disease detection from leaf images
-  EfficientNetB0 transfer-learning classifier
-  Image preprocessing and data augmentation
-  Precision, Recall, F1-score and Confusion Matrix evaluation
-  Grad-CAM visualization for model explainability
-  Disease-specific remedy recommendations
-  Symptoms and prevention guidelines
-  Confidence-based warnings for uncertain predictions
-  Streamlit web interface
-  Structured disease knowledge base

##  Project Architecture

```text
AI-Powered-Plant-Disease-Detection-Remedy-Recommendation/
│
├── backend/
│   └── app/
│       ├── routes/
│       │   ├── health.py
│       │   └── prediction.py
│       ├── services/
│       │   ├── predictor.py
│       │   ├── preprocessing.py
│       │   └── remedy_service.py
│       ├── config.py
│       ├── main.py
│       └── schemas.py
│
├── data/
│   └── disease_knowledge.json
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── training/
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   └── requirements.txt
│
├── dataset/
│   └── Plant leaf image dataset
│
├── requirements.txt
├── .env.example
└── README.md
```

> **Note:** The dataset is excluded from Git using `.gitignore` because of its large size.

## Machine Learning Pipeline

```text
Leaf Image
    │
    ▼
Image Preprocessing
    │
    ├── Resize
    ├── Normalization
    └── Augmentation
    │
    ▼
EfficientNetB0
    │
    ▼
Disease Classification
    │
    ├── Predicted Disease
    └── Confidence Score
    │
    ▼
Confidence Check
    │
    ├── High Confidence → Prediction
    └── Low Confidence → Warning
    │
    ▼
Disease Knowledge Base
    │
    ├── Symptoms
    ├── Remedies
    └── Prevention
    │
    ▼
Final Recommendation
```

##  Model

The project uses **EfficientNetB0** with transfer learning.

### Training approach

1. Load and preprocess plant leaf images.
2. Resize images to the required input dimensions.
3. Normalize image pixel values.
4. Apply data augmentation such as rotation, zoom, and horizontal flipping.
5. Use an ImageNet-pretrained EfficientNetB0 backbone.
6. Fine-tune the model on plant disease classes.
7. Evaluate the model using multiple classification metrics.
8. Save the trained model for inference.

## 📊 Model Evaluation

The trained classifier is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics help evaluate both overall classification performance and class-specific performance.

##  Grad-CAM Explainability

The application uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions of the leaf image that contributed most to the model's prediction.

This helps users understand whether the model is focusing on relevant diseased regions of the leaf.

```text
Input Leaf Image
       │
       ▼
EfficientNetB0
       │
       ▼
Disease Prediction
       │
       ▼
Grad-CAM
       │
       ▼
Highlighted Disease Regions
```

##  Remedy Recommendation System

After identifying the predicted disease, the application retrieves relevant information from a structured knowledge base.

For each supported disease, the system can provide:

- Disease name
- Symptoms
- Recommended remedies
- Prevention guidelines

The knowledge base is stored in:

```text
data/disease_knowledge.json
```

This separates the machine-learning prediction system from the recommendation logic.

## Confidence-Based Warning

The application uses the model's prediction confidence to provide additional safety information.

If the model's confidence is below a configured threshold, the application warns the user that the prediction may be uncertain.

This helps avoid presenting low-confidence predictions as definitive diagnoses.

## Application Workflow

The project is designed with a user-friendly **Streamlit interface**.

Typical workflow:

```text
1. Upload leaf image
        ↓
2. Image preprocessing
        ↓
3. Disease prediction
        ↓
4. Confidence score
        ↓
5. Symptoms
        ↓
6. Recommended remedies
        ↓
7. Prevention guidelines
        ↓
8. Grad-CAM visualization
```

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Model development |
| EfficientNetB0 | Transfer-learning backbone |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| Streamlit | Web application |
| scikit-learn | Model evaluation |
| JSON | Disease knowledge base |
| Grad-CAM | Model explainability |

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/adityamore11/AI-Powered-Plant-Disease-Detection-Remedy-Recommendation.git
```

### 2. Navigate to the project

```bash
cd AI-Powered-Plant-Disease-Detection-Remedy-Recommendation
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 📁 Dataset

The project uses a plant leaf image dataset for training and evaluation.

The dataset is intentionally **not included in this GitHub repository** because of its large size.

Place the dataset inside:

```text
dataset/
```

Expected structure:

```text
dataset/
├── color/
├── grayscale/
└── segmented/
```

The `dataset/` directory is excluded using `.gitignore`.

##  Training

After placing the dataset in the appropriate directory, run the training pipeline:

```bash
python training/train.py
```

The training configuration can be adjusted in:

```text
training/config.py
```

## 🌐 Running the Application

Start the application's API/backend according to the project configuration.

For the Streamlit interface, use the Streamlit entry point included in the project:

```bash
streamlit run <streamlit-entry-file>.py
```

> Replace `<streamlit-entry-file>.py` with the actual Streamlit application file in your project.

## 📈 Future Improvements

- Support for additional plant species and diseases
- Larger and more diverse datasets
- Mobile-friendly deployment
- Cloud deployment
- Real-time camera-based disease detection
- Improved multilingual remedy recommendations
- Integration with weather and environmental data
- More advanced model explainability

##  Disclaimer

This application is intended for **educational and informational purposes**.

Predictions and recommendations should not be considered a substitute for professional agricultural advice. Users should verify disease identification and treatment recommendations with qualified agricultural experts when necessary.

##  Author

**Aditya More**

GitHub: https://github.com/adityamore11
