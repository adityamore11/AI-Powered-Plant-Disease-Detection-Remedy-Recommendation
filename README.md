# 🌱 Plant Doctor AI

An end-to-end Streamlit application that classifies a newly uploaded leaf image with a locally saved EfficientNetB0 model, then retrieves practical disease information from a separate JSON knowledge base. It is designed as a beginner-friendly college and portfolio project.

> The PlantVillage/Kaggle dataset is used only to train and evaluate the model. Live predictions are performed locally using the trained model on new images uploaded by the user.

## Problem and features

Plant disease symptoms can be difficult to identify early. This project demonstrates an AI-assisted screening workflow:

```text
New leaf image → RGB resize/preparation → saved EfficientNetB0 model
→ class + confidence + top 3 → remedy knowledge base → Streamlit result
```

- Automatic class discovery for varied PlantVillage folder layouts
- Reproducible train/validation/test split when the dataset is not pre-split
- EfficientNetB0 transfer learning, augmentation, two training phases, and checkpoints
- Accuracy, precision, recall, F1, confusion matrix, and training-curve outputs after evaluation
- Confidence warning, safe fallback for missing remedy information, and optional Grad-CAM
- No web API or external frontend: the complete application is Streamlit and runs locally

## Project layout

```text
app.py                 Streamlit interface
config.py              project-wide settings
src/train.py           training and fine-tuning
src/evaluate.py        held-out test evaluation and plots
src/predict.py         reusable inference and Grad-CAM
src/preprocessing.py   PIL/RGB/resize preparation
src/remedy.py          JSON knowledge-base lookup
src/utils.py           class discovery and mappings
data/disease_info.json reviewed remedy content + template
models/                created model and class_names.json (ignored by Git)
results/               evaluation artifacts (ignored by Git)
tests/                 no-dataset unit tests
```

## Dataset setup

Download the PlantVillage dataset from Kaggle, extract it, and use this existing folder:

`/Users/aditya/Documents/plants_project/dataset`

Your extraction has a PlantVillage wrapper with variants such as `color`, `grayscale`, and `segmented`. The training script detects class folders automatically and prefers `color` when it is present. Do not copy the dataset into this Git repository; it is intentionally ignored.

The default project dataset folder is `dataset/`. To point training at your external location, use `--dataset /Users/aditya/Documents/plants_project/dataset` (shown below), or set `PLANT_DATASET_DIR`.

## Install and run

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the basic checks first (no dataset or trained model needed):

```bash
pytest -q
```

Train the model. This prints the actual discovered labels and only produces a model when valid class folders are present:

```bash
python src/train.py --dataset /Users/aditya/Documents/plants_project/dataset
```

Evaluate the best saved model and create `results/classification_report.json`, `results/confusion_matrix.png`, and `results/training_curves.png`:

```bash
python src/evaluate.py
```

For evaluation with the external dataset in this setup, set the location for the process:

```bash
PLANT_DATASET_DIR=/Users/aditya/Documents/plants_project/dataset python src/evaluate.py
```

Launch the web app after training:

```bash
streamlit run app.py
```

## How training and prediction work

Training creates a stable `models/class_names.json`; inference always reads this mapping, so a model output index is reliably translated to the original PlantVillage label. The model has an augmentation layer, ImageNet-initialized EfficientNetB0 base, global average pooling, dropout, and a softmax classifier. It trains the head first, then fine-tunes the last part of the base at a lower learning rate. Early stopping, learning-rate reduction, and best-validation-accuracy checkpointing protect against replacing the best model with a weaker epoch.

The Streamlit app caches the model in memory. A prediction under the configurable `CONFIDENCE_THRESHOLD` (default 80%) receives a warning and is not presented as certain. The user still sees the top three possibilities.

## Remedies and Grad-CAM

`data/disease_info.json` is deliberately separate from the ML model. Add a locally reviewed record for each class you intend to support using the included `__template__`; a missing record produces a safe, non-crashing expert-consultation message. Avoid adding pesticide doses or unverified chemical directions. Healthy predictions show care and monitoring guidance rather than disease remedies.

Grad-CAM is optional in the app. It visualizes regions that contributed strongly to the predicted class; it is a model explanation, not proof of a diagnosis.

## Screenshots and results

Add screenshots here after you train and run the Streamlit app. No accuracy, precision, recall, F1, or screenshots are included because training has not been run by this repository. Actual metrics are created only by `src/evaluate.py`.

## Limitations and future improvements

PlantVillage images are curated and may not represent field lighting, backgrounds, mixed symptoms, pests, or nutrient problems. Use it as screening support, not agronomic advice. Future work could include field-image validation, a fuller reviewed knowledge base, multilingual guidance, feedback collection, and crop-region-specific expert review.

## Resume / interview description

Built an end-to-end plant disease screening system using EfficientNetB0 transfer learning, data augmentation, fine-tuning, test-set evaluation, Grad-CAM explainability, and a Streamlit interface. The model predicts locally from newly uploaded leaf images while a separate knowledge base supplies safety-conscious remedy and prevention guidance.

In an interview, explain the separation of concerns: the neural network performs only image classification; JSON retrieval owns advice; the saved class mapping prevents prediction-label drift; and the confidence threshold keeps uncertain outputs honest.

## GitHub notes and disclaimer

The `.gitignore` excludes dataset images and trained model files. Commit a model only if its size, license, and reproducibility make that appropriate; otherwise release it separately with its matching `class_names.json`.

This project is an educational AI-assisted screening tool, not a substitute for diagnosis by a qualified agricultural professional. Follow local regulations, expert guidance, and product labels for all treatments.
