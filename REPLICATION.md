# Diabetes Prediction Model Replication Guide

This guide provides step-by-step instructions to replicate our Final Model (Model 7) for diabetes prediction, which achieved 98.6% accuracy and 0.986 ROC AUC.

## Prerequisites

- Python 3.8 or higher
- 16GB RAM recommended
- Required packages (see requirements.txt)
- diabetes_dataset.csv

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The model requires `diabetes_dataset.csv` with the following features:

### Numeric Features
- Age
- BMI
- Blood Pressure
- Serum Urate
- Serum Creatinine
- Total Cholesterol
- HDL Cholesterol
- LDL Cholesterol
- Triglycerides
- HbA1c
- Fasting Glucose
- 2h Glucose
- Fasting Insulin
- 2h Insulin
- HOMA-IR
- HOMA-B
- QUICKI
- Matsuda Index
- Insulinogenic Index
- Disposition Index
- Stumvoll Index
- OGIS
- MCR
- Adiponectin
- Leptin
- Resistin
- TNF-alpha
- IL-6
- PAI-1
- CRP

### Categorical Features
- Gender
- Ethnicity
- Family History
- Smoking Status
- Alcohol Consumption
- Physical Activity
- Diet Quality

### Target Variable
- Diabetes Status (binary: 0 for non-diabetes, 1 for diabetes)

## Replication Steps

1. Ensure all prerequisites are met
2. Place `diabetes_dataset.csv` in the project root directory
3. Run the replication script:
   ```bash
   python model_replication_guide.py
   ```

The script will:
- Load and preprocess the data
- Train both specialized models
- Combine them with optimal weights
- Evaluate the performance
- Save the models and metrics

## Expected Output

The script will generate:
1. Trained models:
   - `final_non_diabetes_model.joblib`
   - `final_diabetes_model.joblib`
2. Evaluation metrics:
   - `model_metrics.json`

## Model Architecture

### Non-Diabetes Model (RandomForestClassifier)
- n_estimators: 2075
- max_depth: 17
- max_samples: 0.965
- class_weights: {0: class_weights[0] * 4.85, 1: class_weights[1] * 0.315}
- impurity_threshold: 0.000023

### Diabetes Model (GradientBoostingClassifier)
- n_estimators: 1775
- max_depth: 9
- learning_rate: 0.0075
- subsample: 0.985
- class_weights: {0: class_weights[0] * 0.275, 1: class_weights[1] * 2.65}

### Combined Model
- Voting weights: Non-Diabetes (0.485), Diabetes (0.515)
- Optimal threshold: 0.2885

## Expected Performance

- Accuracy: 98.6%
- ROC AUC: 0.986
- Precision: High for both classes
- Recall: High for both classes
- F1 Score: High for both classes

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Verify the dataset format matches the expected structure
3. Check available system memory (16GB recommended)
4. Ensure Python version is 3.8 or higher

## Contact

For questions or issues regarding replication, please open an issue in the repository. 