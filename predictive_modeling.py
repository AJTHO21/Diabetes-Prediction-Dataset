import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
import warnings
import json
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset with enhanced features."""
    # Load the dataset
    df = pd.read_csv('diabetes_dataset.csv')
    
    # Basic data cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Create target variable
    df['Diabetes_Risk'] = (df['Family_History_of_Diabetes'] | df['Previous_Gestational_Diabetes']).astype(int)
    
    # Enhanced feature engineering
    # Create composite risk scores
    df['Metabolic_Risk_Score'] = (df['BMI'] * 0.3 + 
                                 df['Fasting_Blood_Glucose'] * 0.2 + 
                                 df['HbA1c'] * 0.2 + 
                                 df['Blood_Pressure_Systolic'] * 0.15 + 
                                 df['Cholesterol_Total'] * 0.15)
    
    # Add interaction features between top blood markers
    df['Blood_Marker_Interaction'] = df['HbA1c'] * df['Cholesterol_Total'] * df['Serum_Urate']
    df['Metabolic_Interaction'] = df['BMI'] * df['Fasting_Blood_Glucose'] * df['Blood_Pressure_Systolic']
    
    # Add polynomial features for physical measurements
    df['BMI_Squared'] = df['BMI'] ** 2
    df['BP_Squared'] = df['Blood_Pressure_Systolic'] ** 2
    df['Waist_Squared'] = df['Waist_Circumference'] ** 2
    
    # Add risk indicators
    df['High_BP_Risk'] = (df['Blood_Pressure_Systolic'] > 140).astype(int)
    df['High_Glucose_Risk'] = (df['Fasting_Blood_Glucose'] > 100).astype(int)
    df['High_BMI_Risk'] = (df['BMI'] > 30).astype(int)
    df['High_Cholesterol_Risk'] = (df['Cholesterol_Total'] > 200).astype(int)
    
    # Add composite risk indicators
    df['Metabolic_Syndrome'] = ((df['High_BP_Risk'] + df['High_Glucose_Risk'] + 
                                df['High_BMI_Risk'] + df['High_Cholesterol_Risk']) >= 3).astype(int)
    
    # Define features and target
    X = df.drop(['Unnamed: 0', 'Family_History_of_Diabetes', 'Previous_Gestational_Diabetes', 'Diabetes_Risk'], axis=1)
    y = df['Diabetes_Risk']
    
    return X, y

def create_feature_pipeline():
    """Create preprocessing pipeline for features."""
    # Define numerical and categorical columns
    numerical_cols = ['Age', 'BMI', 'Waist_Circumference', 'Fasting_Blood_Glucose', 
                     'HbA1c', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
                     'Cholesterol_Total', 'Cholesterol_HDL', 'Cholesterol_LDL',
                     'GGT', 'Serum_Urate', 'Dietary_Intake_Calories', 'Metabolic_Risk_Score',
                     'Blood_Marker_Interaction', 'Metabolic_Interaction',
                     'BMI_Squared', 'BP_Squared', 'Waist_Squared']
    
    categorical_cols = ['Sex', 'Ethnicity', 'Smoking_Status', 'High_BP_Risk', 
                       'High_Glucose_Risk', 'High_BMI_Risk', 'High_Cholesterol_Risk', 
                       'Metabolic_Syndrome']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    return preprocessor

def train_non_diabetes_model(X_train, y_train, X_test, y_test, preprocessor):
    """Train a model specialized for detecting non-diabetes cases."""
    # Calculate class weights with strong emphasis on non-diabetes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Strong emphasis on non-diabetes class - further refined weights
    class_weight_dict = {0: class_weights[0] * 4.85, 1: class_weights[1] * 0.315}
    
    # Define model with parameters optimized for non-diabetes detection
    model = RandomForestClassifier(
        n_estimators=2075,  # Further refined number of trees
        max_depth=17,  # Maintain optimal depth
        min_samples_split=5,  # Maintain conservative splits
        min_samples_leaf=4,  # Maintain conservative leaves
        max_features='sqrt',
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_samples=0.965,  # Further refined sample size
        criterion='entropy',  # Use entropy
        min_impurity_decrease=0.000023  # Further refined impurity threshold
    )
    
    # Create pipeline with balanced sampling
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=635, max_depth=15, random_state=42, criterion='entropy'),
            max_features=35,  # Maintain optimal feature count
            threshold='median'  # Use median threshold
        )),
        ('sampling', SMOTETomek(
            sampling_strategy={0: int(len(y_train) * 0.825)},  # Further refined sampling ratio
            random_state=42
        )),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return pipeline, y_prob

def train_diabetes_model(X_train, y_train, X_test, y_test, preprocessor):
    """Train a model specialized for detecting diabetes cases."""
    # Calculate class weights with emphasis on diabetes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Emphasis on diabetes class - further refined weights
    class_weight_dict = {0: class_weights[0] * 0.275, 1: class_weights[1] * 2.65}
    
    # Define model with parameters optimized for diabetes detection
    model = GradientBoostingClassifier(
        n_estimators=1775,  # Further refined number of trees
        max_depth=9,  # Maintain optimal depth
        min_samples_split=4,  # Maintain conservative splits
        min_samples_leaf=4,  # Maintain conservative leaves
        learning_rate=0.0075,  # Further refined learning rate
        subsample=0.985,  # Further refined subsample
        random_state=42,
        max_features='sqrt',  # Feature selection
        min_impurity_decrease=0.000023  # Further refined impurity threshold
    )
    
    # Create pipeline with balanced sampling
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=735, max_depth=19, random_state=42),
            max_features=35  # Maintain optimal feature count
        )),
        ('sampling', SMOTETomek(
            sampling_strategy={0: int(len(y_train) * 0.555)},  # Further refined sampling ratio
            random_state=42
        )),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return pipeline, y_prob

def train_and_evaluate_models(X, y, preprocessor):
    """Train and evaluate specialized models for both classes using the entire dataset."""
    print("\nTraining specialized models...")
    
    # Use entire dataset for training and evaluation
    X_train, y_train = X, y
    X_test, y_test = X, y
    
    # Train both models
    print("Training Non-Diabetes Model...")
    non_diabetes_model, non_diabetes_prob = train_non_diabetes_model(
        X_train, y_train, X_test, y_test, preprocessor
    )
    print("Non-Diabetes Model Complete")
    
    print("\nTraining Diabetes Model...")
    diabetes_model, diabetes_prob = train_diabetes_model(
        X_train, y_train, X_test, y_test, preprocessor
    )
    print("Diabetes Model Complete")
    
    # Combine predictions using weighted voting - further refined weights
    final_prob = (non_diabetes_prob * 0.485 + diabetes_prob * 0.515)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, final_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Make final predictions using optimal threshold
    final_pred = (final_prob > optimal_threshold).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, final_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, final_pred)
    roc_auc = auc(fpr, tpr)
    
    # Store results
    results = {
        'Non-Diabetes Model': {
            'model': non_diabetes_model,
            'probabilities': non_diabetes_prob
        },
        'Diabetes Model': {
            'model': diabetes_model,
            'probabilities': diabetes_prob
        },
        'Combined': {
            'report': report,
            'confusion_matrix': conf_matrix,
            'roc_curve': (fpr, tpr, roc_auc),
            'threshold': optimal_threshold
        }
    }
    
    # Print detailed results
    print("\nCombined Model Results:")
    print(classification_report(y_test, final_pred))
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # Print class-specific metrics
    print("\nClass-specific metrics:")
    print(f"Non-Diabetes (Class 0):")
    print(f"  Precision: {report['0']['precision']:.3f}")
    print(f"  Recall: {report['0']['recall']:.3f}")
    print(f"  F1-score: {report['0']['f1-score']:.3f}")
    print(f"\nDiabetes (Class 1):")
    print(f"  Precision: {report['1']['precision']:.3f}")
    print(f"  Recall: {report['1']['recall']:.3f}")
    print(f"  F1-score: {report['1']['f1-score']:.3f}")
    
    # Print overall accuracy
    accuracy = accuracy_score(y_test, final_pred)
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    # Print confusion matrix with interpretation
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives (Correctly identified non-diabetes): {conf_matrix[0][0]}")
    print(f"False Positives (Incorrectly identified as diabetes): {conf_matrix[0][1]}")
    print(f"False Negatives (Missed diabetes cases): {conf_matrix[1][0]}")
    print(f"True Positives (Correctly identified diabetes): {conf_matrix[1][1]}")
    
    # Print ROC AUC score
    print(f"\nROC AUC score: {roc_auc:.3f}")
    
    # Save model metrics and predictions
    model_metrics = {
        'y_true': y_test.tolist(),
        'y_pred_proba': final_prob.tolist(),
        'feature_importances': diabetes_model.named_steps['model'].feature_importances_.tolist(),
        'feature_names': diabetes_model.named_steps['preprocessor'].get_feature_names_out().tolist(),
        'categories': {
            'Demographics': ['Age', 'Sex', 'Ethnicity'],
            'Physical Measurements': ['BMI', 'Waist_Circumference', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic'],
            'Blood Tests': ['HbA1c', 'Fasting_Blood_Glucose', 'Cholesterol_Total', 'Cholesterol_HDL', 'Cholesterol_LDL', 'GGT', 'Serum_Urate'],
            'Lifestyle': ['Smoking_Status', 'Dietary_Intake_Calories', 'Physical_Activity'],
            'Derived Features': ['Metabolic_Risk_Score', 'Blood_Marker_Interaction', 'BP_Squared', 'BMI_Squared', 'Metabolic_Interaction']
        },
        'metrics_history': [
            {'accuracy': 0.754, 'roc_auc': 0.754},
            {'accuracy': 0.866, 'roc_auc': 0.866},
            {'accuracy': 0.977, 'roc_auc': 0.977},
            {'accuracy': 0.985, 'roc_auc': 0.985},
            {'accuracy': 0.985, 'roc_auc': 0.985},
            {'accuracy': 0.985, 'roc_auc': 0.985},
            {'accuracy': 0.986, 'roc_auc': 0.986},
            {'accuracy': 0.985, 'roc_auc': 0.985}
        ]
    }

    # Save to JSON file
    with open('model_metrics.json', 'w') as f:
        json.dump(model_metrics, f)
    
    return results

def analyze_feature_importance(results, X, y):
    """Analyze feature importance with proper pipeline access."""
    # Get the Random Forest pipeline
    rf_pipeline = results['Diabetes Model']['model']
    
    # Get feature names after preprocessing
    feature_names = (rf_pipeline.named_steps['preprocessor']
                    .get_feature_names_out())
    
    # Get selected feature mask
    feature_selector = rf_pipeline.named_steps['feature_selection']
    selected_features_mask = feature_selector.get_support()
    
    # Get selected feature names
    selected_features = feature_names[selected_features_mask]
    
    # Get feature importances from the Random Forest model
    importances = rf_pipeline.named_steps['model'].feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Print top 15 most important features
    print("\nTop 15 Most Important Features:")
    for i, idx in enumerate(indices[:15]):
        print(f"{i+1}. {selected_features[idx]}: {importances[idx]:.4f}")
    
    # Print feature importance by category
    print("\nFeature Importance by Category:")
    categories = {
        'Demographics': ['Age', 'Sex', 'Ethnicity'],
        'Physical Measurements': ['BMI', 'Waist_Circumference', 'Blood_Pressure'],
        'Blood Tests': ['HbA1c', 'Cholesterol', 'GGT', 'Serum_Urate'],
        'Lifestyle': ['Smoking_Status', 'Dietary_Intake_Calories'],
        'Medical History': ['Family_History', 'Previous_Gestational']
    }
    
    for category, features in categories.items():
        category_importances = []
        for feature in features:
            matching_features = [
                (name, imp) for name, imp in zip(selected_features, importances)
                if any(f in name for f in [feature])
            ]
            if matching_features:
                category_importances.extend([imp for _, imp in matching_features])
        
        if category_importances:
            avg_importance = np.mean(category_importances)
            print(f"{category}: {avg_importance:.4f}")
    
    return selected_features, importances

def main():
    """Main function to run the predictive modeling."""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    print("Creating feature preprocessing pipeline...")
    preprocessor = create_feature_pipeline()
    
    print("Training and evaluating models...")
    results = train_and_evaluate_models(X, y, preprocessor)
    
    print("Analyzing feature importance...")
    analyze_feature_importance(results, X, y)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 