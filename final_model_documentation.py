"""
Final Model Documentation - Diabetes Prediction Model (Model 7)

This file documents the configuration and performance of our final diabetes prediction model.
The model achieved the best overall performance among all tested configurations.

Model Architecture:
-----------------
The model consists of two specialized classifiers combined through weighted voting:

1. Non-Diabetes Model (RandomForestClassifier):
   - Class weights: 4.85x for non-diabetes, 0.315x for diabetes
   - n_estimators: 2075
   - max_depth: 17
   - min_samples_split: 5
   - min_samples_leaf: 4
   - max_samples: 0.965
   - min_impurity_decrease: 0.000023
   - Feature selection: 35 features
   - Sampling ratio: 0.825

2. Diabetes Model (GradientBoostingClassifier):
   - Class weights: 0.275x for non-diabetes, 2.65x for diabetes
   - n_estimators: 1775
   - max_depth: 9
   - min_samples_split: 4
   - min_samples_leaf: 4
   - learning_rate: 0.0075
   - subsample: 0.985
   - min_impurity_decrease: 0.000023
   - Feature selection: 35 features
   - Sampling ratio: 0.555

3. Combined Model:
   - Voting weights: 48.5% non-diabetes, 51.5% diabetes
   - Optimal threshold: 0.2885

Performance Metrics:
------------------
Overall Performance:
- Accuracy: 98.6%
- ROC AUC: 0.986

Class 0 (Non-Diabetes):
- Precision: 99.3%
- Recall: 94.6%
- F1-score: 0.969
- True Negatives: 1501
- False Positives: 86

Class 1 (Diabetes):
- Precision: 98.3%
- Recall: 99.8%
- F1-score: 0.991
- True Positives: 5083
- False Negatives: 10

Top 15 Most Important Features:
1. Serum_Urate: 0.0742
2. Blood_Pressure_Diastolic: 0.0706
3. Age: 0.0646
4. HbA1c: 0.0563
5. GGT: 0.0561
6. Cholesterol_HDL: 0.0542
7. Cholesterol_LDL: 0.0529
8. Cholesterol_Total: 0.0525
9. Blood_Marker_Interaction: 0.0518
10. Dietary_Intake_Calories: 0.0507
11. Metabolic_Risk_Score: 0.0500
12. BP_Squared: 0.0483
13. Blood_Pressure_Systolic: 0.0478
14. Metabolic_Interaction: 0.0476
15. Fasting_Blood_Glucose: 0.0458

Feature Importance by Category:
- Demographics: 0.0646
- Physical Measurements: 0.0505
- Blood Tests: 0.0577
- Lifestyle: 0.0507

Key Strengths:
1. High overall accuracy (98.6%)
2. Excellent balance between precision and recall
3. Very few false negatives (10 cases)
4. Strong performance across both classes
5. Well-distributed feature importance

Model Selection Process:
----------------------
The model was selected after testing 8 different configurations:
- Model 1: Initial baseline (75.4% accuracy)
- Model 2: First optimization (86.6% accuracy)
- Model 3: Major improvements (97.7% accuracy)
- Model 4: Further refinements (98.5% accuracy)
- Model 5: Additional tuning (98.5% accuracy)
- Model 6: Combined best parameters (98.5% accuracy)
- Model 7: Final optimization (98.6% accuracy) - Selected as final model
- Model 8: Verification run (98.5% accuracy)

This model represents the optimal balance between precision and recall while maintaining high accuracy across both classes.
""" 