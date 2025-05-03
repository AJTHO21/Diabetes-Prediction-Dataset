# Diabetes Prediction Model

A highly accurate diabetes prediction model achieving 98.6% accuracy and 0.986 ROC AUC through an ensemble of specialized models.

## Overview

This project implements a sophisticated diabetes prediction system using machine learning. The model combines two specialized classifiers:
- A RandomForestClassifier optimized for non-diabetes cases
- A GradientBoostingClassifier optimized for diabetes cases

![Model Performance](Visualizations/model_performance.png)

Description:
This composite visualization provides a comprehensive overview of the model’s predictive performance using four key plots:
ROC Curve: Shows the trade-off between sensitivity (true positive rate) and specificity (false positive rate). The area under the curve (AUC) quantifies the model’s ability to distinguish between diabetes and non-diabetes cases. An AUC close to 1.0 indicates excellent discrimination.
Precision-Recall Curve: Highlights the balance between precision (positive predictive value) and recall (sensitivity). This is especially important in imbalanced datasets, as it shows how well the model identifies true diabetes cases without too many false positives.
Confusion Matrix: Displays the counts of true positives, true negatives, false positives, and false negatives. This helps you see where the model is making correct and incorrect predictions, and whether errors are skewed toward one class.
Prediction Probability Distribution: Visualizes the distribution of predicted probabilities for each class, with a vertical line indicating the decision threshold. This helps assess how confidently the model separates diabetes from non-diabetes cases.

Interpretation:
High ROC AUC and PR AUC values confirm strong model performance.
The confusion matrix should show high numbers on the diagonal (correct predictions).
The probability distribution should show clear separation between classes, indicating confident predictions.

## Documentation

For comprehensive technical details, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), which covers:
- Problem statement and clinical context
- Detailed dataset analysis
- Data preprocessing and feature engineering
- Mathematical approaches and model architecture
- Model validation and performance metrics
- Limitations and future work

## Dataset Analysis

The model uses a comprehensive dataset with features across multiple categories:

### Feature Categories
![Feature Categories](Visualizations/feature_categories.png)

Description:
This pie chart summarizes the average importance of features grouped by category (e.g., Demographics, Physical Measurements, Blood Tests, Lifestyle, Derived Features).
Interpretation:
The size of each slice shows the proportion of total model importance attributed to each category.
For example, a large “Blood Tests” slice indicates that blood biomarkers are highly predictive in this model.

### Class Distribution
![Class Distribution](Visualizations/class_distribution.png)

Description:
This pie chart shows the proportion of diabetes and non-diabetes cases in the dataset.
Interpretation:
The chart reveals any class imbalance, which is important for understanding model evaluation metrics.
A significant imbalance (e.g., more diabetes than non-diabetes cases) can affect precision, recall, and the need for class weighting or sampling strategies.

### Feature Importance
![Feature Importance](Visualizations/feature_importance.png)

Description:
This horizontal bar chart ranks the top 15 most important features used by the model, based on their contribution to the model’s predictions.
Interpretation:
Features at the top (e.g., Serum Urate, Blood Pressure, Age) have the greatest influence on the model’s decision-making.
The length of each bar represents the relative importance; longer bars mean greater impact.
This visualization helps identify which clinical measurements and patient characteristics are most predictive of diabetes risk in your dataset.

### Feature Correlations
![Correlation Matrix](Visualizations/correlation_matrix.png)

Description:
This heatmap displays the pairwise correlations between numeric features in the dataset.
Interpretation:
Darker or more intense colors indicate stronger positive or negative correlations.
High correlations between features may suggest redundancy, which can inform feature selection or engineering.
This visualization helps ensure that the model is not relying on highly collinear features, which could reduce generalizability.

## Methodology

### Model Evolution and Performance Progression
The model underwent 8 iterations of refinement, each bringing significant improvements:
![Model Evolution](Visualizations/model_evolution.png)

Description:
This line plot tracks the evolution of model performance (Accuracy and ROC AUC) across eight major versions, with annotations for each version’s key changes and actual performance values.
Interpretation:
The upward trend demonstrates how iterative improvements (feature engineering, model tuning, ensemble methods) led to higher accuracy and discrimination.
The table below the plot details what was changed in each version, providing transparency and insight into the model development process.

The project follows a systematic approach:
1. Data preprocessing and feature engineering
2. Specialized model development
3. Ensemble combination
4. Performance optimization

## Results

The Final Model (Model 7) achieved:
- Overall accuracy: 98.6%
- ROC AUC: 0.986
- High precision and recall for both classes

## Key Findings

1. **Predictive Power**: The model demonstrates exceptional predictive capabilities across all metrics.
2. **Feature Importance**: Serum Urate, Blood Pressure, and Age emerged as the most significant predictors.
3. **Model Robustness**: The ensemble approach provides stability and reduces overfitting.
4. **Clinical Relevance**: The model's high accuracy makes it suitable for clinical decision support.

## Usage

For detailed instructions on replicating the model, see [REPLICATION.md](REPLICATION.md).

## Future Improvements

1. Integration with electronic health records
2. Real-time prediction capabilities
3. Additional feature engineering
4. Model deployment as a web service

## Conclusion

This diabetes prediction model represents a significant advancement in medical machine learning, offering high accuracy and robust performance across various metrics. The ensemble approach and careful feature selection contribute to its success.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
