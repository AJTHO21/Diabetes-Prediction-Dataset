import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import json

# Set the style for all visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def create_custom_colormap():
    """Create a custom colormap for our visualizations"""
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f1c40f"]
    return LinearSegmentedColormap.from_list("custom", colors)

def plot_model_performance(y_true, y_pred_proba, threshold=0.2885):
    """Create a comprehensive performance visualization"""
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='#e74c3c', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    # Confusion Matrix
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    
    # Probability Distribution
    sns.histplot(data=pd.DataFrame({'Probability': y_pred_proba, 'Class': y_true}),
                x='Probability', hue='Class', bins=50, ax=ax4)
    ax4.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    ax4.set_title('Prediction Probability Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('Visualizations/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importances, feature_names, top_n=15):
    """Create a feature importance visualization"""
    plt.figure(figsize=(12, 8))
    
    # Convert feature names to array if it's a list
    feature_names = np.array(feature_names)
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[-top_n:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    # Create horizontal bar plot
    plt.barh(pos, importances[sorted_idx], align='center', color='#2a6091')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    
    # Add value labels
    for i, v in enumerate(importances[sorted_idx]):
        plt.text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('Visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_categories(importances, categories, feature_names):
    """Create a visualization of feature importance by category"""
    plt.figure(figsize=(10, 6))
    
    # Calculate average importance for each category
    category_importance = {}
    feature_names = np.array(feature_names)
    
    # Only consider numeric features (those starting with 'num__')
    numeric_indices = [i for i, name in enumerate(feature_names) if name.startswith('num__')]
    numeric_names = feature_names[numeric_indices]
    numeric_importances = importances[numeric_indices]
    
    # Map full feature names to their base names
    feature_map = {name: name.split('__')[-1] if '__' in name else name 
                  for name in numeric_names}
    
    for category, features in categories.items():
        # Find indices of features in this category
        feature_indices = []
        for f in features:
            for i, full_name in enumerate(numeric_names):
                base_name = feature_map[full_name]
                if f in base_name:
                    feature_indices.append(i)
        
        if feature_indices:
            category_importance[category] = np.mean(numeric_importances[feature_indices])
    
    if not category_importance:
        print("Warning: No categories were matched!")
        return
    
    # Create pie chart with earthy color palette
    colors = [
        '#898989',  # Demographics
        '#9f8534',  # Physical measurements
        '#7E8C54',  # Moss green
        '#8bb3b4',  # Lifestyle
        '#f0944d'   # Orange
    ]
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(list(category_importance.values()), 
            labels=list(category_importance.keys()),
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors[:len(category_importance)])
    
    # Set text colors to charcoal
    plt.setp(texts, color='#2c3e50')
    plt.setp(autotexts, color='#2c3e50')
    
    plt.axis('equal')
    plt.title('Feature Importance by Category')
    
    plt.tight_layout()
    plt.savefig('Visualizations/feature_categories.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_evolution(metrics_history):
    """Create a visualization of model performance evolution"""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy evolution
    plt.plot(range(1, len(metrics_history) + 1), 
             [m['accuracy'] for m in metrics_history],
             marker='o', label='Accuracy', color='#2ecc71')
    
    # Plot ROC AUC evolution
    plt.plot(range(1, len(metrics_history) + 1),
             [m['roc_auc'] for m in metrics_history],
             marker='s', label='ROC AUC', color='#3498db')
    
    plt.xlabel('Model Version')
    plt.ylabel('Score')
    plt.title('Model Performance Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Visualizations/model_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(y_true):
    """Create a visualization of class distribution"""
    plt.figure(figsize=(8, 6))
    
    # Calculate class distribution
    class_counts = pd.Series(y_true).value_counts()
    labels = ['Non-Diabetes', 'Diabetes']
    
    # Create pie chart with custom colors
    colors = ['#2c3e50', '#e67e22']  # Charcoal grey and orange
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(class_counts, labels=labels, autopct='%1.1f%%',
                                      startangle=90, colors=colors)
    
    # Set text colors
    plt.setp(texts, color='#2c3e50')  # Set all labels to charcoal grey
    plt.setp(autotexts[1], color='#2c3e50')  # Set diabetes percentage to charcoal grey
    plt.setp(autotexts[0], color='#e67e22')  # Set non-diabetes percentage to orange
    
    plt.axis('equal')
    plt.title('Class Distribution in Dataset')
    
    plt.tight_layout()
    plt.savefig('Visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(data, feature_names):
    """Create a correlation matrix visualization"""
    plt.figure(figsize=(15, 12))
    
    # Only consider numeric features
    numeric_features = [name for name in feature_names if name.startswith('num__')]
    base_names = [name.split('__')[-1] for name in numeric_features]
    
    # Select numeric columns from data
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    valid_cols = [col for col in numeric_cols if col in base_names]
    
    if not valid_cols:
        print("Warning: No valid numeric columns found for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr = data[valid_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap with custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('Visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the saved model metrics
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Convert lists back to numpy arrays
    y_true = np.array(metrics['y_true'])
    y_pred_proba = np.array(metrics['y_pred_proba'])
    feature_importances = np.array(metrics['feature_importances'])
    feature_names = metrics['feature_names']
    categories = metrics['categories']
    metrics_history = metrics['metrics_history']
    
    # Load the dataset for correlation matrix
    data = pd.read_csv('diabetes_dataset.csv')
    
    # Generate all visualizations
    print("Generating model performance visualization...")
    plot_model_performance(y_true, y_pred_proba)
    
    print("Generating feature importance visualization...")
    plot_feature_importance(feature_importances, feature_names)
    
    print("Generating feature categories visualization...")
    plot_feature_categories(feature_importances, categories, feature_names)
    
    print("Generating model evolution visualization...")
    plot_model_evolution(metrics_history)
    
    print("Generating class distribution visualization...")
    plot_class_distribution(y_true)
    
    print("Generating correlation matrix visualization...")
    plot_correlation_matrix(data, feature_names)
    
    print("All visualizations have been generated and saved to the Visualizations directory.")

if __name__ == "__main__":
    main() 