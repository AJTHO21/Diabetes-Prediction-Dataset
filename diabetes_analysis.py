import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Ensure Visualizations directory exists
os.makedirs('Visualizations', exist_ok=True)

def load_and_clean_data():
    """Load and clean the diabetes dataset."""
    # Load the dataset
    df = pd.read_csv('diabetes_dataset.csv')
    
    # Basic data cleaning
    df = df.dropna()  # Remove rows with missing values
    df = df.drop_duplicates()  # Remove duplicate rows
    
    # Convert categorical variables to appropriate types
    categorical_cols = ['Sex', 'Ethnicity', 'Smoking_Status']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

def create_demographic_visualizations(df):
    """Create visualizations for demographic data."""
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', bins=20, kde=True)
    plt.title('Age Distribution of Participants')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.savefig('Visualizations/age_distribution.png')
    plt.close()
    
    # Gender distribution
    plt.figure(figsize=(8, 6))
    df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.savefig('Visualizations/gender_distribution.png')
    plt.close()
    
    # Ethnicity distribution
    plt.figure(figsize=(12, 6))
    df['Ethnicity'].value_counts().plot(kind='bar')
    plt.title('Ethnicity Distribution')
    plt.xlabel('Ethnicity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Visualizations/ethnicity_distribution.png')
    plt.close()

def create_health_metrics_visualizations(df):
    """Create visualizations for health metrics."""
    # BMI distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='BMI', bins=20, kde=True)
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.savefig('Visualizations/bmi_distribution.png')
    plt.close()
    
    # Blood pressure relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Blood_Pressure_Systolic', y='Blood_Pressure_Diastolic')
    plt.title('Systolic vs Diastolic Blood Pressure')
    plt.xlabel('Systolic (mmHg)')
    plt.ylabel('Diastolic (mmHg)')
    plt.savefig('Visualizations/blood_pressure_relationship.png')
    plt.close()
    
    # Cholesterol levels
    plt.figure(figsize=(12, 6))
    cholesterol_cols = ['Cholesterol_Total', 'Cholesterol_HDL', 'Cholesterol_LDL']
    df[cholesterol_cols].boxplot()
    plt.title('Cholesterol Levels Distribution')
    plt.ylabel('mg/dL')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Visualizations/cholesterol_levels.png')
    plt.close()

def create_lifestyle_visualizations(df):
    """Create visualizations for lifestyle factors."""
    # Smoking status
    plt.figure(figsize=(8, 6))
    df['Smoking_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Smoking Status Distribution')
    plt.ylabel('')
    plt.savefig('Visualizations/smoking_status.png')
    plt.close()

def create_correlation_analysis(df):
    """Create correlation analysis and heatmap."""
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Health Metrics')
    plt.tight_layout()
    plt.savefig('Visualizations/correlation_matrix.png')
    plt.close()

def generate_statistical_analysis(df):
    """Generate statistical analysis of the dataset."""
    analysis_results = {}
    
    # Basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    analysis_results['basic_stats'] = df[numerical_cols].describe()
    
    # Correlation with key health metrics
    key_metrics = ['Fasting_Blood_Glucose', 'HbA1c']
    for metric in key_metrics:
        correlations = df[numerical_cols].corr()[metric].sort_values(ascending=False)
        analysis_results[f'{metric}_correlations'] = correlations
    
    return analysis_results

def main():
    """Main function to run the analysis."""
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    
    print("Creating demographic visualizations...")
    create_demographic_visualizations(df)
    
    print("Creating health metrics visualizations...")
    create_health_metrics_visualizations(df)
    
    print("Creating lifestyle visualizations...")
    create_lifestyle_visualizations(df)
    
    print("Creating correlation analysis...")
    create_correlation_analysis(df)
    
    print("Generating statistical analysis...")
    stats_results = generate_statistical_analysis(df)
    
    # Save statistical results to a text file
    with open('statistical_analysis.txt', 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("==========================\n\n")
        
        f.write("Basic Statistics:\n")
        f.write(str(stats_results['basic_stats']))
        f.write("\n\n")
        
        f.write("Fasting Blood Glucose Correlations:\n")
        f.write(str(stats_results['Fasting_Blood_Glucose_correlations']))
        f.write("\n\n")
        
        f.write("HbA1c Correlations:\n")
        f.write(str(stats_results['HbA1c_correlations']))
    
    print("Analysis complete! Check the generated files for results.")

if __name__ == "__main__":
    main() 