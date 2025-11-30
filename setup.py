# setup.py - Run this FIRST to create the project structure
import os
import pandas as pd
import numpy as np

def create_project_structure():
    """Create the necessary folders and files for the project"""
    
    # Create directories
    directories = ['data', 'models', 'notebooks', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}/")
    
    # Create sample data
    create_sample_data()
    
    print("\n✅ Project structure created successfully!")
    print("📁 Folders created: data/, models/, notebooks/, src/")
    print("📊 Sample dataset created: data/heart.csv")

def create_sample_data():
    """Create realistic sample heart disease data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'trestbps': np.random.normal(130, 17, n_samples).astype(int),
        'chol': np.random.normal(250, 55, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1]),
        'thalach': np.random.normal(150, 22, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'oldpeak': np.round(np.random.exponential(0.8, n_samples), 1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.6, 0.3]),
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'thal': np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.7, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable
    risk_score = (
        (df['age'] > 55) * 1.5 +
        (df['sex'] == 1) * 0.8 +
        (df['cp'] >= 2) * 1.2 +
        (df['trestbps'] > 140) * 1.0 +
        (df['chol'] > 240) * 1.1 +
        (df['fbs'] == 1) * 0.7 +
        (df['exang'] == 1) * 1.5 +
        (df['oldpeak'] > 1) * 1.3 +
        (df['ca'] >= 2) * 1.8 +
        (df['thal'] == 3) * 1.4
    )
    
    df['target'] = (risk_score > 6).astype(int)
    df.to_csv('data/heart.csv', index=False)

if __name__ == "__main__":
    create_project_structure()