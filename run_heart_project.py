# run_heart_project.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

print("🚀 Starting Heart Disease Prediction Project...")

# Create sample data
def create_heart_data():
    np.random.seed(42)
    n_samples = 500
    
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
    
    # Create target based on risk factors
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
    return df

# Create and save data
print("📊 Creating heart disease dataset...")
df = create_heart_data()
os.makedirs('data', exist_ok=True)
df.to_csv('data/heart.csv', index=False)
print("✅ Dataset saved: data/heart.csv")

# Prepare data for training
print("⚙️ Preprocessing data...")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2%}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/heart_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("💾 Model saved: models/heart_model.pkl")

# Test prediction
sample_patient = [[55, 1, 2, 140, 250, 1, 1, 150, 1, 2.5, 1, 2, 2]]
sample_scaled = scaler.transform(sample_patient)
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print(f"\n🎯 Sample Prediction:")
print(f"   Heart Disease: {'YES' if prediction == 1 else 'NO'}")
print(f"   Probability: {probability:.2%}")
print(f"   Risk Level: {'HIGH' if probability > 0.7 else 'MODERATE' if probability > 0.3 else 'LOW'}")

print("\n🎉 Project setup completed successfully!")
print("Next: Run 'streamlit run app.py' to start the web app")