import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_startup_pipeline():
    # Ensure required directories exist for saving assets
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('static'):
        os.makedirs('static')

    # 1. Load Dataset
    if not os.path.exists('data/dataset.csv'):
        print("Error: data/dataset.csv not found! Please place your data file.")
        return
        
    df = pd.read_csv('data/dataset.csv')
    
    # 2. Extract Features matching your dataset layout
    X = df[['Funding_Amount', 'Team_Size', 'Growth_Rate', 'Sentiment_Score']]
    y = df['Success_Label']
    
    # 70:30 Train-test split specification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Initialize Random Forest & Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, None],
        'min_samples_split': [2, 5]
    }
    
    print("⚡ Optimizing Hyperparameters via GridSearchCV (Using cv=3 for smaller sample constraints)...")
    # Setting cv=3 dynamically accommodates the limited class distributions in our training split
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 4. Evaluation
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎉 Optimized Model Training Complete!")
    print(f"📊 Accuracy Achieved: {accuracy * 100:.2f}%")
    print("\n📋 Classification Report:\n", classification_report(y_test, predictions))
    
    # 5. Save Model Assets
    joblib.dump(best_model, 'models/startup_model.pkl')
    print("💾 Model saved securely to 'models/startup_model.pkl'")
    
    # 6. Track and Save Feature Importances via Matplotlib
    importances = best_model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title('Top Feature Importances in Startup Success Prediction')
    plt.barh(range(len(indices)), importances[indices], color='#6366f1', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance Weight')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    print("📈 Feature importance plot exported to 'static/feature_importance.png'")

if __name__ == "__main__":
    train_startup_pipeline()