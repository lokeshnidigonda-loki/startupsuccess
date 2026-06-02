import joblib
import numpy as np

def predict_startup_performance(funding, team_size, growth_rate, sentiment_score):
    try:
        # Load optimized pipeline asset
        model = joblib.load('models/startup_model.pkl')
        
        # Arrange input vector structure
        features = np.array([[funding, team_size, growth_rate, sentiment_score]])
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        if prediction[0] == 1:
            return "High Success Potential", confidence
        else:
            return "Low Success / High Risk Operational Profile", confidence
            
    except FileNotFoundError:
        return "Model configuration file missing. Run train_model.py first.", 0.0