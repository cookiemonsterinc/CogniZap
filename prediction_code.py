import joblib
import numpy as np

# Load your trained model
model = joblib.load('svm_fatigue_model.pkl')
scaler = joblib.load('scaler_fatigue.pkl')

def predict_fatigue(eeg_features):
    """
    Input: Dictionary with 22 EEG features from headset
    Output: Fatigue prediction
    """
    
    # ONLY the 22 EEG features (IN THIS EXACT ORDER!)
    feature_names = [
        'a_mean', 'a_std',           # Alpha
        'b_mean', 'b_std',           # Beta
        'd_mean', 'd_std',           # Delta
        'g_mean', 'g_std',           # Gamma
        't_mean', 't_std',           # Theta
        'Aa_mean', 'Aa_std',         # Absolute Alpha
        'Ab_mean', 'Ab_std',         # Absolute Beta
        'Ad_mean', 'Ad_std',         # Absolute Delta
        'Ag_mean', 'Ag_std',         # Absolute Gamma
        'At_mean', 'At_std',         # Absolute Theta
        'noise_mean', 'noise_std'    # Noise
    ]
    
    # Convert to array
    features_array = np.array([[eeg_features[name] for name in feature_names]])
    
    # Scale (using training scaler)
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction] * 100
    
    return {
        'fatigued': bool(prediction),  # True or False
        'confidence': confidence,       # 0-100%
        'prob_not_fatigued': probabilities[0] * 100,
        'prob_fatigued': probabilities[1] * 100
    }


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Example: Your friend gives you this from OpenBCI
    eeg_data = {
        'a_mean': 0.25, 'a_std': 0.12,
        'b_mean': 0.18, 'b_std': 0.09,
        'd_mean': 0.35, 'd_std': 0.15,
        'g_mean': 0.08, 'g_std': 0.04,
        't_mean': 0.22, 't_std': 0.11,
        'Aa_mean': 1.5, 'Aa_std': 0.8,
        'Ab_mean': 1.2, 'Ab_std': 0.6,
        'Ad_mean': 2.1, 'Ad_std': 1.1,
        'Ag_mean': 0.5, 'Ag_std': 0.3,
        'At_mean': 1.8, 'At_std': 0.9,
        'noise_mean': 0.05, 'noise_std': 0.02
    }
    
    # Make prediction
    result = predict_fatigue(eeg_data)
    
    # Show result
    if result['fatigued']:
        print(f"⚠️  FATIGUED (Confidence: {result['confidence']:.1f}%)")
    else:
        print(f"✅ NOT FATIGUED (Confidence: {result['confidence']:.1f}%)")
