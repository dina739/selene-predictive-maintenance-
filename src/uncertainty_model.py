"""
Uncertainty Quantification using Conformal Prediction
Tells you HOW CONFIDENT the model is
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib

# Import for your MAPIE version
from mapie.regression import SplitConformalRegressor

print("="*50)
print("UNCERTAINTY QUANTIFICATION WITH CONFORMAL PREDICTION")
print("="*50)

# ============================================
# STEP 1: Load and Prepare Data
# ============================================

# Load data
df = pd.read_csv('data/raw/synthetic_data.csv')
print(f"\n📊 Loaded {len(df)} rows of data")

# Prepare features (X) and target (y)
# We'll predict Remaining Useful Life (RUL)
sensor_cols = ['vibration', 'temperature']
X = df[sensor_cols].values
y = df['rul'].values

print(f"Features: {sensor_cols}")
print(f"Target: RUL (Remaining Useful Life)")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n📊 Train set: {len(X_train)} samples, shape: {X_train.shape}")
print(f"Test set: {len(X_test)} samples, shape: {X_test.shape}")

# ============================================
# STEP 2: Train Base Model
# ============================================

print("\n🚀 Training base Random Forest model...")

# First, train a regular model
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)
base_score = base_model.score(X_test, y_test)
print(f"✅ Base model R² score: {base_score:.3f}")

# Test base model predictions
y_pred_base = base_model.predict(X_test)
print(f"✅ Base model predictions shape: {y_pred_base.shape}")

# ============================================
# STEP 3: Add Conformal Prediction (Uncertainty)
# ============================================

print("\n🎲 Adding conformal prediction for uncertainty...")

# Create MAPIE model with correct parameters for your version
mapie_model = SplitConformalRegressor(
    estimator=base_model, 
    confidence_level=0.9,  # This means 90% confidence
    prefit=True  # Since we already trained base_model
)

# For prefit=True, you need to call conformalize() first!
print("Calling conformalize...")
mapie_model.conformalize(X_train, y_train)
print("✅ Conformalize complete")

# Now get predictions and intervals
print("\n📊 Getting predictions with intervals...")

# Get predictions from base model (MAPIE predict might just return predictions)
y_pred = base_model.predict(X_test)

# Try to get intervals from MAPIE
if hasattr(mapie_model, 'conformity_scores_'):
    print("✅ Found conformity scores")
    
    # Get conformity scores (these are the errors on calibration data)
    conformity_scores = mapie_model.conformity_scores_
    print(f"Conformity scores shape: {conformity_scores.shape}")
    
    # Calculate quantile for desired confidence level
    alpha = 0.1  # For 90% confidence
    n_calib = len(conformity_scores)
    quantile_idx = int(np.ceil((1 - alpha) * (n_calib + 1)))
    quantile = np.sort(conformity_scores)[quantile_idx - 1]
    
    print(f"Quantile value: {quantile:.4f}")
    
    # Create prediction intervals
    lower = y_pred - quantile
    upper = y_pred + quantile
    
    print("✅ Created conformal prediction intervals")
    
else:
    print("⚠️ No conformity scores found. Using approximate intervals.")
    # Create approximate intervals based on prediction error
    errors = np.abs(y_pred_base - y_test)
    error_quantile = np.percentile(errors, 90)
    lower = y_pred - error_quantile
    upper = y_pred + error_quantile

print(f"Predictions shape: {y_pred.shape}")
print(f"Lower bounds shape: {lower.shape}")
print(f"Upper bounds shape: {upper.shape}")

# ============================================
# STEP 4: Evaluate Uncertainty
# ============================================

# Calculate coverage (how often true value falls in interval)
coverage = np.mean((lower <= y_test) & (y_test <= upper)) * 100
interval_width = np.mean(upper - lower)

print(f"\n📊 Uncertainty Results (90% confidence):")
print(f"   Coverage: {coverage:.1f}% (should be ~90%)")
print(f"   Average interval width: {interval_width:.2f} cycles")
print(f"   Prediction error (MAE): {np.mean(np.abs(y_pred - y_test)):.2f} cycles")

# ============================================
# STEP 5: Visualize Results
# ============================================

# Sort by true value for better visualization
sorted_idx = np.argsort(y_test)[:100]  # Show first 100 for clarity

plt.figure(figsize=(14, 6))

# Plot predictions with intervals
plt.plot(y_test[sorted_idx], 'b-', label='True RUL', linewidth=2, alpha=0.7)
plt.plot(y_pred[sorted_idx], 'r--', label='Predicted RUL', linewidth=2, alpha=0.7)
plt.fill_between(
    range(len(sorted_idx)),
    lower[sorted_idx],
    upper[sorted_idx],
    alpha=0.3,
    color='blue',
    label='90% Confidence Interval'
)

plt.xlabel('Sample (sorted)')
plt.ylabel('Remaining Useful Life (cycles)')
plt.title('RUL Prediction with Uncertainty Intervals')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/uncertainty_results.png', dpi=150)
plt.show()

# ============================================
# STEP 6: Save the Model
# ============================================

os.makedirs('models', exist_ok=True)
joblib.dump(mapie_model, 'models/uncertainty_model.pkl')
print("\n✅ Uncertainty model saved to models/uncertainty_model.pkl")

print("\n" + "="*50)
print("🎉 UNCERTAINTY MODEL COMPLETE! 🎉")
print("="*50)
print("\nWhat this means:")
print("- For any prediction, you get a range, not just a single number")
print("- 90% of the time, the true RUL falls in this range")
print("- Narrow intervals = high confidence")
print("- Wide intervals = low confidence - be careful!")