"""
Test the autoencoder on ALL data (including failures)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pickle

print("="*50)
print("TESTING AUTOENCODER ON ALL DATA")
print("="*50)

# Load the saved model (with weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load checkpoint (weights_only=False is safe because you created this file)
checkpoint = torch.load('models/autoencoder_complete.pth', map_location=device, weights_only=False)

# Recreate the model
from src.autoencoder_model import Autoencoder
model = Autoencoder(input_dim=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load the scaler
scaler = checkpoint['scaler']

# Load all data
df = pd.read_csv('data/raw/synthetic_data.csv')
sensor_cols = ['vibration', 'temperature']
X_all = df[sensor_cols].values

# Scale the data
X_scaled = scaler.transform(X_all)

# Convert to tensor
X_tensor = torch.FloatTensor(X_scaled).to(device)

# Calculate reconstruction errors
print("📊 Calculating anomaly scores...")
with torch.no_grad():
    reconstructions = model(X_tensor)
    errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()

# Add to dataframe
df['anomaly_score'] = errors

# Calculate threshold from healthy data (first 150 cycles of machine 0)
healthy_data = df[(df['machine'] == 0) & (df['cycle'] <= 150)]
threshold = np.percentile(healthy_data['anomaly_score'], 95)
print(f"✅ Anomaly threshold: {threshold:.4f} (95th percentile of healthy data)")

# Mark anomalies
df['is_anomaly'] = df['anomaly_score'] > threshold

print(f"\n📊 Results:")
print(f"   Total samples: {len(df)}")
print(f"   Anomalies detected: {df['is_anomaly'].sum()}")
print(f"   Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")

# Plot for machine 0
machine_0 = df[df['machine'] == 0]

plt.figure(figsize=(14, 10))

# Plot 1: Vibration
plt.subplot(3, 1, 1)
plt.plot(machine_0['cycle'], machine_0['vibration'], 'b-', linewidth=2)
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7, label='Degradation starts')
plt.title('Vibration Over Time')
plt.ylabel('Vibration')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Temperature
plt.subplot(3, 1, 2)
plt.plot(machine_0['cycle'], machine_0['temperature'], 'orange', linewidth=2)
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7)
plt.title('Temperature Over Time')
plt.ylabel('Temperature')
plt.grid(True, alpha=0.3)

# Plot 3: Anomaly Score
plt.subplot(3, 1, 3)
plt.plot(machine_0['cycle'], machine_0['anomaly_score'], 'r-', linewidth=2, label='Anomaly Score')
plt.axhline(y=threshold, color='g', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7)
plt.fill_between(machine_0['cycle'], 0, machine_0['anomaly_score'], 
                 where=(machine_0['anomaly_score'] > threshold),
                 color='red', alpha=0.3, label='Anomaly Detected')
plt.title('Anomaly Detection Results')
plt.xlabel('Cycle')
plt.ylabel('Anomaly Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/anomaly_detection_results.png', dpi=150)
plt.show()

# Calculate how many anomalies detected after degradation
after_150 = machine_0[machine_0['cycle'] > 150]
detected = after_150[after_150['anomaly_score'] > threshold]
print(f"\n🎯 Performance:")
print(f"   Cycles after degradation: {len(after_150)}")
print(f"   Anomalies detected: {len(detected)}")
print(f"   Detection rate: {len(detected)/len(after_150)*100:.1f}%")

print("\n" + "="*50)
print("🎉 TESTING COMPLETE! 🎉")
print("="*50)