"""
Simple Autoencoder Model - Learns "normal" machine behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

print("="*50)
print("BUILDING AUTOENCODER MODEL")
print("="*50)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# STEP 1: Define the Autoencoder Model
# ============================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================
# STEP 2: Load and Prepare Data
# ============================================

# Load data
df = pd.read_csv('data/raw/synthetic_data.csv')
print(f"\n📊 Loaded {len(df)} rows of data")

# Use only vibration and temperature (2 sensors)
sensor_cols = ['vibration', 'temperature']
X = df[sensor_cols].values
print(f"Using sensors: {sensor_cols}")

# Use ONLY healthy data for training (first 150 cycles of each machine)
healthy_data = []
for machine in df['machine'].unique():
    machine_data = df[df['machine'] == machine]
    healthy_part = machine_data.iloc[:150]  # First 150 cycles are healthy
    healthy_data.append(healthy_part[sensor_cols].values)

X_healthy = np.vstack(healthy_data)
print(f"Training on {len(X_healthy)} healthy samples")

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_healthy)

# Split into train and validation sets
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)

# ============================================
# STEP 3: Train the Model
# ============================================

# Create model
model = Autoencoder(input_dim=len(sensor_cols)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🚀 Training started...")

epochs = 100
batch_size = 32
train_loader = torch.utils.data.DataLoader(X_train_t, batch_size=batch_size, shuffle=True)

train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_loss = criterion(val_output, X_val_t).item()
        val_losses.append(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")

print("\n✅ Training complete!")

# ============================================
# STEP 4: Save the Model
# ============================================

os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'sensor_cols': sensor_cols
}, 'models/autoencoder_complete.pth')
print("✅ Model saved to models/autoencoder_complete.pth")

# ============================================
# STEP 5: Plot Training Progress
# ============================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reports/figures/training_progress.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("🎉 MODEL BUILD COMPLETE! 🎉")
print("="*50)
print("\nNext step: Test your model on ALL data (including failures)")