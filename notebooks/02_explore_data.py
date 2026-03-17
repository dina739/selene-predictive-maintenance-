"""
Explore the synthetic data - See what your machines look like!
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*50)
print("EXPLORING SYNTHETIC DATA")
print("="*50)

# Load the data you created
df = pd.read_csv('data/raw/synthetic_data.csv')
print(f"✅ Loaded {len(df)} rows of data")
print(f"✅ {df['machine'].nunique()} machines")
print(f"✅ Columns: {list(df.columns)}")

# Show basic statistics
print("\n📊 Data Statistics:")
print(df.describe())

# Check for missing values
print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")

# Look at one machine's entire life
machine_0 = df[df['machine'] == 0]

print(f"\n🔧 Machine 0: {len(machine_0)} cycles of data")
print(f"   Starts at cycle {machine_0['cycle'].min()}")
print(f"   Ends at cycle {machine_0['cycle'].max()}")
print(f"   Initial RUL: {machine_0['rul'].iloc[0]}")
print(f"   Final RUL: {machine_0['rul'].iloc[-1]}")

# Create visualizations
print("\n📈 Creating visualizations...")

plt.figure(figsize=(15, 8))

# Plot 1: Vibration over time
plt.subplot(2, 2, 1)
plt.plot(machine_0['cycle'], machine_0['vibration'], 'b-', linewidth=2)
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7, label='Degradation starts')
plt.title('Vibration Over Time')
plt.xlabel('Cycle')
plt.ylabel('Vibration (g)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Temperature over time
plt.subplot(2, 2, 2)
plt.plot(machine_0['cycle'], machine_0['temperature'], 'orange', linewidth=2)
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7)
plt.title('Temperature Over Time')
plt.xlabel('Cycle')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)

# Plot 3: RUL over time
plt.subplot(2, 2, 3)
plt.plot(machine_0['cycle'], machine_0['rul'], 'purple', linewidth=2)
plt.axvline(x=150, color='r', linestyle='--', alpha=0.7)
plt.title('Remaining Useful Life')
plt.xlabel('Cycle')
plt.ylabel('RUL (cycles)')
plt.grid(True, alpha=0.3)

# Plot 4: Vibration vs Temperature (colored by health)
plt.subplot(2, 2, 4)
healthy = machine_0[machine_0['cycle'] <= 150]
degraded = machine_0[machine_0['cycle'] > 150]

plt.scatter(healthy['vibration'], healthy['temperature'], 
            alpha=0.5, label='Healthy', c='blue', s=30)
plt.scatter(degraded['vibration'], degraded['temperature'], 
            alpha=0.5, label='Degraded', c='red', s=30)
plt.title('Vibration vs Temperature')
plt.xlabel('Vibration')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/data_exploration.png', dpi=150)
print(f"✅ Saved figure to reports/figures/data_exploration.png")

plt.show()

print("\n" + "="*50)
print("🎉 EXPLORATION COMPLETE! 🎉")
print("="*50)
print("\nWhat did you notice?")
print("- Vibration increases after cycle 150")
print("- Temperature trends upward")
print("- The relationship between sensors changes when machine degrades")
print("\nNext step: Build your first AI model!")