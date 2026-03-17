"""
Create synthetic machine data for practice
"""

import pandas as pd
import numpy as np
import os

print("="*50)
print("CREATING SYNTHETIC MACHINE DATA")
print("="*50)

# Create 10 machines, each running for 200 cycles
np.random.seed(42)
data = []

for machine_id in range(10):
    for cycle in range(200):
        # VIBRATION SENSOR
        vibration = 0.5 + 0.1 * np.sin(cycle/10)
        if cycle > 150:
            vibration = vibration * (1 + (cycle-150)/50)
        vibration = vibration + np.random.randn() * 0.05
        
        # TEMPERATURE SENSOR
        temperature = 70 + 0.2 * cycle + np.random.randn() * 2
        
        # REMAINING USEFUL LIFE
        rul = 200 - cycle
        
        # Save data (NO CURRENT sensor for now)
        data.append({
            'machine': machine_id,
            'cycle': cycle,
            'vibration': vibration,
            'temperature': temperature,
            'rul': rul
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Print summary
print(f"\n✅ Created {len(df)} rows of data")
print(f"✅ {df['machine'].nunique()} machines")
print(f"✅ {df['cycle'].nunique()} cycles per machine")
print(f"✅ Columns: {list(df.columns)}")

# Create data folder and save
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/synthetic_data.csv', index=False)
print(f"✅ Saved to data/raw/synthetic_data.csv")

# Show first 5 rows
print("\n📋 First 5 rows:")
print(df.head())

print("\n" + "="*50)
print("🎉 DATA CREATION COMPLETE! 🎉")
print("="*50)