"""
SELENE - Hugging Face Spaces Deployment
Predictive Maintenance with Self-Supervised Learning + Explainable AI
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.autoencoder_model import Autoencoder
from src.explainer import MaintenanceExplainer

# Load models (cached for performance)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    """Load all SELENE models"""
    # Load autoencoder
    autoencoder = Autoencoder(input_dim=2).to(device)
    checkpoint = torch.load('models/autoencoder_complete.pth', 
                           map_location=device, 
                           weights_only=False)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    scaler = checkpoint['scaler']
    
    # Load explainer
    explainer = MaintenanceExplainer()
    
    return autoencoder, scaler, explainer

autoencoder, scaler, explainer = load_models()

def predict_anomaly(vibration, temperature, cycle):
    """Main prediction function"""
    
    # Prepare input
    input_data = np.array([[vibration, temperature]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled).to(device)
    
    # Get anomaly score
    with torch.no_grad():
        reconstruction = autoencoder(input_tensor)
        anomaly_score = torch.mean((input_tensor - reconstruction) ** 2).item()
    
    # Threshold (from your training)
    threshold = 0.5
    rul = max(0, 200 - cycle)
    
    # Get explanation
    sensors = {'vibration': vibration, 'temperature': temperature}
    explanation = explainer.explain_anomaly(
        machine_id=1,
        cycle=cycle,
        sensors=sensors,
        anomaly_score=anomaly_score,
        threshold=threshold,
        rul=rul
    )
    
    # Determine status
    if anomaly_score > threshold:
        status = "⚠️ ANOMALY DETECTED"
        color = "red"
    else:
        status = "✅ NORMAL OPERATION"
        color = "green"
    
    return status, f"Score: {anomaly_score:.3f}", explanation

# Create Gradio interface
with gr.Blocks(title="SELENE Predictive Maintenance", 
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 SELENE: AI-Powered Predictive Maintenance
    *Self-Supervised Learning + Uncertainty Quantification + Explainable AI*
    
    Enter machine sensor readings to get real-time analysis and maintenance recommendations.
    """)
    
    with gr.Row():
        with gr.Column():
            vibration = gr.Slider(0.2, 1.5, value=0.52, step=0.01, 
                                 label="Vibration (g)")
            temperature = gr.Slider(60, 120, value=78.3, step=0.1, 
                                   label="Temperature (°C)")
            cycle = gr.Slider(0, 200, value=100, step=1, 
                             label="Operating Cycle")
            analyze_btn = gr.Button("Analyze Machine", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status")
            score = gr.Textbox(label="Anomaly Score")
    
    explanation = gr.Markdown()
    
    analyze_btn.click(
        fn=predict_anomaly,
        inputs=[vibration, temperature, cycle],
        outputs=[status, score, explanation]
    )
    
    gr.Markdown("""
    ---
    **How it works:** 
    - Model trained ONLY on healthy data (self-supervised)
    - Detects anomalies when patterns deviate from normal
    - Provides confidence levels and maintenance instructions
    - Powered by GroqCloud for AI explanations
    """)

demo.launch()