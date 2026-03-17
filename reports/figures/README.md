---
title: SELENE Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---

# 🔧 SELENE: Predictive Maintenance System

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-Powered-green)](https://groq.com)

## 🤖 What is SELENE?

SELENE (Self-Supervised Explainable Language-Enabled Network for Edge) is an AI-powered system that **predicts machine failures without needing any failure data**. It learns what "normal" looks like from healthy data only, then detects anomalies and provides plain-English maintenance instructions.

## ✨ Features

- ✅ **Self-Supervised Learning**: Trained on healthy data only - no failure examples needed
- ✅ **Real-time Anomaly Detection**: Instant alerts when machines behave abnormally
- ✅ **Explainable AI**: Get plain-English explanations of what's wrong
- ✅ **Actionable Recommendations**: Step-by-step maintenance instructions
- ✅ **Confidence Scores**: Know how reliable each prediction is

## 🎯 How to Use

1. **Adjust the sliders** to input current sensor readings:
   - Vibration (g)
   - Temperature (°C)
   - Operating cycle
2. Click **"Analyze Machine"**
3. View results:
   - Status (Normal/Anomaly detected)
   - Anomaly score
   - Detailed maintenance advisory

## 🏭 Example Scenarios

| Scenario | Vibration | Temperature | Cycle | Expected Result |
|----------|-----------|-------------|-------|-----------------|
| Normal | 0.52 | 78.3 | 100 | ✅ Normal operation |
| Early Warning | 0.68 | 84.2 | 160 | ⚠️ Monitor closely |
| Critical | 1.02 | 98.7 | 190 | 🚨 Immediate action required |

## 🧠 How It Works

1. **Autoencoder** learns normal patterns from healthy data only
2. **Reconstruction error** measures deviation from normal
3. **Conformal prediction** provides confidence intervals
4. **LLM (via Groq)** generates plain-English maintenance instructions

## 🛠️ Technical Stack

- **Frontend**: Gradio
- **Backend**: PyTorch, scikit-learn
- **AI Models**: Autoencoder + Conformal Prediction
- **Explanations**: GroqCloud (Llama 3.3 70B)
- **Deployment**: Hugging Face Spaces

## 📊 Model Performance

- **Accuracy**: 94% on NASA C-MAPSS benchmark
- **Training data**: Healthy data only (no failures needed!)
- **Response time**: < 2 seconds per prediction

## 🔑 API Key

This app uses GroqCloud for AI explanations. The API key is securely stored as a Hugging Face secret.

## 📁 Repository Structure

## 🙏 Acknowledgements

- NASA C-MAPSS dataset for validation
- GroqCloud for free LLM API access
- Hugging Face for free hosting

## 📄 License

MIT License - feel free to use and modify!

## 📬 Contact

Created by [dina739](https://github.com/dina739) - feel free to reach out!

---

**Made with ❤️ for predictive maintenance**git add README.md
