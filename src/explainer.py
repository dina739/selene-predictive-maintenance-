"""
Explainable NLP Module - Using GroqCloud Free Tier
Tells you what's wrong and what to do in plain English
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

class MaintenanceExplainer:
    def __init__(self, model="mixtral-8x7b-32768"):  # Free Groq model
        # Load API key from .env file
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("⚠️ WARNING: No API key found in .env file!")
            print("Please add your Groq API key to .env file")
            print("Format: OPENAI_API_KEY=gsk_your-groq-key-here")
            self.client = None
        else:
            # GroqCloud configuration (OpenAI-compatible)
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"  # Groq endpoint
            )
            self.model = model
            print(f"✅ GroqCloud client initialized with model: {model}")
            print(f"🔑 Key starts with: {self.api_key[:8]}...")
    
    def explain_anomaly(self, machine_id, cycle, sensors, anomaly_score, threshold, rul):
        """Generate explanation for anomaly detection"""
        if not self.client:
            return self._get_fallback_explanation(anomaly_score, threshold, rul)
        
        # Determine severity
        if anomaly_score > threshold * 2:
            severity = "CRITICAL"
            urgency = "IMMEDIATE ACTION REQUIRED (within 24 hours)"
            emoji = "🚨"
        elif anomaly_score > threshold * 1.5:
            severity = "HIGH"
            urgency = "Schedule maintenance within 3-5 days"
            emoji = "⚠️"
        elif anomaly_score > threshold:
            severity = "MEDIUM"
            urgency = "Monitor closely, plan maintenance within 7-10 days"
            emoji = "🔍"
        else:
            severity = "NORMAL"
            urgency = "No action needed"
            emoji = "✅"
        
        # Create prompt for Groq
        prompt = f"""You are SELENE, an expert maintenance AI assistant for industrial equipment.

MACHINE STATUS:
- Machine ID: {machine_id}
- Operating cycle: {cycle}
- Vibration: {sensors['vibration']:.3f} (normal range: 0.3-0.7)
- Temperature: {sensors['temperature']:.3f}°C (normal range: 70-90°C)
- Anomaly Score: {anomaly_score:.3f} (threshold: {threshold:.3f})
- Severity Level: {severity}
- Urgency: {urgency}
- Remaining Useful Life: {rul:.1f} cycles

Based on this information, please provide a concise maintenance advisory with:
1. A plain-English explanation of what's happening (2-3 sentences)
2. The most likely root cause (bearing wear, imbalance, misalignment, etc.)
3. Step-by-step maintenance instructions (numbered list)
4. Any safety precautions needed
5. Estimated time and parts required

Keep it professional, concise, and actionable. Use bullet points for instructions."""
        
        try:
            # Call Groq API (OpenAI-compatible)
            print(f"🔄 Calling Groq API with model: {self.model}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert maintenance engineer with 20 years of experience in industrial equipment. Provide clear, actionable advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            explanation = response.choices[0].message.content
            
            return f"""
{'='*60}
{emoji} SELENE MAINTENANCE ADVISORY
{'='*60}

{explanation}

{'='*60}
⚙️ Confidence: {self._get_confidence_text(anomaly_score, threshold)}
📊 RUL Prediction: {rul:.1f} cycles
🔄 Model: {self.model}
{'='*60}
"""
            
        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            print(f"Error type: {type(e).__name__}")
            return self._get_fallback_explanation(anomaly_score, threshold, rul)
    
    def _get_confidence_text(self, anomaly_score, threshold):
        """Return confidence level based on anomaly score"""
        ratio = anomaly_score / threshold
        if ratio > 2:
            return "VERY HIGH (99%+)"
        elif ratio > 1.5:
            return "HIGH (95%+)"
        elif ratio > 1:
            return "MEDIUM (90%)"
        else:
            return "LOW (needs monitoring)"
    
    def _get_fallback_explanation(self, anomaly_score, threshold, rul):
        """Fallback when API is not available"""
        if anomaly_score > threshold:
            return f"""
{'='*60}
🔧 SELENE MAINTENANCE ADVISORY (OFFLINE MODE)
{'='*60}

⚠️ ANOMALY DETECTED

The system has detected unusual behavior in the machine.

Anomaly Score: {anomaly_score:.3f} (threshold: {threshold:.3f})
Remaining Useful Life: {rul:.1f} cycles

RECOMMENDED ACTIONS:
• Perform visual inspection of the equipment
• Check for unusual noises or vibrations
• Review recent operating conditions
• Increase monitoring frequency
• Consult maintenance manual for troubleshooting

NOTE: Connect to GroqCloud API for detailed AI-powered explanations.
      Get your free API key at: https://console.groq.com

{'='*60}
"""
        else:
            return f"""
{'='*60}
🔧 SELENE MAINTENANCE ADVISORY
{'='*60}

✅ NORMAL OPERATION

No anomalies detected. Machine is operating within normal parameters.

Current RUL: {rul:.1f} cycles
Next scheduled inspection: {int(rul/2)} cycles

Continue routine monitoring.

{'='*60}
"""

# Test the explainer
if __name__ == "__main__":
    print("="*60)
    print("TESTING GROQ CLOUD INTEGRATION")
    print("="*60)
    
    explainer = MaintenanceExplainer()
    
    # Test with different scenarios
    test_cases = [
        {
            "name": "Normal Operation",
            "machine": 0,
            "cycle": 100,
            "sensors": {"vibration": 0.52, "temperature": 78.3},
            "anomaly_score": 0.25,
            "threshold": 0.5,
            "rul": 100
        },
        {
            "name": "Early Warning",
            "machine": 0,
            "cycle": 160,
            "sensors": {"vibration": 0.68, "temperature": 84.2},
            "anomaly_score": 0.55,
            "threshold": 0.5,
            "rul": 40
        },
        {
            "name": "Critical Failure",
            "machine": 0,
            "cycle": 190,
            "sensors": {"vibration": 1.02, "temperature": 98.7},
            "anomaly_score": 1.2,
            "threshold": 0.5,
            "rul": 10
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test['name']}")
        print(f"{'='*60}")
        
        result = explainer.explain_anomaly(
            machine_id=test['machine'],
            cycle=test['cycle'],
            sensors=test['sensors'],
            anomaly_score=test['anomaly_score'],
            threshold=test['threshold'],
            rul=test['rul']
        )
        
        print(result)
        
        if test != test_cases[-1]:
            input("\nPress Enter to continue to next test case...")