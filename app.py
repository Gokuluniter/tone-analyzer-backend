from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
# For production, you should restrict this to your frontend's URL
# Example: CORS(app, resources={r"/*": {"origins": "https://your-frontend-url.vercel.app"}})
CORS(app) 

print("Loading models...")
tone_analyzer = None
email_rewriter = None
ocean_analyzer = None # NEW: Variable for the OCEAN model

# This mapping is for your original tone analyzer
id2label = {
    0: 'Positive',
    1: 'Agitated',
    2: 'Inquisitive',
    3: 'Casual'
}

try:
    ANALYZER_MODEL_ID = "goks24/platinum-tone-analyzer" 
    REWRITER_MODEL_ID = "goks24/Email_rewriter_cum_tone_analyzer"
    # NEW: Add the v2 psychological tone analyzer model ID
    OCEAN_MODEL_ID = "goks24/psychological-tone-analyzer-v2"

    print(f"Loading Tone Analyzer: {ANALYZER_MODEL_ID}...")
    tone_analyzer = pipeline("text-classification", model=ANALYZER_MODEL_ID, top_k=None)
    
    print(f"Loading Email Rewriter: {REWRITER_MODEL_ID}...")
    email_rewriter = pipeline("text2text-generation", model=REWRITER_MODEL_ID)

    # NEW: Load the OCEAN model
    print(f"Loading Psychological Analyzer: {OCEAN_MODEL_ID}...")
    ocean_analyzer = pipeline("text-classification", model=OCEAN_MODEL_ID, top_k=None)
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models from Hub: {e}")

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if both analysis models are loaded
    if not tone_analyzer or not ocean_analyzer:
        return jsonify({"error": "An analysis model is not available"}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        # --- 1. Perform original tone analysis ---
        all_tone_predictions = tone_analyzer(data['text'])[0] 
        translated_predictions = []
        for pred in all_tone_predictions:
            label_id = int(pred['label'].split('_')[-1])
            label_name = id2label.get(label_id, "Unknown")
            translated_predictions.append({'label': label_name, 'score': pred['score']})
        
        dominant_prediction = translated_predictions[0]
        
        # --- 2. NEW: Perform psychological trait (OCEAN) analysis ---
        ocean_results = ocean_analyzer(data['text'])[0]
        # Convert the list of dicts into a single object for easier frontend use
        # e.g., [{'label': 'Openness', 'score': 0.8}, ...] -> {'Openness': 0.8, ...}
        ocean_scores = {pred['label']: pred['score'] for pred in ocean_results}

        # --- 3. Combine results into the final response ---
        response_data = {
            "tone": dominant_prediction['label'],
            "confidence": dominant_prediction['score'],
            "allTones": translated_predictions,
            "oceanTraits": ocean_scores, # NEW: Add the OCEAN scores
            # Placeholder data (can be removed if not used)
            "sentiment": "neutral", 
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error during analysis: {e}") # Log the error on the server
        return jsonify({"error": str(e)}), 500

@app.route('/rewrite', methods=['POST'])
def rewrite():
    if not email_rewriter: return jsonify({"error": "Email rewriter model not available"}), 500
    data = request.get_json()
    if not data or 'text' not in data or 'tone' not in data: return jsonify({"error": "Missing 'text' or 'tone'"}), 400
    try:
        # The model expects a prefix to guide the generation
        prefix = f"rewrite in a {data['tone']} tone: "
        input_text = prefix + data['text']
        result = email_rewriter(input_text, max_length=256, num_beams=5, early_stopping=True)
        return jsonify({"rewrittenText": result[0]['generated_text']})
    except Exception as e:
        print(f"Error during rewrite: {e}") # Log the error on the server
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', port=5000)

