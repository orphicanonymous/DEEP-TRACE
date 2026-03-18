from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import DeepTraceInference
import tempfile
import os
import requests
from pathlib import Path

app = Flask(__name__)
CORS(app)  # allows React frontend to call this API

# ── Load all 3 models once at startup ────────────────────────────────────────
print('Starting DeepTrace backend...')
engine = DeepTraceInference(models_dir='./models')
print('Backend ready!')


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'DeepTrace backend is running'})


# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    data      = request.get_json()
    file_url  = data.get('file_url')
    file_name = data.get('file_name', 'upload')

    if not file_url:
        return jsonify({'error': 'No file_url provided'}), 400

    # Download file from Supabase storage URL to a temp file
    try:
        response = requests.get(file_url, timeout=120)
        response.raise_for_status()
        suffix   = Path(file_name).suffix or '.mp4'
        tmp      = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()
        tmp_path = tmp.name
    except Exception as e:
        return jsonify({'error': f'File download failed: {str(e)}'}), 500

    # Run inference
    try:
        result = engine.predict(tmp_path)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
