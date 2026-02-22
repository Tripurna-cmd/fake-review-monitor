from flask import Flask, request, jsonify
import sys
import os
import hashlib
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_review
from database import init_db, save_review, get_all_reviews, get_stats

app = Flask(__name__)

# Initialize database
init_db()

# ── INPUT VALIDATION & SECURITY ──────────────────────
def validate_input(text):
    if not text:
        return False, "Review text cannot be empty"
    if len(text.strip()) < 10:
        return False, "Review too short — minimum 10 characters"
    if len(text) > 5000:
        return False, "Review too long — maximum 5000 characters"
    # Check for script injection
    if re.search(r'<[^>]+>', text):
        return False, "Invalid characters detected in input"
    return True, "Valid"

def mask_review(text):
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL MASKED]', text)
    # Mask phone numbers
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
                  '[PHONE MASKED]', text)
    # Mask credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                  '[CARD MASKED]', text)
    return text

def generate_review_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]

# ── API ENDPOINTS ─────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message'  : 'Fake Review Monitor API',
        'version'  : '1.0',
        'endpoints': {
            'POST /predict'  : 'Predict if a review is fake or real',
            'GET  /stats'    : 'Get database statistics',
            'GET  /reviews'  : 'Get all analyzed reviews',
            'GET  /health'   : 'Check API health status'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'healthy',
        'model'  : 'Logistic Regression',
        'accuracy': '90.52%'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({'error': 'Please provide review text in JSON body'}), 400

        review_text = data['review']

        # Validate input
        is_valid, message = validate_input(review_text)
        if not is_valid:
            return jsonify({'error': message}), 400

        # Mask sensitive data
        masked_text = mask_review(review_text)

        # Get prediction
        result = predict_review(masked_text)

        # Generate unique review ID
        review_id = generate_review_id(review_text)

        # Save to database
        save_review(
            masked_text,
            result['prediction'],
            result['confidence'],
            result['fake_prob'],
            result['real_prob']
        )

        return jsonify({
            'review_id'  : review_id,
            'prediction' : result['prediction'],
            'confidence' : result['confidence'],
            'fake_prob'  : result['fake_prob'],
            'real_prob'  : result['real_prob'],
            'masked_text': masked_text,
            'status'     : 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        data = get_stats()
        return jsonify({
            'total_analyzed' : data['total'],
            'fake_detected'  : data['fake'],
            'real_detected'  : data['real'],
            'status'         : 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reviews', methods=['GET'])
def reviews():
    try:
        rows = get_all_reviews()
        reviews_list = []
        for row in rows:
            reviews_list.append({
                'id'         : row[0],
                'review_text': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
                'prediction' : row[2],
                'confidence' : row[3],
                'fake_prob'  : row[4],
                'real_prob'  : row[5],
                'analyzed_at': row[6]
            })
        return jsonify({
            'total'  : len(reviews_list),
            'reviews': reviews_list,
            'status' : 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake Review Monitor API...")
    print("📡 API running at: http://localhost:5000")
    print("📋 Endpoints:")
    print("   GET  http://localhost:5000/health")
    print("   POST http://localhost:5000/predict")
    print("   GET  http://localhost:5000/stats")
    print("   GET  http://localhost:5000/reviews")
    app.run(debug=True, port=5000)