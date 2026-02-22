import joblib
import re
import os
import nltk

# Download NLTK data silently for cloud deployment
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Fix model path for both local and cloud
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
VEC_PATH   = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Load model and vectorizer
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_review(text: str) -> dict:
    cleaned    = clean_text(text)
    vec        = vectorizer.transform([cleaned])
    pred       = model.predict(vec)[0]
    prob       = model.predict_proba(vec)[0]
    confidence = round(max(prob) * 100, 2)

    return {
        'prediction' : 'FAKE' if pred == 1 else 'REAL',
        'confidence' : confidence,
        'fake_prob'  : round(prob[1] * 100, 2),
        'real_prob'  : round(prob[0] * 100, 2)
    }