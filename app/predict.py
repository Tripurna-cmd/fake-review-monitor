import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer once when app starts
model      = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_review(text: str) -> dict:
    cleaned  = clean_text(text)
    vec      = vectorizer.transform([cleaned])
    pred     = model.predict(vec)[0]
    prob     = model.predict_proba(vec)[0]
    confidence = round(max(prob) * 100, 2)

    return {
        'prediction' : 'FAKE' if pred == 1 else 'REAL',
        'confidence' : confidence,
        'fake_prob'  : round(prob[1] * 100, 2),
        'real_prob'  : round(prob[0] * 100, 2)
    }