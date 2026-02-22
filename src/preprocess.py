import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def preprocess():
    # Load dataset
    df = pd.read_csv('data/raw/reviews.csv')
    print("✅ Dataset loaded successfully!")
    print(f"Total reviews: {df.shape[0]}")
    print(f"Columns: {df.columns.tolist()}")

    # Keep only needed columns
    df = df[['text_', 'label', 'rating', 'category']].copy()

    # Rename text_ to text
    df.rename(columns={'text_': 'text'}, inplace=True)

    # Drop empty rows
    df.dropna(subset=['text', 'label'], inplace=True)

    print(f"\nLabel values found: {df['label'].unique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Clean the review text
    print("\n⏳ Cleaning text... please wait...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Encode label: CG = 1 (Fake), OR = 0 (Real)
    df['label_encoded'] = df['label'].apply(
        lambda x: 1 if str(x).strip().upper() == 'CG' else 0
    )

    # Save processed file
    df.to_csv('data/processed/cleaned_reviews.csv', index=False)

    print("\n✅ Preprocessing complete!")
    print(f"Processed file saved to: data/processed/cleaned_reviews.csv")
    print(f"\nSample output:")
    print(df[['text', 'label', 'label_encoded', 'cleaned_text']].head(3))
    print(f"\nFake reviews (CG): {df[df['label_encoded']==1].shape[0]}")
    print(f"Real reviews (OR): {df[df['label_encoded']==0].shape[0]}")

if __name__ == '__main__':
    preprocess()