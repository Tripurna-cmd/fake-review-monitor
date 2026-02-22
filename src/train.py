import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

def train():
    # Load processed data
    print("⏳ Loading processed dataset...")
    df = pd.read_csv('data/processed/cleaned_reviews.csv')
    df.dropna(subset=['cleaned_text', 'label_encoded'], inplace=True)

    X = df['cleaned_text']
    y = df['label_encoded']

    print(f"✅ Dataset loaded: {X.shape[0]} reviews")
    print(f"Fake reviews: {y.sum()} | Real reviews: {(y==0).sum()}")

    # Split into train and test
    print("\n⏳ Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # TF-IDF Vectorization
    print("\n⏳ Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print("✅ Vectorization complete!")

    # Define models to compare
    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000),
        'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model      = None
    best_model_name = ''
    best_accuracy   = 0

    print("\n" + "="*50)
    print("       TRAINING AND COMPARING MODELS")
    print("="*50)

    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train_vec, y_train)
        preds    = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, preds)

        print(f"✅ {name} Accuracy: {accuracy*100:.2f}%")
        print(classification_report(y_test, preds,
              target_names=['Real (OR)', 'Fake (CG)']))

        if accuracy > best_accuracy:
            best_accuracy   = accuracy
            best_model      = model
            best_model_name = name

    print("="*50)
    print(f"\n🏆 Best Model : {best_model_name}")
    print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")

    # Save best model and vectorizer
    print("\n⏳ Saving best model and vectorizer...")
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer,  'models/vectorizer.pkl')
    print("✅ Model saved to models/best_model.pkl")
    print("✅ Vectorizer saved to models/vectorizer.pkl")

    # Plot and save Confusion Matrix
    print("\n⏳ Generating Confusion Matrix...")
    preds_best = best_model.predict(X_test_vec)
    cm   = confusion_matrix(y_test, preds_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Real', 'Fake'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.savefig('models/confusion_matrix.png')
    print("✅ Confusion matrix saved to models/confusion_matrix.png")
    print("\n🎉 Training Complete!")

if __name__ == '__main__':
    train()