import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

def train():
    print("⏳ Loading processed dataset...")
    df = pd.read_csv('data/processed/cleaned_reviews.csv')
    df.dropna(subset=['cleaned_text', 'label_encoded'], inplace=True)

    X = df['cleaned_text']
    y = df['label_encoded']

    print(f"✅ Dataset loaded  : {X.shape[0]} reviews")
    print(f"Fake reviews (CG) : {y.sum()}")
    print(f"Real reviews (OR) : {(y==0).sum()}")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # TF-IDF Vectorization
    print("\n⏳ Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print("✅ Vectorization complete!")

    # ── BASELINE NAIVE BAYES ─────────────────────────
    print("\n" + "="*55)
    print("      BASELINE MULTINOMIAL NAIVE BAYES")
    print("="*55)

    baseline_nb    = MultinomialNB()
    baseline_nb.fit(X_train_vec, y_train)
    baseline_preds = baseline_nb.predict(X_test_vec)
    baseline_acc   = accuracy_score(y_test, baseline_preds)

    print(f"✅ Baseline Naive Bayes Accuracy: {baseline_acc*100:.2f}%")
    print(classification_report(y_test, baseline_preds,
          target_names=['Real (OR)', 'Fake (CG)']))

    # ── HYPERPARAMETER TUNING ─────────────────────────
    print("\n" + "="*55)
    print("   HYPERPARAMETER TUNING — Multinomial Naive Bayes")
    print("="*55)
    print("⏳ Tuning with GridSearchCV... please wait...")

    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    }

    grid = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv      = 5,
        scoring = 'accuracy',
        n_jobs  = -1,
        verbose = 1
    )
    grid.fit(X_train_vec, y_train)

    tuned_nb    = grid.best_estimator_
    tuned_preds = tuned_nb.predict(X_test_vec)
    tuned_acc   = accuracy_score(y_test, tuned_preds)

    print(f"\n✅ Best Alpha Parameter : {grid.best_params_['alpha']}")
    print(f"✅ Baseline Accuracy    : {baseline_acc*100:.2f}%")
    print(f"✅ Tuned Accuracy       : {tuned_acc*100:.2f}%")
    print(f"✅ Improvement          : +{(tuned_acc-baseline_acc)*100:.2f}%")

    print(f"\n📊 Tuned Naive Bayes Classification Report:")
    print(classification_report(y_test, tuned_preds,
          target_names=['Real (OR)', 'Fake (CG)']))

    # ── SAVE MODEL & VECTORIZER ───────────────────────
    print("\n⏳ Saving model and vectorizer...")
    joblib.dump(tuned_nb,   'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    tuning_results = {
        'model_name'       : 'Multinomial Naive Bayes',
        'best_params'      : grid.best_params_,
        'baseline_accuracy': baseline_acc,
        'tuned_accuracy'   : tuned_acc,
    }
    joblib.dump(tuning_results, 'models/tuning_results.pkl')

    print("✅ Model saved     : models/best_model.pkl")
    print("✅ Vectorizer saved: models/vectorizer.pkl")
    print("✅ Tuning results  : models/tuning_results.pkl")

    # ── CONFUSION MATRIX ─────────────────────────────
    print("\n⏳ Generating Confusion Matrix...")
    cm   = confusion_matrix(y_test, tuned_preds)
    disp = ConfusionMatrixDisplay(cm,
           display_labels=['Real', 'Fake'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix — Tuned Multinomial Naive Bayes")
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    print("✅ Confusion matrix saved!")

    # ── ACCURACY COMPARISON CHART ─────────────────────
    print("\n⏳ Generating Accuracy Chart...")
    labels = ['Baseline\nNaive Bayes', 'Tuned\nNaive Bayes']
    values = [baseline_acc*100, tuned_acc*100]
    colors = ['#4b9eff', '#ff4b4b']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor='white', width=0.4)
    ax.set_ylim(60, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Baseline vs Tuned Naive Bayes')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center',
                fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/model_comparison.png')
    plt.close()
    print("✅ Accuracy chart saved!")

    print("\n" + "="*55)
    print("🎉 Training Complete!")
    print(f"🏆 Model    : Tuned Multinomial Naive Bayes")
    print(f"🏆 Alpha    : {grid.best_params_['alpha']}")
    print(f"🏆 Accuracy : {tuned_acc*100:.2f}%")
    print("="*55)

if __name__ == '__main__':
    train()