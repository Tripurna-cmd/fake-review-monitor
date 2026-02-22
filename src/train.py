import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

def train():
    print("⏳ Loading processed dataset...")
    df = pd.read_csv('data/processed/cleaned_reviews.csv')
    df.dropna(subset=['cleaned_text', 'label_encoded'], inplace=True)

    X = df['cleaned_text']
    y = df['label_encoded']

    print(f"✅ Dataset loaded: {X.shape[0]} reviews")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    print("\n⏳ Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print("✅ Vectorization complete!")

    # ── BASELINE MODELS ──────────────────────────────
    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000),
        'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results      = {}
    best_model   = None
    best_name    = ''
    best_acc     = 0

    print("\n" + "="*55)
    print("        BASELINE MODEL TRAINING")
    print("="*55)

    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train_vec, y_train)
        preds    = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, preds)
        results[name] = accuracy
        print(f"✅ {name} Accuracy: {accuracy*100:.2f}%")
        print(classification_report(y_test, preds,
              target_names=['Real (OR)', 'Fake (CG)']))
        if accuracy > best_acc:
            best_acc   = accuracy
            best_model = model
            best_name  = name

    print(f"\n🏆 Best Baseline: {best_name} — {best_acc*100:.2f}%")

    # ── HYPERPARAMETER TUNING ─────────────────────────
    print("\n" + "="*55)
    print("     HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*55)
    print("⏳ Tuning Logistic Regression... please wait...")

    param_grid = {
        'C'       : [0.1, 1, 10],
        'solver'  : ['lbfgs', 'liblinear'],
        'max_iter': [500, 1000]
    }

    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv      = 5,
        scoring = 'accuracy',
        n_jobs  = -1,
        verbose = 1
    )
    grid.fit(X_train_vec, y_train)

    tuned_model    = grid.best_estimator_
    tuned_preds    = tuned_model.predict(X_test_vec)
    tuned_accuracy = accuracy_score(y_test, tuned_preds)

    print(f"\n✅ Best Parameters : {grid.best_params_}")
    print(f"✅ Tuned Accuracy  : {tuned_accuracy*100:.2f}%")
    print(f"✅ Improvement     : +{(tuned_accuracy - best_acc)*100:.2f}%")

    # Save tuning results
    tuning_results = {
        'best_params'       : grid.best_params_,
        'baseline_accuracy' : best_acc,
        'tuned_accuracy'    : tuned_accuracy,
        'all_results'       : results
    }
    joblib.dump(tuning_results, 'models/tuning_results.pkl')
    joblib.dump(tuned_model,    'models/best_model.pkl')
    joblib.dump(vectorizer,     'models/vectorizer.pkl')

    print("\n✅ Model saved to models/best_model.pkl")
    print("✅ Vectorizer saved to models/vectorizer.pkl")
    print("✅ Tuning results saved to models/tuning_results.pkl")

    # ── CONFUSION MATRIX ─────────────────────────────
    cm   = confusion_matrix(y_test, tuned_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix — Tuned Logistic Regression")
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    print("✅ Confusion matrix saved!")

    # ── MODEL COMPARISON CHART ────────────────────────
    all_models = list(results.keys()) + ['Tuned LR']
    all_scores = [v*100 for v in results.values()] + [tuned_accuracy*100]
    colors     = ['#4b9eff', '#4bff91', '#ffaa00', '#ff4b4b']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(all_models, all_scores, color=colors, edgecolor='white')
    ax.set_ylim(60, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison — Baseline vs Tuned')
    for bar, score in zip(bars, all_scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{score:.2f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/model_comparison.png')
    plt.close()
    print("✅ Model comparison chart saved!")

    print("\n🎉 Training + Tuning Complete!")
    print(f"🏆 Final Model Accuracy: {tuned_accuracy*100:.2f}%")

if __name__ == '__main__':
    train()