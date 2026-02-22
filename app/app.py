import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_review
from database import init_db, save_review, get_all_reviews, get_stats

# Initialize database
init_db()

# Page config
st.set_page_config(
    page_title="Fake Review Monitor",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fake Review Monitor")
    st.markdown("---")

    page = st.radio("📌 Navigate", [
        "🏠 Home — Analyze Review",
        "📊 Model Evaluation",
        "📈 Metrics Analysis",
        "🗄️ Review History"
    ])

    st.markdown("---")
    st.markdown("### 🏆 Model Info")
    st.success("Logistic Regression")
    st.metric("Accuracy",     "89.77%")
    st.metric("Dataset Size", "40,432")
    st.markdown("---")

    stats = get_stats()
    st.markdown("### 📈 Live Stats")
    st.metric("Total Analyzed", stats['total'])
    st.metric("Fake Detected",  stats['fake'])
    st.metric("Real Detected",  stats['real'])

# ─── PAGE 1: HOME ──────────────────────────────────────
if page == "🏠 Home — Analyze Review":
    st.title("🔍 Fake Product Review Monitoring System")
    st.markdown("#### Powered by Machine Learning | Logistic Regression + TF-IDF")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter a Product Review")
        review = st.text_area(
            label="",
            height=220,
            placeholder="Paste any product review here to check if it is real or fake..."
        )

        if st.button("🚀 Analyze Review"):
            if review.strip():
                with st.spinner("🔍 Analyzing review with ML model..."):
                    result = predict_review(review)
                    save_review(
                        review,
                        result['prediction'],
                        result['confidence'],
                        result['fake_prob'],
                        result['real_prob']
                    )

                st.markdown("---")
                st.markdown("### 📊 Analysis Result")

                if result['prediction'] == 'FAKE':
                    st.error("⚠️ This review is most likely **FAKE**")
                else:
                    st.success("✅ This review appears to be **REAL**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction",      result['prediction'])
                c2.metric("Confidence",      f"{result['confidence']}%")
                c3.metric("Fake Probability",f"{result['fake_prob']}%")

                st.markdown("### 📈 Probability Breakdown")
                st.markdown("**🟢 Real Review Probability**")
                st.progress(result['real_prob'] / 100)
                st.markdown(f"**{result['real_prob']}%**")
                st.markdown("**🔴 Fake Review Probability**")
                st.progress(result['fake_prob'] / 100)
                st.markdown(f"**{result['fake_prob']}%**")
                st.info("✅ Review saved to database successfully!")
            else:
                st.warning("⚠️ Please enter a review before clicking Analyze.")

    with col2:
        st.markdown("### 🧠 How It Works")
        st.markdown("""
        1. 📝 **Paste** a product review
        2. 🧹 **Text is cleaned** and processed
        3. 🔢 **TF-IDF** converts text to numbers
        4. 🤖 **ML Model** predicts Real or Fake
        5. 💾 **Result saved** to database
        """)
        st.markdown("---")
        st.markdown("### 🎯 Labels")
        st.error("CG = Computer Generated = FAKE")
        st.success("OR = Original Review = REAL")

# ─── PAGE 2: MODEL EVALUATION ──────────────────────────
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation Metrics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "89.77%")
    col2.metric("Precision", "90%")
    col3.metric("Recall",    "90%")
    col4.metric("F1 Score",  "90%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Model Comparison Chart")
        models     = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
        accuracies = [89.77, 86.18, 77.94]
        colors     = ['#ff4b4b', '#4b9eff', '#4bff91']

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(models, accuracies, color=colors, edgecolor='white', linewidth=1.2)
        ax.set_ylim(60, 100)
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.set_title('Model Accuracy Comparison', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1e2130')
        fig.patch.set_facecolor('#1e2130')
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{acc}%', ha='center', color='white', fontweight='bold'
            )
        st.pyplot(fig)

    with col2:
        st.markdown("### 🥧 Real vs Fake Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(
            [20216, 20216],
            labels     = ['Real (OR)', 'Fake (CG)'],
            colors     = ['#4bff91', '#ff4b4b'],
            autopct    = '%1.1f%%',
            startangle = 90,
            textprops  = {'color': 'white'}
        )
        fig2.patch.set_facecolor('#1e2130')
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### 🖼️ Confusion Matrix")
    if os.path.exists('models/confusion_matrix.png'):
        img = Image.open('models/confusion_matrix.png')
        st.image(img, caption='Confusion Matrix — Logistic Regression', width=500)
    else:
        st.warning("Confusion matrix not found. Run train.py first.")

    st.markdown("---")
    st.markdown("### 📋 Detailed Classification Report")
    report_data = {
        'Class'    : ['Real (OR)', 'Fake (CG)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.90, 0.90, 0.90, 0.90],
        'Recall'   : [0.90, 0.90, 0.90, 0.90],
        'F1-Score' : [0.90, 0.90, 0.90, 0.90],
        'Support'  : [4044, 4043, 8087, 8087]
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True)

# ─── PAGE 3: REVIEW HISTORY ────────────────────────────
elif page == "🗄️ Review History":
    st.title("🗄️ Review History — Database")
    st.markdown("All analyzed reviews are saved here automatically.")
    st.markdown("---")

    stats = get_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analyzed", stats['total'])
    c2.metric("Fake Detected",  stats['fake'])
    c3.metric("Real Detected",  stats['real'])

    st.markdown("---")
    rows = get_all_reviews()

    if rows:
        df = pd.DataFrame(rows, columns=[
            'ID', 'Review Text', 'Prediction',
            'Confidence %', 'Fake %', 'Real %', 'Analyzed At'
        ])
        df['Review Text'] = df['Review Text'].str[:80] + '...'
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label    = "⬇️ Download History as CSV",
            data     = csv,
            file_name= "review_history.csv",
            mime     = "text/csv"
        )
    else:
        st.info("No reviews analyzed yet. Go to Home and analyze some reviews first!")

# ─── PAGE: METRICS ANALYSIS ────────────────────────────
elif page == "📈 Metrics Analysis":
    st.title("📈 Intermediate Model Metrics Analysis")
    st.markdown("---")

    # Load tuning results
    import joblib
    import os

    st.markdown("### 🔧 Hyperparameter Tuning Results")

    if os.path.exists('models/tuning_results.pkl'):
        tuning = joblib.load('models/tuning_results.pkl')

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Accuracy",
                    f"{tuning['baseline_accuracy']*100:.2f}%")
        col2.metric("Tuned Accuracy",
                    f"{tuning['tuned_accuracy']*100:.2f}%",
                    delta=f"+{(tuning['tuned_accuracy']-tuning['baseline_accuracy'])*100:.2f}%")
        col3.metric("Best C Parameter",
                    tuning['best_params']['C'])

        st.markdown("### ⚙️ Best Parameters Found")
        params_df = pd.DataFrame([tuning['best_params']])
        st.dataframe(params_df, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Baseline vs Tuned Accuracy")
        labels  = ['Baseline LR\n89.77%', 'Tuned LR\n90.52%']
        values  = [89.77, 90.52]
        colors  = ['#4b9eff', '#ff4b4b']

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.4)
        ax.set_ylim(85, 95)
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.set_title('Baseline vs Tuned Model', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1e2130')
        fig.patch.set_facecolor('#1e2130')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{val}%', ha='center',
                    color='white', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.markdown("### 📊 All Models Comparison")
        if os.path.exists('models/model_comparison.png'):
            from PIL import Image
            img = Image.open('models/model_comparison.png')
            st.image(img, use_column_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Precision vs Recall vs F1")
        metrics_data = {
            'Model'    : ['Logistic Reg', 'Random Forest', 'Gradient Boost', 'Tuned LR'],
            'Precision': [0.90, 0.86, 0.78, 0.91],
            'Recall'   : [0.90, 0.86, 0.78, 0.91],
            'F1-Score' : [0.90, 0.86, 0.78, 0.91],
        }
        metrics_df = pd.DataFrame(metrics_data)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        x     = range(len(metrics_df['Model']))
        width = 0.25
        ax3.bar([i - width for i in x], metrics_df['Precision'],
                width, label='Precision', color='#4b9eff')
        ax3.bar([i for i in x], metrics_df['Recall'],
                width, label='Recall',    color='#4bff91')
        ax3.bar([i + width for i in x], metrics_df['F1-Score'],
                width, label='F1-Score',  color='#ff4b4b')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_df['Model'], rotation=15,
                            color='white', fontsize=8)
        ax3.set_ylim(0.6, 1.0)
        ax3.set_ylabel('Score', color='white')
        ax3.set_title('Precision vs Recall vs F1', color='white')
        ax3.tick_params(colors='white')
        ax3.legend(facecolor='#1e2130', labelcolor='white')
        ax3.set_facecolor('#1e2130')
        fig3.patch.set_facecolor('#1e2130')
        st.pyplot(fig3)

    with col2:
        st.markdown("### 🖼️ Confusion Matrix")
        if os.path.exists('models/confusion_matrix.png'):
            from PIL import Image
            img2 = Image.open('models/confusion_matrix.png')
            st.image(img2, use_column_width=True)

    st.markdown("---")
    st.markdown("### 📋 Full Metrics Table")
    full_metrics = {
        'Model'     : ['Logistic Regression', 'Random Forest',
                       'Gradient Boosting',   'Tuned LR (Final)'],
        'Accuracy'  : ['89.77%', '86.18%', '77.94%', '90.52%'],
        'Precision' : ['90%',    '86%',    '78%',    '91%'],
        'Recall'    : ['90%',    '86%',    '78%',    '91%'],
        'F1-Score'  : ['90%',    '86%',    '78%',    '91%'],
        'Status'    : ['Baseline', 'Baseline', 'Baseline', '✅ Best Model']
    }
    st.dataframe(pd.DataFrame(full_metrics), use_container_width=True)