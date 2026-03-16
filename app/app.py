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
    st.success("Multinomial Naive Bayes")
    st.metric("Accuracy",     "88.04%")
    st.metric("Alpha",        "0.1")
    st.metric("Dataset Size", "40,431")
    st.markdown("---")

    stats = get_stats()
    st.markdown("### 📈 Live Stats")
    st.metric("Total Analyzed", stats['total'])
    st.metric("Fake Detected",  stats['fake'])
    st.metric("Real Detected",  stats['real'])

# ─── PAGE 1: HOME ──────────────────────────────────────
if page == "🏠 Home — Analyze Review":
    st.title("🔍 Fake Product Review Monitoring System")
    st.markdown("#### Powered by Machine Learning | Multinomial Naive Bayes + TF-IDF")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Input method selection
        st.markdown("### 📥 Choose Input Method")
        input_method = st.radio(
            label     = "",
            options   = ["✍️ Type or Paste Review Manually",
                         "🔗 Paste Product URL"],
            horizontal= True
        )
        st.markdown("---")

        # ── METHOD 1: MANUAL INPUT ──────────────────────
        if input_method == "✍️ Type or Paste Review Manually":
            st.markdown("### 📝 Enter a Product Review")
            review = st.text_area(
                label       = "",
                height      = 220,
                placeholder = "Paste any product review here to check if it is real or fake..."
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
                    c1.metric("Prediction",       result['prediction'])
                    c2.metric("Confidence",        f"{result['confidence']}%")
                    c3.metric("Fake Probability",  f"{result['fake_prob']}%")

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

        # ── METHOD 2: URL INPUT ─────────────────────────
        elif input_method == "🔗 Paste Product URL":
            st.markdown("### 🔗 Paste Product Page URL")
            st.info("💡 Paste any Amazon or Flipkart product URL to extract and analyze reviews!")

            url = st.text_input(
                label       = "",
                placeholder = "https://www.amazon.in/product-name/dp/XXXXXXXX"
            )

            if st.button("🔍 Extract & Analyze Reviews"):
                if url.strip():
                    with st.spinner("🌐 Fetching reviews from URL... please wait..."):
                        try:
                            import requests
                            from bs4 import BeautifulSoup

                            headers = {
                                'User-Agent'      : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                                'Accept-Language' : 'en-US,en;q=0.9',
                            }

                            response = requests.get(url, headers=headers, timeout=10)

                            if response.status_code == 200:
                                soup          = BeautifulSoup(response.content, 'html.parser')
                                reviews_found = []

                                # Amazon selectors
                                amazon_reviews = soup.find_all('span', {'data-hook': 'review-body'})
                                for r in amazon_reviews:
                                    text = r.get_text().strip()
                                    if len(text) > 20:
                                        reviews_found.append(text)

                                # Flipkart selectors
                                if not reviews_found:
                                    flipkart_reviews = soup.find_all('div', class_='t-ZTKy')
                                    for r in flipkart_reviews:
                                        text = r.get_text().strip()
                                        if len(text) > 20:
                                            reviews_found.append(text)

                                # Generic fallback
                                if not reviews_found:
                                    paragraphs = soup.find_all('p')
                                    for p in paragraphs:
                                        text = p.get_text().strip()
                                        if len(text) > 50:
                                            reviews_found.append(text)

                                if reviews_found:
                                    st.success(f"✅ Found {len(reviews_found)} reviews!")
                                    st.markdown("---")
                                    st.markdown("### 📊 Analysis Results")

                                    fake_count = 0
                                    real_count = 0

                                    for i, review_text in enumerate(reviews_found[:10]):
                                        result = predict_review(review_text)
                                        save_review(
                                            review_text,
                                            result['prediction'],
                                            result['confidence'],
                                            result['fake_prob'],
                                            result['real_prob']
                                        )
                                        if result['prediction'] == 'FAKE':
                                            fake_count += 1
                                            st.error(f"**Review {i+1}:** ⚠️ FAKE ({result['confidence']}% confidence)")
                                        else:
                                            real_count += 1
                                            st.success(f"**Review {i+1}:** ✅ REAL ({result['confidence']}% confidence)")

                                        with st.expander(f"See Review {i+1} Text"):
                                            st.write(review_text[:300] + "..." if len(review_text) > 300 else review_text)

                                    st.markdown("---")
                                    st.markdown("### 📈 Overall Summary")
                                    s1, s2, s3 = st.columns(3)
                                    s1.metric("Total Reviews", len(reviews_found[:10]))
                                    s2.metric("Fake Reviews",  fake_count)
                                    s3.metric("Real Reviews",  real_count)

                                    if fake_count > real_count:
                                        st.error("⚠️ WARNING: This product has MORE FAKE reviews!")
                                    else:
                                        st.success("✅ This product appears to have mostly genuine reviews!")
                                else:
                                    st.warning("⚠️ Could not extract reviews from this URL.")
                                    st.info("💡 Try copying a review manually instead.")
                            else:
                                st.error(f"❌ Could not access URL. Status: {response.status_code}")

                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out.")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Could not connect to URL.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a URL before clicking Analyze.")

    with col2:
        st.markdown("### 🧠 How It Works")
        st.markdown("""
        **✍️ Manual Input:**
        1. 📝 Paste a review
        2. 🧹 Text is cleaned
        3. 🔢 TF-IDF converts to numbers
        4. 🤖 Naive Bayes predicts
        5. 💾 Saved to database

        ---

        **🔗 URL Input:**
        1. 🔗 Paste product URL
        2. 🌐 Reviews extracted
        3. 🤖 Each review analyzed
        4. 📊 Summary shown
        5. 💾 All saved to database
        """)
        st.markdown("---")
        st.markdown("### 🛒 Supported Sites")
        st.markdown("- ✅ Amazon.in")
        st.markdown("- ✅ Amazon.com")
        st.markdown("- ⚠️ Flipkart (partial)")
        st.markdown("---")
        st.markdown("### 🎯 Labels")
        st.error("CG = Fake Review")
        st.success("OR = Real Review")

# ─── PAGE 2: MODEL EVALUATION ──────────────────────────
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation — Tuned Multinomial Naive Bayes")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "88.04%")
    col2.metric("Precision", "88%")
    col3.metric("Recall",    "88%")
    col4.metric("F1 Score",  "88%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Confusion Matrix")
        if os.path.exists('models/confusion_matrix.png'):
            img = Image.open('models/confusion_matrix.png')
            st.image(img,
                     caption='Confusion Matrix — Tuned Multinomial Naive Bayes',
                     use_column_width=True)
        else:
            st.warning("Run train.py first.")

    with col2:
        st.markdown("### 🥧 Real vs Fake Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [20216, 20215],
            labels     = ['Real (OR)', 'Fake (CG)'],
            colors     = ['#4bff91', '#ff4b4b'],
            autopct    = '%1.1f%%',
            startangle = 90,
            textprops  = {'color': 'white'}
        )
        fig.patch.set_facecolor('#1e2130')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 📋 Classification Report — Tuned Naive Bayes")
    report_data = {
        'Class'    : ['Real (OR)', 'Fake (CG)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.87, 0.89, 0.88, 0.88],
        'Recall'   : [0.90, 0.86, 0.88, 0.88],
        'F1-Score' : [0.88, 0.88, 0.88, 0.88],
        'Support'  : [4044, 4043, 8087, 8087]
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True)

# ─── PAGE 3: METRICS ANALYSIS ──────────────────────────
elif page == "📈 Metrics Analysis":
    st.title("📈 Metrics Analysis — Multinomial Naive Bayes")
    st.markdown("---")

    import joblib

    st.markdown("### 🔧 Hyperparameter Tuning Results")
    if os.path.exists('models/tuning_results.pkl'):
        tuning = joblib.load('models/tuning_results.pkl')

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Accuracy",
                    f"{tuning['baseline_accuracy']*100:.2f}%")
        col2.metric("Tuned Accuracy",
                    f"{tuning['tuned_accuracy']*100:.2f}%",
                    delta=f"+{(tuning['tuned_accuracy']-tuning['baseline_accuracy'])*100:.2f}%")
        col3.metric("Best Alpha", tuning['best_params']['alpha'])

        st.markdown("### ⚙️ Best Parameters Found")
        params_df = pd.DataFrame([tuning['best_params']])
        st.dataframe(params_df, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Baseline vs Tuned Accuracy")
        labels = ['Baseline NB', 'Tuned NB']
        values = [87.70, 88.04]
        colors = ['#4b9eff', '#ff4b4b']

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors,
                      edgecolor='white', width=0.4)
        ax.set_ylim(80, 95)
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.set_title('Baseline vs Tuned Naive Bayes', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1e2130')
        fig.patch.set_facecolor('#1e2130')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{val}%', ha='center',
                    color='white', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📉 Precision vs Recall vs F1")
        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Score' : [0.88, 0.88, 0.88]
        }
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars2 = ax2.bar(
            metrics_data['Metric'],
            metrics_data['Score'],
            color=['#4b9eff', '#4bff91', '#ff4b4b'],
            edgecolor='white', width=0.4
        )
        ax2.set_ylim(0.8, 1.0)
        ax2.set_ylabel('Score', color='white')
        ax2.set_title('Tuned NB — Precision / Recall / F1', color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#1e2130')
        fig2.patch.set_facecolor('#1e2130')
        for bar, val in zip(bars2, metrics_data['Score']):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f'{val}', ha='center',
                     color='white', fontweight='bold')
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("### 🖼️ Confusion Matrix")
    if os.path.exists('models/confusion_matrix.png'):
        img = Image.open('models/confusion_matrix.png')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption='Confusion Matrix',
                     use_column_width=True)

    st.markdown("---")
    st.markdown("### 📋 Final Model Metrics Table")
    final_metrics = {
        'Model'     : ['Tuned Multinomial Naive Bayes'],
        'Accuracy'  : ['88.04%'],
        'Precision' : ['88%'],
        'Recall'    : ['88%'],
        'F1-Score'  : ['88%'],
        'Parameters': ['alpha=0.1']
    }
    st.dataframe(pd.DataFrame(final_metrics),
                 use_container_width=True)

# ─── PAGE 4: REVIEW HISTORY ────────────────────────────
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
            label     = "⬇️ Download History as CSV",
            data      = csv,
            file_name = "review_history.csv",
            mime      = "text/csv"
        )
    else:
        st.info("No reviews analyzed yet. Go to Home and analyze some reviews first!")