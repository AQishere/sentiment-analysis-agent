import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- PAGE CONFIG ---
st.set_page_config(page_title="Twitter Sentiment AI", page_icon="🐦")

# --- LOAD ASSETS ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every click
def load_models():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()

# Setup NLTK (needed for cleaning)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- CLEANING FUNCTION ---
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# --- USER INTERFACE ---
st.title("🐦 Twitter Sentiment Analyzer")
st.markdown("Type a tweet below to see if the AI thinks it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your tweet here:", placeholder="e.g., I'm having a great day!")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        # 1. Preprocess
        cleaned_input = clean_text(user_input)
        
        # 2. Vectorize
        vectorized_input = tfidf.transform([cleaned_input])
        
        # 3. Predict
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input).max()
        
        # 4. Display Result
        st.divider()
        if prediction == 1:
            st.success(f"### Sentiment: POSITIVE (Confidence: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"### Sentiment: NEGATIVE (Confidence: {probability:.2%})")
    else:
        st.warning("Please enter some text first!")

# --- SIDEBAR INFO ---
st.sidebar.title("About the Model")
st.sidebar.info("""
This AI was trained on **1.6 Million tweets** using a **Logistic Regression** model and **TF-IDF Vectorization**.
""")