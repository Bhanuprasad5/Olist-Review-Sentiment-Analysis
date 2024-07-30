import pickle
import pandas as pd
from PIL import Image
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# Initialize translator
translator = Translator(raise_exception=True)
sent = pickle.load(open(r"olist_review.pkl", "rb"))

# Streamlit app configuration
st.set_page_config(page_title="Olist e-commerce", page_icon="🏪", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        text-align: center;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-section h2 {
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        color: #007BFF;
        text-align: center;
    }
    .review-section {
        font-family: 'Arial', sans-serif;
        color: #333333;
    }
    .past-reviews h5 {
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #9e9276;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown('<h1 class="title">Olist e-commerce Review Classification</h1>', unsafe_allow_html=True)
st.markdown('<h5 class="subtitle">Analyze customer reviews and sentiments</h5>', unsafe_allow_html=True)

# Display an image
image = Image.open(r"Olist.png")
st.image(image, use_column_width=True)

# Input section
st.markdown('<h5>Enter the Review:</h5>', unsafe_allow_html=True)
text = st.text_input("Enter a review in Portuguese")

if st.button("Submit",):
    if text.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        # Translation and sentiment analysis
        translated_review = translator.translate(text, src='pt', dest='en').text
        score = sent.polarity_scores(translated_review)["compound"]

        # Display translated review
        st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
        st.markdown(f'<h5> English : "{translated_review}"</h5>', unsafe_allow_html=True)

        # Determine sentiment
        if score > 0.1:
            out = "Positive Review"
            st.markdown(f'<h2>🟢 {out}</h2>', unsafe_allow_html=True)
        elif score <= -0.1:
            out = "Negative Review"
            st.markdown(f'<h2>🔴 {out}</h2>', unsafe_allow_html=True)
        else:
            out = "Neutral Review"
            st.markdown(f'<h2>⚪ {out}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Display past reviews
st.markdown('<div class="past-reviews"><h5>Past Reviews  with Classification :</h5></div>', unsafe_allow_html=True)
df = pd.read_csv(r"review.csv")
df = df.drop("Unnamed: 0", axis=1)
st.write(df)
