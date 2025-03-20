import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Loading the trained model
with open("fake_news_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Loading the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    TfidfVectorizer = pickle.load(file)

# Ensure the model is valid
if not hasattr(model, "predict"):
    st.error("Loaded model is not valid. Please check your .pkl file.")
    st.stop()

st.title("Fake News Detection System")

news_text = st.text_area("Enter news text for classification")

if st.button("Predict"):
    if news_text:
        # Transform the text using TF-IDF Vectorizer
        transformed_text = TfidfVectorizer.transform([news_text])

        # Predict
        prediction = model.predict(transformed_text)[0]

        # Displaying the result
        if prediction ==1:
            st.error("🚨 This news is FAKE")

        else:
            st.success(" ✅ This news is REAL")

    else:
        st.warning("Please enter some text for prediction")