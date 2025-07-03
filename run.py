from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import joblib
import streamlit as st
import time

def encode_long_reviews(reviews, model_name='all-mpnet-base-v2', sentences_per_chunk=5):
    model = SentenceTransformer(model_name)
    final_embeddings = []

    for review in tqdm(reviews, desc="Encoding reviews"):
        sentences = sent_tokenize(review)
        if not sentences:
            final_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue
        chunks = [" ".join(sentences[i:i + sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
        chunk_embeddings = model.encode(chunks)
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        final_embeddings.append(avg_embedding)

    return np.array(final_embeddings)

model=joblib.load("model_lgb.pkl")

st.title("🎯 Review Sentiment Predictor")

# Input area
review_input = st.text_area("Enter your review below:", height=200)

# Submit button
if st.button("Submit"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Processing... Please wait..."):
            x=encode_long_reviews([review_input])
            prediction = model.predict(x)
            if prediction==0:
                prediction="Negative"
            else:
                prediction="Positive"
        st.success("✅ Done!")
        st.markdown(f"### 📊 Prediction: {prediction}") 

