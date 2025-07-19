from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
import numpy as np

# Load model and tokenizer
model_path = "./minilm-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

st.title("🎯 Review Sentiment Predictor")

# Input area
review_input = st.text_area("Enter your review below:", height=200)

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative"

# Submit button
if st.button("Submit"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Processing... Please wait..."):
            prediction = predict_sentiment(review_input)
        st.success("✅ Done!")
        st.markdown(f"### 📊 Prediction: **{prediction}**")
