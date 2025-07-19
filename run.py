from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
import numpy as np
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = "./minilm-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

st.title("🎯 Review Sentiment Predictor")

# Input area
review_input = st.text_area("Enter your review below:", height=200)

# Predict function with probability
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        positive_prob = probs[0][1].item()
        predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment, positive_prob * 100  # return percentage

# Submit button
if st.button("Submit"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Processing... Please wait..."):
            prediction, score = predict_sentiment(review_input)
        st.success("✅ Done!")
        st.markdown(f"### 📊 Prediction: **{prediction}**")
        st.markdown(f"### ⭐ Review Score: **{score:.2f}% Positive**")
