import os
import re
import torch
import streamlit as st
import nltk
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration
from summa import summarizer

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")


nltk.download("stopwords")



stop = set(stopwords.words('english'))


# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_extractive_summary(text):
    text = ' '.join([word for word in text.split() if word not in stop])
    extractive_sum = summarizer.summarize(text, ratio=0.5, language='english')
    return extractive_sum

def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("Summarizer Text")
st.markdown("""
<style>
body {
    color: #2E2E2E;
    background-color: #2B3033;
}
.stTextArea {
    background-color: #2B3033;
    color: white;
    border: 1px solid #2E2E2E;
    border-radius: 5px;
}
.stButton>button {
    color: white;
    background-color: #007BFF;
    border: none;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #0056b3;
}
footer {
    text-align: center;
    padding: 10px 0;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

st.header("Enter your Text")

text_input = st.text_area("text", height=200, label_visibility="collapsed")

def copy_to_clipboard(summary):
    js = f"""navigator.clipboard.writeText({summary})"""
    return f'<script>{js}</script>'

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Abstractive Summary"):
        if text_input.strip() == "":
            st.error("Please enter some text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                summary_output = summarize_text(text_input)
                st.subheader("Abstractive Summary")
                st.write(summary_output)
                
with col2:
    if st.button("Generate Extractive Summary"):
        if text_input.strip() == "":
            st.error("Please enter some text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                summary_output = generate_extractive_summary(text_input)
                st.subheader("Extractive Summary")
                st.write(summary_output)
