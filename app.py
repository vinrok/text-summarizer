import streamlit as st
from transformers import pipeline
@st.cache(allow_output_mutation=True)
def summarize_model():
    model = pipeline("summarization")
    return model
summ = summarize_model()
st.title("Text Summarizer")
st.sub_header("Paste any article in the text area below and get the summary by clicking on 'Summarize Text' button")
st.caption("Text summarization using HuggingFace's transformers pre-trained model")
sentence = st.text_area('Paste your copied data here...', height=100)
button = st.button("Summarize Text")
max_lengthy = st.sidebar.slider('Maximum summary length (words)', min_value=30, max_value=700, value=100, step=10)
num_beamer = st.sidebar.slider('Speed vs quality of Summary (1 is fastest but less accurate)', min_value=1, max_value=8, value=4, step=1)
with st.spinner("Summarizing..."):
    if button and sentence:
        summary = summ(sentence, max_length = max_lengthy, min_length = 50, num_beams=num_beamer, do_sample=True,early_stopping=True, repetition_penalty=1.5, length_penalty=1.5)[0]
        st.write(summary['summary_text'])
