#!pip install transformers[sentencepiece]
import torch
import streamlit as st
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer

url = 'http://3.237.106.170:8000'

@st.cache
def instantiate_tokenizer(tokenizer = 'mrm8488/t5-base-finetuned-span-sentiment-extraction'):
    return tokenizer

@st.cache
def instantiate_model(model = 'mrm8488/t5-base-finetuned-span-sentiment-extraction'):
    return model

tokenizer = AutoTokenizer.from_pretrained(instantiate_tokenizer())
model = AutoModelWithLMHead.from_pretrained(instantiate_model())

st.title('Streamlit Span Sentiment Extraction using T5')


df = pd.DataFrame(columns = ['Positive Context','Negative Context','Neutral Context'])

def get_sentiment_span(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)  # Batch size 1
    generated_ids = model.generate(input_ids=input_ids, num_beams=1, max_length=80).squeeze()
    predicted_span = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return predicted_span

positve_context = 'question: positive context: '
negative_context = 'question: negative context: '

with st.form(key='my_form'):
    text_input = st.text_input(label='Enter text')
    submit_button = st.form_submit_button(label='Extract')
if submit_button:
    positive = get_sentiment_span(positve_context+text_input)
    negative = get_sentiment_span(negative_context+text_input)
    # if positive == negative: commenting this for the example: "Recession hit Veronique Branquinho, she has to quit her company, such a shame!"
    #     row = {'Neutral Context': positive} # positive or negative both are same
    # else:
    row = {'Positive Context':positive,'Negative Context':negative}
    df = df.append(row,ignore_index=True)
    st.dataframe(df)

#The Kia Sonet looks good but actually sucks.
#My mother didn't think the movie was logical.
#Recession hit Veronique Branquinho, she has to quit her company, such a shame!