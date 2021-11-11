# pip3 install transformers[sentencepiece]
import torch
import streamlit as st
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

url = 'http://3.237.106.170:8000'

@st.cache(allow_output_mutation=True)
def instantiate_tokenizer():
    return AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-span-sentiment-extraction')

@st.cache(allow_output_mutation=True)
def instantiate_model():
    return AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-span-sentiment-extraction')

tokenizer = instantiate_tokenizer()
model = instantiate_model()

st.title('Streamlit Span Sentiment Extraction using AllenNLP & T5')

df = pd.DataFrame(columns = ['Confidence','Sentiment','Span'])

@st.cache
def get_sentiment_span(text):
    input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True) 
    generated_ids = model.generate(input_ids=input_ids, num_beams=1, max_length=80).squeeze()
    predicted_span = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return predicted_span

@st.cache
def global_sentiment(text_input):
    predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz')
    global_sentiment = predictor.predict(text_input)
    return global_sentiment

@st.cache
def span_sentiment(global_sentiment):
    for key,value in global_sentiment.items():
        if key == 'probs':
            max_probs = max(value[0],value[1])
            if max_probs>0.4 and max_probs<0.6:
                sentiment = 'Neutral'
                context = ''
                return sentiment,context,max_probs
            
        if key == 'label':
            if value == '0':
                sentiment = 'Negative'
                context = 'question: negative context: '
            else:
                sentiment = 'Positive'
                context = 'question: positive context: '
            return sentiment,context,max_probs

@st.cache
def radio_format_func(raw_option):
	if raw_option == 'example_paragraph':
		return 'Select an example paragraph.'
	else:
		return 'Type your own paragraph.'

chosen_mode = st.radio(
	label='Choose mode:',
	options=('example_paragraph', 'own_paragraph'),
	format_func=radio_format_func,
	key='radio_key'
)

example_paragraphs = [
	'My sister has a dog. She loves him.',
	'Deepika too has a dog. The movie star has always been fond of animals.',
	'Sam has a Parker pen. He loves writing with it.',
	'Coronavirus quickly spread worldwide in 2020. The virus mostly affects elderly people. They can easily catch it.',
	"Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party.",
	'Jane told her friend that she was about to go to college.',
	"In 1916, a Polish American employee of Feltman's named Nathan Handwerker was encouraged by Eddie Cantor and Jimmy Durante, both working as waiters/musicians, to go into business in competition with his former employer. Handwerker undercut Feltman's by charging five cents for a hot dog when his former employer was charging ten.",
	'A dog named Teddy ran to his owner Jane. Jane loves her dog very much.',
	'Ana and Tom are siblings. Ana is older but her brother is taller.',
	'Angelica has three kittens. Her cats are very cute.',
	"'I like her', said Adam about Julia.",
	'Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high school, Lakeside, to develop their programming skills on several time-sharing computer systems.',
	'The legal pressures facing Michael Cohen are growing in a wide-ranging investigation of his personal business affairs and his work on behalf of his former client, President Trump. In addition to his work for Mr. Trump, he pursued his own business interests, including ventures in real estate, personal loans and investments in taxi medallions.',
	'We are looking for a region of central Italy bordering the Adriatic Sea. The area is mostly mountainous and includes Mt. Corno, the highest peak of the mountain range. It also includes many sheep and an Italian entrepreneur has an idea about how to make a little money of them.'
]

with st.form(key='form_key'):
    if chosen_mode == 'example_paragraph':
	    text_input = st.selectbox(
			label='Paragraph:',
			options=example_paragraphs,
			key='selectbox_key'
		)
    else:
	    text_input = st.text_area(
			label='Paragraph:',
			key='text_area_key'
		)
    submitted = st.form_submit_button(label='Extract')

if submitted:
    global_sentiment = global_sentiment(text_input)
    with st.expander('JSON Extraction'):
            st.json(global_sentiment)
    sentiment,context,prob = span_sentiment(global_sentiment)
    if sentiment != 'Neutral':
        predicted_span = get_sentiment_span(context+text_input)
        row = {'Confidence':prob, 'Sentiment':sentiment, 'Span':predicted_span}
    else:
        row = {'Confidence':prob, 'Sentiment':sentiment, 'Span':'<N/A>'}
    
    df = df.append(row,ignore_index=True)
    with st.expander('Dataframe'):
        st.dataframe(df)


#The Kia Sonet looks good but actually sucks.
#My mother didn't think the movie was logical.
#Recession hit Veronique Branquinho, she has to quit her company, such a shame!
