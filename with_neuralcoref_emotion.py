# pip3 install scipy
# pip3 install tabulate
# spaCy 2.1.0 doesn't have a wheel for Python 3.8. Use Python 3.7.
# pip3 install -U spacy==2.1.0
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz --no-deps
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.1.0/en_core_web_md-2.1.0.tar.gz --no-deps
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz --no-deps
# If the above command doesn't work, then try the following command:
# pip3 install https://storage.googleapis.com/wheels_python/en_core_web_lg-2.1.0.tar.gz --no-deps
# pip3 install neuralcoref
# If the above command gives an error, then try the following two commands:
# pip3 uninstall neuralcoref
# pip3 install neuralcoref --no-binary neuralcoref
# pip3 install pyopenie
# pip3 install transformers[torch]
# pip3 install sentencepiece
# pip3 install streamlit

import urllib
import requests
import re
import numpy as np
from scipy.special import softmax
import pandas as pd
import tabulate
import spacy
import en_core_web_sm
import en_core_web_md
import en_core_web_lg
import neuralcoref
import time
from pyopenie import OpenIE5
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

api_key = ""
endpoint = "https://api.embedly.com/1/extract"
sentiment_labels = ['negative', 'neutral', 'positive']
emotion_labels = ['anger','joy','optimism','sadness']

@st.cache
def clean_embedly_content(content):
    cleaned_content = re.sub('<[^<]*?/?>', '', content)
    return cleaned_content

@st.cache(allow_output_mutation=True)
def get_spacy_model(size="large"):
    if size == "small":
        # nlp = spacy.load("en_core_web_sm")
        nlp = en_core_web_sm.load()
    elif size == "medium":
        # nlp = spacy.load("en_core_web_md")
        nlp = en_core_web_md.load()
    else:
        # nlp = spacy.load("en_core_web_lg")
        nlp = en_core_web_lg.load()
    return nlp

@st.cache
def get_extractor(url):
    extractor = OpenIE5(url)
    return extractor

@st.cache
def get_extractions(sentence):
    extractions = extractor.extract(sentence)
    return extractions

@st.cache
def json_to_df(json):
    arg1 = []
    rel = []
    arg2 = []
    for i in range(len(json)):
        if len(json[i]["extraction"]["arg2s"]) > 0:
            for j in range(len(json[i]["extraction"]["arg2s"])):
                arg1.append(json[i]["extraction"]["arg1"]["text"])
                rel.append(json[i]["extraction"]["rel"]["text"])            
                arg2.append(json[i]["extraction"]["arg2s"][j]["text"])
        else:
            arg1.append(json[i]["extraction"]["arg1"]["text"])
            rel.append(json[i]["extraction"]["rel"]["text"])        
            arg2.append("")
    df = pd.DataFrame({
        'Entity Phrase': arg1,
        'Relation': rel,
        'Argument 2': arg2
    })
    if len(df) > 0:
        df['Sentiment Phrase'] = df['Relation'] + " " + df['Argument 2']
    else:
        df['Sentiment Phrase'] = []
    df.drop(columns=['Relation', 'Argument 2'], inplace=True)
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df

@st.cache
def num_common_words(s0, s1):
    s0_list = s0.split(" ")
    s1_list = s1.split(" ")
    return len(list(set(s0_list) & set(s1_list)))

@st.cache(allow_output_mutation=True)
def get_sentiment_tokenizer_and_model(path="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
    sentiment_tokenizer = AutoTokenizer.from_pretrained(path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(path)
    sentiment_model.save_pretrained(path)
    sentiment_tokenizer.save_pretrained(path)
    return sentiment_tokenizer, sentiment_model

@st.cache
def sentiment_emotion_preprocess(text):
    new_text = [] 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Don't cache:
def get_sentiment_emotion_proba(phrase):
    text = sentiment_emotion_preprocess(phrase)
    
    sentiment_encoded_input = sentiment_tokenizer(text, return_tensors='pt')
    sentiment_output = sentiment_model(**sentiment_encoded_input)
    sentiment_scores = sentiment_output[0][0].detach().numpy()
    sentiment_scores = softmax(sentiment_scores)

    emotion_encoded_input = emotion_tokenizer(text, return_tensors='pt')
    emotion_output = emotion_model(**emotion_encoded_input)
    emotion_scores = emotion_output[0][0].detach().numpy()
    emotion_scores = softmax(emotion_scores)

    return sentiment_scores, emotion_scores

# Don't cache:
def get_sentiments_and_emotions_df(extractions_df, candidate_entities):
    df_copy = extractions_df.copy()
    entities = []
    emotions = []
    sentiments = []
    for i in range(len(df_copy)):
        if len(candidate_entities) > 0:
            num_common_words_array = np.array([num_common_words(df_copy['Entity Phrase'][i], c) for c in candidate_entities])
            if num_common_words_array.max() > 1:
                entity = candidate_entities[np.argmax(num_common_words_array)]
            else:
                entity = df_copy['Entity Phrase'][i]
        else:
            entity = df_copy['Entity Phrase'][i]
        entities.append(entity)
        
        sentiment_scores, emotion_scores = get_sentiment_emotion_proba(df_copy['Sentiment Phrase'][i])

        predicted_sentiment = sentiment_labels[np.argmax(sentiment_scores)]
        sentiments.append(predicted_sentiment)

        predicted_emotion = emotion_labels[np.argmax(emotion_scores)]
        emotions.append(predicted_emotion)

    df_copy['Entity'] = entities
    df_copy['Sentiment'] = sentiments
    df_copy['Emotions'] = emotions
    df_copy = df_copy[['Entity', 'Entity Phrase', 'Sentiment Phrase', 'Sentiment','Emotions']]
    return df_copy

@st.cache
def get_highlighted_df(sentiments_df):
    df_copy = sentiments_df.copy()
    for i in range(len(df_copy)):
        if df_copy['Sentiment'][i] == 'negative':
            df_copy['Sentiment Phrase'][i] = "<span style='color: #E94547;'>" + df_copy['Sentiment Phrase'][i] + "</span>"
        elif df_copy['Sentiment'][i] == 'positive':
            df_copy['Sentiment Phrase'][i] = "<span style='color: #17B169;'>" + df_copy['Sentiment Phrase'][i] + "</span>"
    return df_copy

@st.cache(allow_output_mutation=True)
def get_emotion_tokenizer_and_model(path="cardiffnlp/twitter-roberta-base-emotion"):
    emotion_tokenizer = AutoTokenizer.from_pretrained(path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(path)
    emotion_model.save_pretrained(path)
    emotion_tokenizer.save_pretrained(path)
    return emotion_tokenizer, emotion_model

st.title("Extract Entities & Associated Sentiment Phrases From Articles")

url = st.text_input(label="URL:", value="")
if url != "": # Form validation to be added...
    # escaped_url = urllib.parse.quote(url)
    escaped_url = url
    http_request_params = {'key': api_key, 'url': escaped_url}
    response = requests.get(url=endpoint, params=http_request_params)
    if response.status_code == 200:
        json = response.json()
        cleaned_content = clean_embedly_content(json['content'])

        with st.expander(label="Cleaned Content"):
            st.write(cleaned_content)

        nlp = get_spacy_model("large")
        try:
            neuralcoref.add_to_pipe(nlp, greedyness=0.42, max_dist=50, max_dist_match=500, blacklist=True)
        except ValueError:
            nlp.remove_pipe("neuralcoref")
            neuralcoref.add_to_pipe(nlp, greedyness=0.42, max_dist=50, max_dist_match=500, blacklist=True)
        content_doc = nlp(cleaned_content)

        with st.expander(label="Coreference Resolution"):
            if content_doc._.has_coref:
                st.markdown("**Any coreferences found?** <span style='color: #17B169;'>YES</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Any coreferences found?** <span style='color: #E94547;'>NO</span>", unsafe_allow_html=True)

            st.markdown("**Resolution:**")
            if content_doc._.has_coref:
                highlighted_original_text = ""
                for token in content_doc:
                    if token._.in_coref:
                        title_text = ""
                        for i in range(len(token._.coref_clusters)):
                            if i == 0:
                                title_text += token._.coref_clusters[i].main.text
                            else:
                                title_text += ", " + token._.coref_clusters[i].main.text
                        highlighted_original_text += "<span title='" + title_text + "' style='background: yellow; padding: 4px;'>" + token.text + "</span>" + token.whitespace_
                    else:
                        highlighted_original_text += token.text + token.whitespace_
                st.markdown("<div style='line-height: 32px;'>" + highlighted_original_text + "</div><br>", unsafe_allow_html=True)
                st.caption("Note: Hover on each highlighted word to see the most respresentative mentions in all the coreference clusters that contain the word.")
            else:
                st.markdown("<div style='line-height: 32px;'>" + content_doc.text + "</div><br>", unsafe_allow_html=True)

        resolved_doc = nlp(content_doc._.coref_resolved)
        sentences = []
        if resolved_doc.is_sentenced:
            for sent in resolved_doc.sents:
                sentences.append(sent.text)

        with st.expander(label="Sentence Boundary Detection"):
            if len(sentences) > 0:
                st.markdown("**Any sentence boundaries found?** <span style='color: #17B169;'>YES</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Any sentence boundaries found?** <span style='color: #E94547;'>NO</span>", unsafe_allow_html=True)

            if len(sentences) > 0:
                st.markdown("**Sentences:**")
                sentences_df = pd.DataFrame({'Sentence Number': range(len(sentences)), 'Sentence': sentences})	
                st.table(sentences_df)

        with st.expander(label="All Extractions"):
            if len(sentences) > 0:
                extractor = get_extractor('http://3.237.106.170:8000')
                emotion_tokenizer, emotion_model =  get_emotion_tokenizer_and_model()
                sentiment_tokenizer, sentiment_model = get_sentiment_tokenizer_and_model()
                full_df_list = []
                sentence_numbers = []
                sentence_number = 0
                for sentence in sentences:
                    time.sleep(3)
                    num_words = len(sentence.split(" "))
                    if num_words > 1:
                        try:
                            extractions_json = get_extractions(sentence)
                        except requests.exceptions.ConnectionError:
                            sentence_number += 1
                            continue
                        else:
                            extractions_df = json_to_df(extractions_json)
                            if len(extractions_df) > 0:
                                candidate_entities = []
                                sentence_doc = nlp(sentence)
                                if sentence_doc.is_parsed:
                                    for chunk in sentence_doc.noun_chunks:
                                        candidate_entities.append(chunk.text)
                                sentiments_emotions_df = get_sentiments_and_emotions_df(extractions_df, candidate_entities)
                                highlighted_df = get_highlighted_df(sentiments_emotions_df)
                                full_df_list.append(highlighted_df)
                                sentence_numbers += [sentence_number] * len(highlighted_df)
                            sentence_number += 1
                    else:
                        sentence_number += 1
                        continue
                if len(full_df_list) > 0:
                    full_df = pd.concat(full_df_list, axis=0)
                    assert len(full_df) == len(sentence_numbers), "Lengths of `full_df` and `sentence_numbers` do not match!"
                    full_df['Sentence Number'] = sentence_numbers
                    full_df = full_df[['Sentence Number', 'Entity', 'Entity Phrase', 'Sentiment Phrase', 'Sentiment', 'Emotions']]
                    full_df.reset_index(drop=True, inplace=True)
                    st.markdown(full_df.to_markdown(), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.write("No extractions found!")
            else:
                st.write("No sentences found!")

        with st.expander(label="Extractions Containing Sentiment"):
            if full_df is not None:
                final_df = full_df.loc[full_df['Sentiment'] != 'neutral']
                final_df.reset_index(drop=True, inplace=True)
                st.markdown(final_df.to_markdown(), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.write("No extractions found!")
    elif response.status_code == 400:
        st.write("Bad request!")
    else:
        st.write("Error!")
