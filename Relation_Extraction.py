import streamlit as st
import pandas as pd
import numpy as np
from pyopenie import OpenIE5
import json
import requests 

st.title('Streamlit Relation Extraction')

extractor = OpenIE5('http://18.208.212.160:8000')

df = pd.DataFrame(columns = ['Entity 1','Relation','Entity 2','Confidence','Context'])

confidence = []
arg1=[]
arg2=[]
relation=[]
df_lst = []
context = []
negated = []

def get_all_values(x):
    global confidence
    for key,value in x.items():
        # For confidence
        if key == 'confidence':
            confidence.append(value)
        # For entity 1
        if key == 'arg1':
            for key1,val1 in value.items():
                if key1 == 'text':
                    arg1.append(val1)
        # For entity 2
        if key == 'arg2s':
            if len(value) == 0:
                arg2.append("NULL")
            elif len(value)>1:
                lst = []
                for values in value:
                    for key1,val1 in values.items():
                        if key1 == 'text':
                            lst.append(val1)
                arg2.append(lst)
            else: #len(value) == 1
                lst = []
                for values in value:
                    for key1,val1 in values.items():
                        if key1 == 'text':
                            arg2.append(val1)
        # For relation
        if key == 'rel':
            for key1,val1 in value.items():
                if key1 == 'text':
                    relation.append(val1)
        # For context
        if key == 'context':
            if type(value) is dict:
                for key1,val1 in value.items():
                    if key1 == 'text':
                        context.append(val1)
            else:
                context.append('NULL')
        # For negation
        if key == 'negated':
            if value == 1:
                negated.append('TRUE')
            else:
                negated.append('FALSE')
        else:
            if type(value) is dict:
                get_all_values(value)

def info_text(text_input):
        extractions = extractor.extract(text_input)
        with st.expander('JSON Extraction'):
            st.json(extractions)

        for x in extractions:        
            get_all_values(x)

def display_info(arg1,relation,arg2,confidence,context,negation,df):
    for i in range(len(arg1)):
        if type(arg2[i]) is list:
            for j in range(len(arg2[i])):
                df_lst.append([arg1[i],relation[i],arg2[i][j],confidence[i],context[i],negated[i]])
        else:
            df_lst.append([arg1[i],relation[i],arg2[i],confidence[i],context[i],negated[i]])

    for row in df_lst:
        df = df.append(pd.Series(row, index = ['Entity 1','Relation','Entity 2','Confidence','Context','Negation']),ignore_index=True)

    with st.expander('List Extraction'):
        st.text('Entity 1 :')
        st.write(arg1)
        st.text('Relation :')
        st.write(relation)
        st.text('Entity 2 :')
        st.write(arg2)
        st.text('Confidence :')
        st.write(confidence)
        st.text('Context :')
        st.write(context)
        st.text('Negation :')
        st.write(negated)

    with st.expander('Dataframe'):
        df.drop_duplicates(inplace=True)
        st.dataframe(df)


with st.form(key='my_form'):
    text_input = st.text_input(label='Enter text')
    submit_button = st.form_submit_button(label='Submit')
if submit_button:
    info_text(text_input)
    display_info(arg1,relation,arg2,confidence,context,negated,df)

#The Kia Sonet looks good but actually sucks.
#My mother didn't think the movie was logical.