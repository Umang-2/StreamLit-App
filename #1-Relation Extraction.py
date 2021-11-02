import streamlit as st
import pandas as pd
import numpy as np
from pyopenie import OpenIE5
import json
import requests 

st.title('Streamlit Relation Extraction')

extractor = OpenIE5('http://18.208.212.160:8000')

df = pd.DataFrame(columns = ['Entity 1','Relation','Entity 2'])

arg1=[]
arg2=[]
relation=[]
df_lst = []

def get_all_values(x):
    for key,value in x.items():
        # For entity 1
        if key == 'arg1':
            for key1,val1 in value.items():
                if key1 == 'text':
                    arg1.append(val1)
        # For entity 2
        if key == 'arg2s':
            if len(value)>1:
                lst = []
                for values in value:
                    for key1,val1 in values.items():
                        if key1 == 'text':
                            lst.append(val1)
                arg2.append(lst)
            else:
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
        
        else:
            if type(value) is dict:
                get_all_values(value)

def info_text(text_input):
        extractions = extractor.extract(text_input)
        with st.expander('JSON Extraction'):
            st.json(extractions)

        for x in extractions:
            get_all_values(x)

def display_info(arg1,relation,arg2,df):
    for i in range(len(arg1)):
        if type(arg2[i]) is list:
            for j in range(len(arg2[i])):
                df_lst.append([arg1[i],relation[i],arg2[i][j]])
        else:
            df_lst.append([arg1[i],relation[i],arg2[i]])

    for row in df_lst:
        df = df.append(pd.Series(row, index = ['Entity 1','Relation','Entity 2']),ignore_index=True)

    with st.expander('List Extraction'):
        st.text('Entity 1 :')
        st.write(arg1)
        st.text('Relation :')
        st.write(relation)
        st.text('Entity 2 :')
        st.write(arg2)
        st.text('Dataframe List:')
        st.write(df_lst)

    with st.expander('Dataframe'):
        df.drop_duplicates(inplace=True)
        st.dataframe(df)


with st.form(key='my_form'):
    text_input = st.text_input(label='Enter text')
    submit_button = st.form_submit_button(label='Submit')
if submit_button:
    info_text(text_input)
    display_info(arg1,relation,arg2,df)