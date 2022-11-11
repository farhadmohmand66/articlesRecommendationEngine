import streamlit as st
from joblib import load
from sklearn.neighbors import NearestNeighbors
import numpy as np 
import pandas as pd

data = load('savedModels/data.joblib')
vectors = load('savedModels/vectors.joblib')
nnModel = load('savedModels/nnModel.joblib')

st.title("ARTICLES RECOMMENDATION ENGINE")
st.markdown('All these research publications are Conferences and journal papers of IEEE\
    of the year 2022 and early access of 2023, and belong to the area of\
        AI, Data Science, Machine learning, Deep Learning, Data Mining, and NLP. Total records are 1506.')

def recommend(paper):
    id = np.where(vectors.index == paper)[0][0]
    distances, suggestions = (nnModel.kneighbors(vectors.iloc[id,:].values.reshape(1, -1), n_neighbors=7))
    paperList = []
    for i in suggestions:
        paperList.append(data.iloc[i])
    return paperList

articlesList = vectors.index
selectedPaper = st.selectbox(
    'How would you like to be recommended?', articlesList)

if st.button('Recommend Papers'):
    suggest = recommend(selectedPaper)
    for a in suggest:
        st.write(a)

st.caption(' ')
st.caption(' ')
st.caption('Designed by @Farhad Khan')