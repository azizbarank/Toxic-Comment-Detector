# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:56:08 2022

@author: Aziz Baran Kurtuluş
"""
import os
os.system('pip install nltk')
os.system('pip install sklearn')

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer



site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
contact = st.container()

with site_header:
    st.title('Toxic Comment Detection')
   

with tweet_input:
    st.header('Is Your Text Considered Toxic?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    user_text = st.text_input('Enter Text', max_chars=280)

with model_results:    
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating tfidf vectorizor
        tfidf = TfidfVectorizer(stop_words= stop_words, ngram_range=(1,2))
        X_train = joblib.load(open('resources/X_train.pickel', 'rb'))
        X_test = lemmatized_output
        X_train_count = tfidf.fit_transform(X_train)
        X_test_count = tfidf.transform(X_test)

        # loading in model
        final_model = joblib.load(open('resources/final_bayes.pickel', 'rb'))

        # applying the model to make predictions
        prediction = final_model.predict(X_test_count[0])

        if prediction == 0:
            st.subheader('**Not Toxic**')
        else:
            st.subheader('**Toxic**')
        st.text('')
