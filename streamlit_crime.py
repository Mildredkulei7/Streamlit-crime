# -*- coding: utf-8 -*-

print('the begining')
import streamlit as st
#import pandas as pd
#import numpy as np
#import sklearn

#Create the horizontal sections of our app
header = st.container()
desc = st.container()
plot = st.container()
prediction = st.container()
model = st.container()

#The first section
with header:
    st.title('Crime Analysis')
    st.write("""
    # Text Classification
    """)
    st.write('This is an application that classifies crime related text into various subcategories')
    col1, col2 = st.columns(2)
    with col1:
        st.image('E:\pythonProject\Appearance - Ruby-throated Hummingbird... (1).jpg')
        st.button('Image 1')
    with col2:
        st.image('E:\pythonProject\Appearance - Ruby-throated Hummingbird... (1).jpg')
        st.button('Image 2')

with model:
    st.title('Classification Algorithms')
    models = st.sidebar.selectbox('Select Classification Model', ('RNN','Transformers', 'Logistic', 'Naive Bayes','KNN'))
    dataset = st.sidebar.selectbox('Select dataset', ("African_tweets_dataset",'African_tweets_dataset' ))
    st.write('Classification model : ',models)
    st.write('Dataset : ',dataset)

#Loading the dataset
    def get_data(dataset_name):
        data = pd.read_csv('E:\pythonProject\Classified data.csv')
        X = data['tweet']
        y = data['class']
        return X,y
    X,y = get_data(dataset)

    st.write('Number of tweets : ', len(X))
    st.write('Dataset classes : ', y.unique().tolist())

#Adding model parameters
    def model_params(clf_name):
        params = dict()
        if models == 'KNN':
            k = st.sidebar.slider('Number of K-folds', 1,10)
            params['K'] = k
        elif models == 'RNN':
            activation = st.sidebar.selectbox('select activation function',('relu','softmax'))
            params['activation function'] = activation
        return params

    model_params(models)

#Second section of the app
with desc:
    st.title('Overview')
    st.write('This is the overview of the app')


#The third section
#with plot:


#The input and prediction section
#with prediction:
