from flask import Flask, request, render_template
import pickle
import pandas as pd
#import os
#import numpy as np
#import re
#import nltk
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from feature import preprocessor
#import pickle


model=pickle.load(open('model.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify',methods=['POST'])
def classify():
    
    data=pd.read_csv('C:/Users/shaun/OneDrive/Desktop/My_Files/Codes/Self Project 2/Application/corona_fake.csv')

    data["label"]= data["label"].str.replace("fake", "FAKE", case = False) 
    data["label"]= data["label"].str.replace("Fake", "FAKE", case = False) 

    data.loc[5]['label'] = 'FAKE'
    data.loc[15]['label'] = 'TRUE'
    data.loc[43]['label'] = 'FAKE'
    data.loc[131]['label'] = 'TRUE'
    data.loc[242]['label'] = 'FAKE'

    data_trial=data
    data_trial=data_trial.fillna(' ')
    data_trial['total']=data_trial['text']+' '+data_trial['title']

    data_trial['total'] = data_trial['total'].str.replace('[^\w\s]','')
    data_trial['total'] = data_trial['total'].str.lower()

    y=data_trial.label
    data_trial.drop("label", axis=1,inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data_trial['total'], y, test_size=0.2,random_state=102)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.80) 
    #pickle.dump(tfidf_vectorizer,open ('tfidf.pkl','wb')) 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    #tfidf_test = tfidf_vectorizer.transform(X_test)

    support_vector_machine=svm.SVC(kernel='linear').fit(tfidf_train, y_train)

    #pickle.dump(support_vector_machine, open('model.pkl','wb'))
    #model=pickle.load(open('model.pkl','rb'))

    
    if request.method == 'POST':    
        query_title=request.form['news_title']
        query_content=request.form['news_content']
        total=query_title+query_content
        clean_text=preprocessor(total)
        data=[clean_text]
        vect=tfidf_vectorizer.transform(data).toarray()
        pred=support_vector_machine.predict(vect)
    
    return render_template('index.html', prediction_text='The news is : {}'.format(pred[0]))
 
    
    
if __name__=="__main__":
    app.run(debug=True)
    