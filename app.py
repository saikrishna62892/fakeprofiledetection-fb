# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:24:20 2021

@author: saikrishna
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import sexmachine.detector as gender
from pybrain.datasets import ClassificationDataSet
from datetime import date
from datetime import datetime
import datetime

app = Flask(__name__)
random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
#naive_bayes = pickle.load(open('naive_bayes.pkl', 'rb'))
neural_network = pickle.load(open('neural_network.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    ['username','location','statuses_count','followers_count','friends_count','favourites_count','sex_code','lang_code','created_at']
    {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2}
    '''
    int_features = request.form.values()
    
    #lang
    lang_dict = {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2, 'hi':8 ,'other': 9}
    
    #location
    users = pd.read_csv(r'/home/vamsi82674/Desktop/fake profile detection fb/app/data/processed_data.csv')
    location_list = list(enumerate(np.unique(users['location'])))   
    location_dict = { name : i for i, name in location_list }
    location_dict['other']=1679
                 
    #created_at
    created_date = datetime.datetime.strptime(datetime.datetime.strptime(int_features[7], '%Y-%m-%d').strftime('%m %d %Y'),'%m %d %Y')
    today =  datetime.datetime.strptime(datetime.datetime.now().strftime('%m %d %Y'),'%m %d %Y') 
    days_count = today - created_date
    days_count = days_count.days
    
                 
    df=pd.DataFrame({'bio':int_features[0],
                     'statuses_count':int_features[1],
                     'followers_count':int_features[5],
                     'friends_count':int_features[2],
                     'favourites_count':int_features[8],
                     'created_at':days_count,
                     'location':location_dict[int_features[6]],
                     'username':int_features[9],
                     'lang':lang_dict[int_features[3]]}, index=[0])

    #predicting sex
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= df['username'].str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)

    #['created_at','location','statuses_count','followers_count','favourites_count','friends_count','sex_code','lang_code']
    params = pd.Series([df['created_at'],df['location'],df['statuses_count'],df['followers_count'],df['favourites_count'],df['friends_count'],sex_code,df['lang']])
    
    #Random forest prediction
    rfr_prediction = random_forest.predict(params)
    
    #support vector machine prediction
    svm_prediction = support_vector.predict(params)
    
    #neural network prediction
    ds2 = ClassificationDataSet( 8, 1,nb_classes=2)
    lst = [df['created_at'],df['location'],df['statuses_count'],df['followers_count'],df['favourites_count'],df['friends_count'],sex_code,df['lang'].astype(int)]
    ds2.addSample(lst,1)
    ds2._convertToOneOfMany( )
    fnn_prediction=neural_network.testOnClassData (dataset=ds2)
    
    if rfr_prediction[0]==0 and svm_prediction[0]==0 and fnn_prediction[0]==0 :
        percent = 100
    elif (rfr_prediction[0]==0 and svm_prediction[0]==0 and fnn_prediction[0]==1) or (rfr_prediction[0]==0 and svm_prediction[0]==1 and fnn_prediction[0]==0) or (rfr_prediction[0]==1 and svm_prediction[0]==0 and fnn_prediction[0]==0) :
        percent = 67
    elif (rfr_prediction[0]==1 and svm_prediction[0]==1 and fnn_prediction[0]==0) or (rfr_prediction[0]==0 and svm_prediction[0]==1 and fnn_prediction[0]==1) or (rfr_prediction[0]==1 and svm_prediction[0]==0 and fnn_prediction[0]==1) :
        percent = 33
    else :
        percent = 0
    
    return render_template('result.html', rfr_prediction = rfr_prediction[0],svm_prediction = svm_prediction[0],fnn_prediction = fnn_prediction[0],percentage=percent,features=params)

    #return render_template('index.html',features=days_count)


if __name__ == "__main__":
    app.run(debug=True)