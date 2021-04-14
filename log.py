from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import sexmachine.detector as gender

random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
naive_bayes = pickle.load(open('naive_bayes.pkl', 'rb'))


users = pd.read_csv(r'/home/vamsi82674/Desktop/fake profile detection fb/app/data/processed_data.csv')
uniq=len(users['location'].unique())
location_list = list(enumerate(np.unique(users['location'])))
location_dict = { name : i for i, name in location_list }  
location_dict['other']=1679

#Random forest prediction
rfr_prediction = random_forest.predict(np.reshape(np.ravel([1,2,3,5,6,6,8,9]), (1, -1)))
print rfr_prediction

#<class 'pandas.core.series.Series'>
# Saving model to disk
pickle.dump(location_dict, open('location_dict.pkl','wb'))