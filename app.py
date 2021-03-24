import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sexmachine.detector as gender
from pybrain.datasets import ClassificationDataSet

app = Flask(__name__)
random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
neural_network = pickle.load(open('neural_network.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2}
    '''
    int_features = request.form.values()
    df=pd.DataFrame({'bio':int_features[0],
                     'statuses_count':int_features[1],
                     'followers_count':int_features[5],
                     'friends_count':int_features[2],
                     'favourites_count':int_features[6],
                     'listed_count':int_features[8],
                     'username':int_features[7],
                     'lang':int_features[3]}, index=[0])
    lang_dict = {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2}
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= df['username'].str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    lang_code = int_features[3]
    
 
    #Random forest prediction
    #rfr_prediction = random_forest.predict([df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,lang_dict[lang_code]])
    rfr_prediction=[0]
    #support vector machine prediction
    svm_prediction = support_vector.predict([df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,lang_dict[lang_code]])
    
    #neural network prediction
    ds2 = ClassificationDataSet( 7, 1,nb_classes=2)
    lst = [df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,lang_dict[lang_code]]
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
    return render_template('result.html', rfr_prediction = rfr_prediction[0],svm_prediction = svm_prediction[0],fnn_prediction = fnn_prediction[0],percentage=percent)
    #return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)