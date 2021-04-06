import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import sexmachine.detector as gender

app = Flask(__name__)
random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
naive_bayes = pickle.load(open('naive_bayes.pkl', 'rb'))
#neural_network = pickle.load(open('neural_network.pkl', 'rb'))

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
    lang_dict = {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2}
    df=pd.DataFrame({'bio':int_features[0],
                     'statuses_count':int_features[1],
                     'followers_count':int_features[5],
                     'friends_count':int_features[2],
                     'favourites_count':int_features[6],
                     'listed_count':int_features[8],
                     'username':int_features[7],
                     'lang':lang_dict[int_features[3]]}, index=[0])
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= df['username'].str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    print type(df['lang'])
    params = [df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,df['lang'].astype(int)]
    
    #Random forest prediction
    rfr_prediction = random_forest.predict([df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,df['lang']])
    
    #support vector machine prediction
    svm_prediction = support_vector.predict([df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,df['lang']])
    
    #naive_bayes prediction
    nvb_prediction = naive_bayes.predict([df['statuses_count'],df['followers_count'],df['friends_count'],df['favourites_count'],df['listed_count'],sex_code,df['lang']])
    
    
    if rfr_prediction[0]==0 and svm_prediction[0]==0 and fnn_prediction[0]==0 :
        percent = 100
    elif (rfr_prediction[0]==0 and svm_prediction[0]==0 and fnn_prediction[0]==1) or (rfr_prediction[0]==0 and svm_prediction[0]==1 and fnn_prediction[0]==0) or (rfr_prediction[0]==1 and svm_prediction[0]==0 and fnn_prediction[0]==0) :
        percent = 67
    elif (rfr_prediction[0]==1 and svm_prediction[0]==1 and fnn_prediction[0]==0) or (rfr_prediction[0]==0 and svm_prediction[0]==1 and fnn_prediction[0]==1) or (rfr_prediction[0]==1 and svm_prediction[0]==0 and fnn_prediction[0]==1) :
        percent = 33
    else :
        percent = 0
    
    return render_template('result.html', rfr_prediction = rfr_prediction[0],svm_prediction = svm_prediction[0],fnn_prediction = fnn_prediction[0],percentage=percent,features=params)

    #return render_template('index.html',features=params)


if __name__ == "__main__":
    app.run(debug=True)