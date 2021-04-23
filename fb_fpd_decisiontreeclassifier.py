'''
Original file is located at
    https://colab.research.google.com/drive/1cxF7-HV256Qdrynv93UEv2YrzHl_O3VL
'''

# Commented out IPython magic to ensure Python compatibility.
import pickle
import numpy as np
import pandas as pd
import sexmachine.detector as gender
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
"""### **Function for reading dataset from csv files**"""

def read_datasets():
    """ Reads users profile from csv files """
    users = pd.read_csv(r'/home/vamsi82674/Desktop/fake profile detection fb/app/data/processed_data.csv')
    y=1337*[1] + 1481*[0]
    return users,y

"""### **Function for predicting gender**"""

def predict_sex(name):
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code

"""### **Function for Feature extraction**"""

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)
    location_list = list(enumerate(np.unique(x['location'])))   
    location_dict = { name : i for i, name in location_list }             
    x.loc[:,'location_code'] = x['location'].map( lambda x: location_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    feature_columns_to_use = ['created_at','location_code','statuses_count','followers_count','favourites_count','friends_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x


"""# **<font color="green">Random Forest Classifier</font>**"""
    

"""### **Reading Dataset**"""

print("reading datasets.....\n")
x,y=read_datasets()

"""### **Extracting Features**"""

print("extracting featues.....\n")
x=extract_features(x)

"""### **Spliting datasets into train and test dataset**"""

print "spliting datasets into train and test dataset...\n"
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)

"""### **Training Dataset**"""

print "training datasets.......\n"
""" Trains and predicts dataset with a Random Forest classifier """

clf=DecisionTreeClassifier(random_state=0,criterion="entropy")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Saving model to disk
pickle.dump(clf, open('decision_tree.pkl','wb'))