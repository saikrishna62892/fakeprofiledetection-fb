'''
Original file is located at
    https://colab.research.google.com/drive/1cxF7-HV256Qdrynv93UEv2YrzHl_O3VL
'''

# Commented out IPython magic to ensure Python compatibility.
import pickle
import numpy as np
import pandas as pd
import sexmachine.detector as gender
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""### **Function for reading dataset from csv files**"""

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("users.csv")
    fake_users = pd.read_csv("fusers.csv")
    x=pd.concat([genuine_users,fake_users])   
    y=len(fake_users)*[0] + len(genuine_users)*[1]
    return x,y

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
    print lang_dict          
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
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
clf=RandomForestClassifier(n_estimators=40,oob_score=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Fake','Genuine'])
# Saving model to disk
pickle.dump(clf, open('random_forest.pkl','wb'))
pickle.dump(report, open('random_forest_report.pkl','wb'))

# Loading model to compare the results
#random_forest = pickle.load(open('random_forest.pkl','rb'))
#print(random_forest.predict([[2, 9, 6,2,9,6,2]]))