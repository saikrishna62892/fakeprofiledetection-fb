'''
Original file is located at
    https://colab.research.google.com/drive/1YqpI9ej7Nr8K47rmMqwKycvrV01aouh5
'''

import os
import pickle
import numpy as np
import pandas as pd
import sexmachine.detector as gender
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

"""### **Function for reading dataset from csv files**"""

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("users.csv")
    fake_users = pd.read_csv("fusers.csv")
    print(genuine_users.columns)
    print(genuine_users.describe())
    print(fake_users.describe())
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
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x

"""### **Reading Dataset**"""

#print("reading datasets.....\n")
x,y=read_datasets()
x.describe()

"""### **Extracting Features**"""

print("extracting featues.....\n")
x=extract_features(x)
#print(x.columns)
#print(x.describe())

"""### **Training Dataset**"""

print "training datasets.......\n"

""" Trains and predicts dataset with a Neural Network classifier """

ds = ClassificationDataSet( len(x.columns), 1,nb_classes=2)
for k in xrange(len(x)): 
	ds.addSample(x.iloc[k],np.array(y[k]))
tstdata, trndata = ds.splitWithProportion( 0.20 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
input_size=len(x.columns)
target_size=1
hidden_size = 5   
fnn=None
if  os.path.isfile('fnn.xml'): 
	fnn = NetworkReader.readFrom('fnn.xml') 
else:
	fnn = buildNetwork( trndata.indim, hidden_size , trndata.outdim, outclass=SoftmaxLayer )	
trainer = BackpropTrainer( fnn, dataset=trndata,momentum=0.05, learningrate=0.1 , verbose=False, weightdecay=0.01)
trainer.trainUntilConvergence(verbose = False, validationProportion = 0.15, maxEpochs = 100, continueEpochs = 10 )
NetworkWriter.writeToFile(fnn, 'oliv.xml')
predictions=trainer.testOnClassData (dataset=tstdata)
y_test,y_pred = tstdata['class'],predictions
      
                       
# Saving model to disk
pickle.dump(trainer, open('neural_network.pkl','wb'))
