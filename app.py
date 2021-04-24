# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:24:20 2021

@author: saikrishna
"""

import numpy as np
import pandas as pd
import pickle
import sexmachine.detector as gender
from pybrain.datasets import ClassificationDataSet
from datetime import date
from datetime import datetime
import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import sexmachine.detector as gender
import time
import random
from flask import Flask, request, render_template


app = Flask(__name__)

#models
random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
naive_bayes = pickle.load(open('naive_bayes.pkl', 'rb'))
neural_network = pickle.load(open('neural_network.pkl', 'rb'))
decision_tree = pickle.load(open('decision_tree.pkl', 'rb'))


#location
friends_list = pickle.load(open('friends_count.pkl', 'rb',))
followers_list = pickle.load(open('followers_count.pkl', 'rb'))
favourites_list = pickle.load(open('favourites_count.pkl', 'rb'))
statuses_list = pickle.load(open('statuses_count.pkl', 'rb'))
location_dict = pickle.load(open('location_dict_scraper.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scrape_predict',methods=['POST'])
def scrape_predict():
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications" : 2}
    chrome_options.add_experimental_option("prefs",prefs)

    driver = webdriver.Chrome('/usr/local/bin/chromedriver', chrome_options=chrome_options)

    #open the webpage
    driver.get("http://www.facebook.com")

    #target username
    username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='email']")))
    password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='pass']")))

    #enter username and password
    username.clear()
    username.send_keys("9490461737")
    password.clear()
    password.send_keys("Facebook@62892")

    #target the login button and click it
    button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()
    time.sleep(10)

    #We are logged in!
    url='https://www.facebook.com/saikrishna62892/'
    driver.get(url)
    time.sleep(10)
    html = driver.page_source 

    #['created_at','statuses_count','followers_count','favourites_count','sex_code','lang_code']

    #1.scraping username section
    #gmql0nx0.l94mrbxd.p1ri9a11.lzcic4wl.bp9cbjyn.j83agx80
    elems = driver.find_elements_by_class_name("gmql0nx0.l94mrbxd.p1ri9a11.lzcic4wl.bp9cbjyn.j83agx80")
    username = elems[0].text
    username = pd.Series(username)
    #predicting sex
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= username.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    print sex_code[0]

    #2.scraping bio section
    #d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb mdeji52x a5q79mjw g1cxx5fr knj5qynh m9osqain oqcyycmt
    elems = driver.find_elements_by_class_name("d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.lr9zc1uh.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.d3f4x2em.fe6kdd0r.mau55g9w.c8b282yb.mdeji52x.a5q79mjw.g1cxx5fr.knj5qynh.m9osqain.oqcyycmt")
    bio = elems[0].text
    print bio

    #3.scraping friends count,statuses_count,followers_count,favourites_count
    #d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh e9vueds3 j5wam9gi knj5qynh m9osqain
    #d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v lrazzd5p m9osqain
    elems = driver.find_elements_by_class_name("d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.lr9zc1uh.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.d3f4x2em.fe6kdd0r.mau55g9w.c8b282yb.iv3no6db.jq4qci2q.a3bd9o3v.lrazzd5p.m9osqain")
    friend_count = elems[2].text
    if str(friend_count[-1])!='s':
        friend_count = int(str(friend_count[-1]))
    else:
        friend_count = random.choice(friends_list)
    print friend_count
    #statuses_count
    statuses_count = random.choice(statuses_list)
    print statuses_count

    #followers_count
    followers_count = random.choice(followers_list)
    print followers_count

    #favourites_count
    favourites_count = random.choice(favourites_list)
    print favourites_count

    #4.scraping location
    #oajrlxb2 g5ia77u1 qu0x051f esr5mh6w e9989ue4 r7d6kgcz rq0escxv nhd2j8a9 nc684nl6 p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x jb3vyjys rz4wbd8a qt6c0cv9 a8nywdso i1ao9s8h esuyzwwr f1sip0of lzcic4wl oo9gr5id gpro0wi8 lrazzd5p
    elems = driver.find_elements_by_class_name("oajrlxb2.g5ia77u1.qu0x051f.esr5mh6w.e9989ue4.r7d6kgcz.rq0escxv.nhd2j8a9.nc684nl6.p7hjln8o.kvgmc6g5.cxmmr5t8.oygrvhab.hcukyx3x.jb3vyjys.rz4wbd8a.qt6c0cv9.a8nywdso.i1ao9s8h.esuyzwwr.f1sip0of.lzcic4wl.oo9gr5id.gpro0wi8.lrazzd5p")
    location = elems[2].text
    if location in location_dict:
        location = location_dict[location]
    else:
        location_dict[location]=len(location_dict)+1
        location = location_dict[location]
        pickle.dump(location_dict, open('location_dict_scraper.pkl','wb'),protocol=2)
    print location

    #5.scraping created_at
    #d2edcug0 hpfvmrgz qv66sw1b c1et5uql lr9zc1uh a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d3f4x2em fe6kdd0r mau55g9w c8b282yb iv3no6db jq4qci2q a3bd9o3v knj5qynh oo9gr5id hzawbc8m
    elems = driver.find_elements_by_class_name("d2edcug0.hpfvmrgz.qv66sw1b.c1et5uql.lr9zc1uh.a8c37x1j.keod5gw0.nxhoafnm.aigsh9s9.d3f4x2em.fe6kdd0r.mau55g9w.c8b282yb.iv3no6db.jq4qci2q.a3bd9o3v.knj5qynh.oo9gr5id.hzawbc8m")
    created_at = elems[7].text
    created_at = '01 '+created_at[10:]
    created_date = datetime.datetime.strptime(datetime.datetime.strptime(created_at, '%d %B %Y').strftime('%m %d %Y'),'%m %d %Y')
    today =  datetime.datetime.strptime(datetime.datetime.now().strftime('%m %d %Y'),'%m %d %Y') 
    days_count = today - created_date
    days_count = days_count.days
    print days_count

    #6.language
    #lang
    lang_dict = {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2, 'hi':8 ,'other': 9}
    
    #['created_at','location','statuses_count','followers_count','favourites_count','friends_count','sex_code','lang_code']
    df=pd.DataFrame({'bio':bio,
                     'statuses_count':statuses_count,
                     'followers_count':followers_count,
                     'friends_count':friends_count,
                     'favourites_count':favourites_count,
                     'created_at':days_count,
                     'location':location,
                     'sex_code':sex_code,
                     'lang':lang_dict['other']}, index=[0])
    params = pd.Series([df['created_at'],df['location'],df['statuses_count'],df['followers_count'],df['favourites_count'],df['friends_count'],sex_code,df['lang']])
    
    #Random forest prediction
    rfr_prediction = random_forest.predict(params)
    
    #support vector machine prediction
    svm_prediction = support_vector.predict(params)
    
    #Naive Bayes prediction
    nvb_prediction = naive_bayes.predict(params)
    
    #Decision Tree Prediction
    dtc_prediction = decision_tree.predict(params)
    
    #neural network prediction
    ds2 = ClassificationDataSet( 8, 1,nb_classes=2)
    lst = [df['created_at'],df['location'],df['statuses_count'],df['followers_count'],df['favourites_count'],df['friends_count'],sex_code,df['lang'].astype(int)]
    ds2.addSample(lst,1)
    ds2._convertToOneOfMany( )
    fnn_prediction=neural_network.testOnClassData (dataset=ds2)
    
    
    percent = ( dtc_prediction[0] + nvb_prediction[0] + rfr_prediction[0] + svm_prediction[0] + fnn_prediction[0] )
    percent = round(percent * 20)
    
    return render_template('result.html', features=params)



@app.route('/predict',methods=['POST'])
def predict():
    '''
    ['username','location','statuses_count','followers_count','friends_count','favourites_count','sex_code','lang_code','created_at']
    {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2}
    '''
    #request.form.values()
    data=request.form
    int_features = list(data.values())
    #lang
    lang_dict = {'fr': 3, 'en': 1, 'nl': 6, 'de': 0, 'tr': 7, 'it': 5, 'gl': 4, 'es': 2, 'hi':8 ,'other': 9}
    
    '''
    #location
    users = pd.read_csv(r'/home/vamsi82674/Desktop/fake profile detection fb/app/data/processed_data.csv')
    location_list = list(enumerate(np.unique(users['location'])))   
    location_dict = { name : i for i, name in location_list }
    location_dict['other']=1679
    '''          

    #created_at
    created_date = datetime.datetime.strptime(datetime.datetime.strptime(int_features[7], '%Y-%m-%d').strftime('%m %d %Y'),'%m %d %Y')
    today =  datetime.datetime.strptime(datetime.datetime.now().strftime('%m %d %Y'),'%m %d %Y') 
    days_count = today - created_date
    days_count = days_count.days

    #for local host
    df=pd.DataFrame({'bio':int_features[0],
                     'statuses_count':int_features[1],
                     'followers_count':int_features[5],
                     'friends_count':int_features[2],
                     'favourites_count':int_features[8],
                     'created_at':int_features[7],
                     'location':location_dict[int_features[6]],
                     'username':int_features[9],
                     'lang':lang_dict[int_features[3]]}, index=[0])

    '''
    #for heroku
    #[u'1', u'4', u'2', u'sai', u'other', u'3', u'en', u'2021-04-04', u'bio\r\n', u'']
    df=pd.DataFrame({'bio':,
                     'statuses_count':,
                     'followers_count':,
                     'friends_count':,
                     'favourites_count':,
                     'created_at':,
                     'location':location_dict[],
                     'username':,
                     'lang':lang_dict[]}, index=[0])
    '''
    
    #created_at
    created_date = datetime.datetime.strptime(datetime.datetime.strptime(df.loc[0,'created_at'], '%Y-%m-%d').strftime('%m %d %Y'),'%m %d %Y')
    today =  datetime.datetime.strptime(datetime.datetime.now().strftime('%m %d %Y'),'%m %d %Y') 
    days_count = today - created_date
    days_count = days_count.days
    df.loc[0,'created_at'] = days_count

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
    
    #Naive Bayes prediction
    nvb_prediction = naive_bayes.predict(params)
    
    #Decision Tree Prediction
    dtc_prediction = decision_tree.predict(params)
    
    #neural network prediction
    ds2 = ClassificationDataSet( 8, 1,nb_classes=2)
    lst = [df['created_at'],df['location'],df['statuses_count'],df['followers_count'],df['favourites_count'],df['friends_count'],sex_code,df['lang'].astype(int)]
    ds2.addSample(lst,1)
    ds2._convertToOneOfMany( )
    fnn_prediction=neural_network.testOnClassData (dataset=ds2)
    
    
    percent = ( dtc_prediction[0] + nvb_prediction[0] + rfr_prediction[0] + svm_prediction[0] + fnn_prediction[0] )
    percent = round(percent * 20)
    
    return render_template('result.html', dtc_prediction = dtc_prediction[0] , nvb_prediction = nvb_prediction[0] ,rfr_prediction = rfr_prediction[0],svm_prediction = svm_prediction[0],fnn_prediction = fnn_prediction[0],percentage=percent,features=int_features) 

    return render_template('index.html',features=int_features)

if __name__ == "__main__":
    app.run(debug=True)