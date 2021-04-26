
# coding: utf-8

# In[1]:

#imports here
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import sexmachine.detector as gender
import pandas as pd
import time
import datetime
import pickle
import random


# In[2]:

#code by pythonjar, not me
chrome_options = webdriver.ChromeOptions()
prefs = {"profile.default_content_setting_values.notifications" : 2}
chrome_options.add_experimental_option("prefs",prefs)


# In[3]:

'''
url='https://www.facebook.com/saikrishna62892/'
driver.get(url)
time.sleep(2)
html = driver.page_source
df = pd.DataFrame([html])
df.to_clipboard(index=False,header=False)
'''


# In[4]:

import pandas as pd


#specify the path to chromedriver.exe (download and save on your computer)
driver = webdriver.Chrome('C:/Users/vamsi/chromedriver.exe', chrome_options=chrome_options)

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
url='https://www.facebook.com/sdmadhukumar/'
driver.get(url)
time.sleep(10)
html = driver.page_source
#df = pd.DataFrame([html])
#df.to_clipboard(index=False,header=False)


# In[5]:

#pickles
friends_list = pickle.load(open('friends_count.pkl', 'rb',))
followers_list = pickle.load(open('followers_count.pkl', 'rb'))
favourites_list = pickle.load(open('favourites_count.pkl', 'rb'))
statuses_list = pickle.load(open('statuses_count.pkl', 'rb'))
location_dict = pickle.load(open('location_dict_scraper.pkl', 'rb'))


# In[6]:

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
print len(elems)
for elem in elems:
    print elem.text
created_at = elems[7].text
created_at = '01 '+created_at[10:]
created_date = datetime.datetime.strptime(datetime.datetime.strptime(created_at, '%d %B %Y').strftime('%m %d %Y'),'%m %d %Y')
today =  datetime.datetime.strptime(datetime.datetime.now().strftime('%m %d %Y'),'%m %d %Y') 
days_count = today - created_date
days_count = days_count.days
print days_count


# In[ ]:




# In[ ]:




# In[ ]:



