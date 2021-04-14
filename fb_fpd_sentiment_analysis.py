# -*- coding: utf-8 -*-
"""FB_FPD_Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SiGnrGqdqOLDlg2Nfg7fbhfU-clY7hzd

## Importing Libraries
"""

import numpy as np
import pandas as pd
import re
import pickle

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import gensim

"""## Loading Data"""

data = pd.read_csv("nlp_data.csv")

data.head()

data.info()

#fillna
data['description'].fillna('')

data['description'] = str(data['description'])

data.info()

"""# <font color = "green">**Processing Message** </font>"""

def processing(review):

    # Remove email addresses with 'emailaddr'    
    raw_review = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', " ", review)
    
    # Remove URLs with 'httpaddr'
    raw_review = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', " ", raw_review) 

    # Remove non-letters        
    raw_review = re.sub("[^a-zA-Z]", " ", raw_review) 
    
    # Remove numbers
    raw_review = re.sub('\d+(\.\d+)?', " ", raw_review)

    # Convert to lower case, split into individual words
    words = raw_review.lower().split()                                             

    # Gather the list of stopwords in English Language
    stops = set(stopwords.words("english"))                  

    # Remove stop words and stemming the remaining words
    meaningful_words = [ps.stem(w) for w in words if not w in stops]   

    # Join the tokens back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

# Corpus
clean_reviews_corpus = []

# Porter Stemmer
ps = PorterStemmer()

# No. of bios
review_count = data['description'].size
review_count

nltk.download('stopwords')
for i in range( 0, review_count):
    clean_reviews_corpus.append(processing(data['description'][i]))



print ("Original Text : \n")
data["description"][0]

print ("Processed Text : \n")

clean_reviews_corpus[:1]

"""# <font color = "green">**Preparing Vectors for each message** </font>"""

cv = CountVectorizer()
data_input = cv.fit_transform(clean_reviews_corpus)
data_input = data_input.toarray()

data_input[0]

data_input.size

"""# <font color = "green">**Creating WordCloud** </font>"""

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color='black', stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(clean_reviews_corpus)

"""# <font color = "green">**Applying Classification** </font>

- **Input** = Prepared Sparse Matrix / Vectors for each message

- **Output** = Negative or Positive Sentiment
"""

data_output = data['dataset']
print (data_output.value_counts())

plt.figure(figsize = (8, 8))
data['dataset'].value_counts().plot.bar()

"""#### Splitting data for Training and Testing"""

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_input, data_output,test_size= 0.20, random_state = 0)

"""## <font color = "green">Preparing ML Models</font>

### Training
"""

model_nvb = GaussianNB()
model_nvb.fit(train_x, train_y)

model_rf = RandomForestClassifier(n_estimators=1000, random_state=0)
model_rf.fit(train_x, train_y)

model_dt = tree.DecisionTreeClassifier()
model_dt.fit(train_x, train_y)

"""### Prediction"""

prediction_nvb = model_nvb.predict(test_x)
prediction_rf = model_rf.predict(test_x)
prediction_dt = model_dt.predict(test_x)

"""### Results Naive Bayes"""

print ("Accuracy for Naive Bayes : %0.5f \n\n" % accuracy_score(test_y, prediction_nvb))
print ("Classification Report Naive bayes: \n", classification_report(test_y, prediction_nvb))

"""### Results Decision Tree"""

print ("Accuracy for Decision Tree: %0.5f \n\n" % accuracy_score(test_y, prediction_dt))
print ("Classification Report Decision Tree: \n", classification_report(test_y, prediction_dt))

"""### Results Random Forest"""

print ("Accuracy for Random Forest: %0.5f \n\n" % accuracy_score(test_y, prediction_rf))
print ("Classification Report Random Forest: \n", classification_report(test_y, prediction_rf))

# Saving model to disk
pickle.dump(model_nvb, open('nvb_sentiment.pkl','wb'))
pickle.dump(model_rf, open('rf_sentiment.pkl','wb'))
pickle.dump(model_dt, open('dt_sentiment.pkl','wb'))

