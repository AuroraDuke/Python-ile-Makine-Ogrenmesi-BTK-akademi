# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:29:15 2024

@author: ilker
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('Restaurant_Reviews.csv',delimiter='\t')



import re

def remove_non_alphabetic(text):
    return re.sub('[^a-zA-Z]', ' ', str(text))  # Replace non-alphabetic characters with space

# Apply the function to the 'Review,Liked' column
review = datas['Review,Liked'].apply(remove_non_alphabetic)

review_df = review.to_frame()

# Define a function to extract only the last numerical value
def extract_last_numerical(text):
    matches = re.findall(r'\d+', text)  # Find all numerical values
    if matches:
        return matches[-1]  # Return the last numerical value found
    else:
        return None  # Return None if no numerical value is found

# Apply the function to the 'Review,Liked' column
star = datas['Review,Liked'].apply(extract_last_numerical)
star_df = star.to_frame()


review_df.columns =['Review']
star_df.columns = ['Liked']
df = pd.concat([review_df,star_df],axis=1)

lowercased_strings = []

for reviewWlow in df.iloc[:,0]:
    lowercased_strings.append(reviewWlow.lower())
    
df['Review'] = lowercased_strings

import nltk 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
edited_strings = []
for row in range(len(lowercased_strings)):
    string = lowercased_strings[row].split()
    edited_strings_row= [ps.stem(word) for word in string if not word in stop_words]
    edited_wordlist = ' '.join(edited_strings_row)
    edited_strings.append(edited_wordlist)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(edited_strings).toarray() #undependent variables
y = star_df.values #dependent variable

from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""
review_df = pd.DataFrame(data=review,index= range(len(review)),columns=['Review'])
star_df = pd.DataFrame(data = star,index = range(len(star)),columns = ['Liked'] )
data_set = pd.concat([review_df,star_df],axis=0)
"""

