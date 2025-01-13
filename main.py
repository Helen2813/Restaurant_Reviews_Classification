import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)

nltk.download('stopwords')

corpus = []
ps = PorterStemmer()

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', (data['Review'][i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review =  ' '.join(review)

    corpus.append(review)

cv = CountVectorizer(max_features=1500)

x = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: ', accuracy)





