import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)

nltk.download('stopwords')

review = re.sub('[^a-zA-Z]', ' ', (data['Review'][0]))

review = review.lower()
review = review.split()
review = [word for word in review if word not in stopwords.words('english')]

ps = PorterStemmer()

review = [ps.stem(word) for word in review]
review =  ' '.join(review)
print(review)

