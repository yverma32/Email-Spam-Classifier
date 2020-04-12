
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('spam.csv')

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
corpus = []
N = 5572
for i in range(0,N):
    review = re.sub('[^a-zA-Z]', ' ', dataset['EmailText'][i])
    review = review.lower()
    review = review.split ()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the bag of words model by tokenization
# Sparse Matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()

Y = dataset.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) 


# Splitting into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state= 0)

# Support Verctor Machine
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the test results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100

print(accuracy)


