# import the necessary Python libraries and the dataset
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# secondary detect method
from langdetect import detect

# offline dataset
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'dataset.csv')
data = pd.read_csv(filename)
# online dataset
# data = pd.read_csv(https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
# print(data.head())

# check if dataset contains any null values
# print(data.isnull().sum())

# languages present in this dataset
# print(data["language"].value_counts())

# split the data into training and test sets
x = np.array(data["Text"])
y = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# train the language detection model on multiclass classification
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# check accuracy score of dataset
# print(model.fit(X_train, y_train))
# print(model.score(X_test, y_test))

# detect the language of a text by taking a user input
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)

# Use the langdetect library to detect the language (secondary detect method)
langdetect_lang = detect(user)
print(f'{langdetect_lang}')
