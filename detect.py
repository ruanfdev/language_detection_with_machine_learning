from langdetect import detect
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the data from the CSV file
df = pd.read_csv(
    'https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv', names=['text', 'lang'])

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract features from the text using the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data['text'])
train_labels = train_data['lang']

# Train a linear support vector classifier on the training data
classifier = LinearSVC()
classifier.fit(train_features, train_labels)

# Evaluate the classifier on the test data
test_features = vectorizer.transform(test_data['text'])
test_labels = test_data['lang']
accuracy = classifier.score(test_features, test_labels)
print(f'Accuracy: {accuracy}')

# Use the classifier to detect the language of a given text
text = input("Enter text to identify:")
text_features = vectorizer.transform([text])
predicted_lang = classifier.predict(text_features)[0]
print(f'The language of "{text}" is {predicted_lang}')

# Use the langdetect library to verify the result
langdetect_lang = detect(text)
print(
    f'The language of "{text}" according to langdetect is {langdetect_lang}')
