import pandas as pd
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  
    return text

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pos_file = r"C:\Users\harsh\OneDrive\Desktop\python\python_project1\rt-polaritydata\rt-polarity.pos"
neg_file = r"C:\Users\harsh\OneDrive\Desktop\python\python_project1\rt-polaritydata\rt-polarity.neg"

with open(pos_file, 'r') as f:
    positive_sentences = [clean_text(line.strip()) for line in f.readlines()]

with open(neg_file, 'r') as f:
    negative_sentences = [clean_text(line.strip()) for line in f.readlines()]

assert len(positive_sentences) >= 5331
assert len(negative_sentences) >= 5331

# Create labels
positive_labels = [1] * len(positive_sentences)
negative_labels = [0] * len(negative_sentences)

data = positive_sentences + negative_sentences
labels = positive_labels + negative_labels

combined = list(zip(data, labels))
random.shuffle(combined)
data, labels = zip(*combined)

train_data = data[:8000]
train_labels = labels[:8000]

val_data = data[8000:9000]
val_labels = labels[8000:9000]

test_data = data[9000:]
test_labels = labels[9000:]

vectorizer = TfidfVectorizer(
    max_features=20000, 
    ngram_range=(1, 3), 
    sublinear_tf=True, 
    min_df=2, 
    max_df=0.9
)

train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
test_features = vectorizer.transform(test_data)

nb_model = MultinomialNB()
nb_model.fit(train_features, train_labels)

val_predictions = nb_model.predict(val_features)
test_predictions = nb_model.predict(test_features)

val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

test_report = classification_report(test_labels, test_predictions, zero_division=1, output_dict=True)
precision = test_report['1']['precision']
recall = test_report['1']['recall']
f1_score = test_report['1']['f1-score']

tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")



cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(7,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
