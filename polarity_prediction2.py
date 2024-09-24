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

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Cell 2: Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cell 3: Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize and remove stopwords
    return text

# Cell 4: Set random seed and file paths
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pos_file = r"C:\Users\harsh\OneDrive\Desktop\python\python_project1\rt-polaritydata\rt-polarity.pos"
neg_file = r"C:\Users\harsh\OneDrive\Desktop\python\python_project1\rt-polaritydata\rt-polarity.neg"

# Cell 5: Read and preprocess data (Updated as per the provided method)
with open(pos_file, 'r') as f:
    positive_sentences = [clean_text(line.strip()) for line in f.readlines()]

with open(neg_file, 'r') as f:
    negative_sentences = [clean_text(line.strip()) for line in f.readlines()]

# Ensure we have enough data
assert len(positive_sentences) >= 5331
assert len(negative_sentences) >= 5331

# Create labels
positive_labels = [1] * len(positive_sentences)
negative_labels = [0] * len(negative_sentences)

# Combine data and labels
data = positive_sentences + negative_sentences
labels = positive_labels + negative_labels

# Shuffle data and labels
combined = list(zip(data, labels))
random.shuffle(combined)
data, labels = zip(*combined)

# Cell 6: Split data into train, validation, and test sets (Updated as per the provided method)
train_data = data[:8000]
train_labels = labels[:8000]

val_data = data[8000:9000]
val_labels = labels[8000:9000]

test_data = data[9000:]
test_labels = labels[9000:]

# Cell 7: TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=20000, 
    ngram_range=(1, 3), 
    sublinear_tf=True, 
    min_df=2, 
    max_df=0.9
)

# Fit and transform train data, transform validation and test data
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
test_features = vectorizer.transform(test_data)

# Cell 8: Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(train_features, train_labels)

# Cell 9: Make predictions
val_predictions = nb_model.predict(val_features)
test_predictions = nb_model.predict(test_features)

# Cell 10: Calculate and display metrics
# Validation and Test Accuracy
val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Precision, Recall, F1-Score for Test Set
test_report = classification_report(test_labels, test_predictions, zero_division=1, output_dict=True)
precision = test_report['1']['precision']
recall = test_report['1']['recall']
f1_score = test_report['1']['f1-score']

# Confusion Matrix for Test Set
tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()

# Display results
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

# Cell 11 (Optional): Visualize Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(7,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
