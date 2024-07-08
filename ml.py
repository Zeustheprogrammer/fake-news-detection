import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pickle
from utils import wordopt

# Load datasets
true_df = pd.read_csv("datasets/True.csv")
false_df = pd.read_csv("datasets/Fake.csv")

# Adding class labels
true_df['class'] = 1
false_df['class'] = 0

# Joining both datasets
news_df = pd.concat([true_df, false_df], axis=0)

# Dropping unnecessary columns
news_df = news_df[['text', 'class']]

# Shuffling the dataset
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Applying text cleaning
news_df['text'] = news_df['text'].apply(wordopt)

# Defining feature and target variables
x = news_df['text']
y = news_df['class']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Vectorizing text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes Classifier": MultinomialNB()
}

# Train and evaluate models
best_model_name = None
best_model = None
best_precision = 0

for name, model in models.items():
    model.fit(xv_train, y_train)
    predictions = model.predict(xv_test)
    precision = precision_score(y_test, predictions)
    print(f"{name}:")
    print(confusion_matrix(y_test, predictions))
    print(f"Precision: {precision}")
    print(classification_report(y_test, predictions))
    if precision > best_precision:
        best_model_name = name
        best_model = model
        best_precision = precision

print(f"Best Model: {best_model_name} with Precision: {best_precision}")

# Save the best model and vectorizer using pickle
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorization, vec_file)

print("Best model and vectorizer saved successfully.")
