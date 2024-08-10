from flask import Flask, request, render_template
from utils import wordopt
import pickle

app = Flask(__name__)

# Load# Load the model and vectorizer
with open("models/best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = wordopt(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    if prediction[0] == 0:
        result = "🚨 This news is likely fake! Stay informed and always verify the source. 🚨"
    else:
        result = "✅ This news appears to be real! Keep up with reliable sources. ✅"
    return render_template('index.html', prediction_text=result)

@app.route('/')
def home():
    return render_template('index.html')


