from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Load your trained model and TF-IDF vectorizer
model = pickle.load(open('model_predict.pkl', 'rb'))

tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the input text
        text_tfidf = tfidf_vectorizer.transform([text])
        # Make a prediction
        predicted_rating = model.predict(text_tfidf)[0]
        return render_template('index.html', text=text, predicted_rating=predicted_rating)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)