from flask import Flask, request, jsonify, render_template
import re, os
import pickle
import nltk
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer



app = Flask(__name__)

# Load saed model and vectorizer
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

vect_path = os.path.join('models', 'vectorizer.pkl')
with open(vect_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocessing function

def preprocess(text):
    snow=SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [snow.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')   # Render the HTML template for the home page

# Predict function
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'info': 'Send a POST request with a JSON body or form-data containing { "message": "Your text" }'})

    try:
        if request.is_json:
            data = request.get_json()
            message = data.get("message", "")
        else:
            message = request.form.get("body", "")

        if not message.strip():
            return jsonify({'error': 'Empty message'}), 400

        cleaned = preprocess(message)
        vect_msg = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vect_msg)[0]

        return render_template('index.html', prediction="SPAM" if prediction == 1 else "HAM")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=8080, debug=True)


port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
