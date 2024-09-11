from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Define the mapping of article types
Article_type = {
    "Commercial": 0,
    "Military": 1,
    "Executives": 2,
    "Others": 3,
    "Support & Services": 4,
    "Financing": 5,
    "Training": 6
}

# Create a reverse mapping from numbers to category names
reverse_category_mapping = {v: k for k, v in Article_type.items()}

# Load the model and vectorizer globally on first request
lg_model = None
tfid_vectorizer = None

def load_model_and_vectorizer():
    """
    Load the pre-trained model and vectorizer if not already loaded.
    """
    global lg_model, tfid_vectorizer
    if lg_model is None or tfid_vectorizer is None:
        lg_model = pickle.load(open('../models/best_model.pkl', 'rb'))
        tfid_vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

def predict_category(text, model, vectorizer):
    """
    Given the input text, predict the article category using the loaded model and vectorizer.
    """
    # Use the fitted vectorizer to transform the input text
    input_vector = vectorizer.transform([text])
    prediction_result = model.predict(input_vector)
    return reverse_category_mapping.get(prediction_result[0], "Unknown Category")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict article type based on input text.
    """
    data = request.json
    text = data.get('text', '')
    print("Text")
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Load model and vectorizer (only if not already loaded)
    load_model_and_vectorizer()

    # Get predicted article type
    articleType = predict_category(text, lg_model, tfid_vectorizer)
    return jsonify({'ArticleType': articleType})

if __name__ == '__main__':
    app.run(debug=True)
