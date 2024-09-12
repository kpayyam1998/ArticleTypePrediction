from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Created a class to handle model loading, prediction, and category mapping
class ArticleClassifier:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.lg_model = None
        self.tfid_vectorizer = None
        self.Article_type = {
            "Commercial": 0,
            "Military": 1,
            "Executives": 2,
            "Others": 3,
            "Support & Services": 4,
            "Financing": 5,
            "Training": 6
        }

        """
        we can give directly instead of reverse category mapping but we need to chnage the above dictionary.
        While training i have used above dict format to convert articletype to numerical category the same format i have used everywhere

        """
        # Create a reverse mapping from numbers to category names
        self.reverse_category_mapping = {v: k for k, v in self.Article_type.items()} 

    def load_model_and_vectorizer(self):
        
        if self.lg_model is None or self.tfid_vectorizer is None:
            print("Loading model and vectorizer...")
            self.lg_model = pickle.load(open(self.model_path, 'rb'))
            self.tfid_vectorizer = pickle.load(open(self.vectorizer_path, 'rb'))
        else:
            print("Model and vectorizer already loaded.")
    
    def predict_category(self, text):
        
        # Transform the input text into a TF-IDF vector
        input_vector = self.tfid_vectorizer.transform([text])
        
        # Predict the category using the loaded model
        prediction_result = self.lg_model.predict(input_vector)
        
        # Map the result to the article type
        return self.reverse_category_mapping.get(prediction_result[0], "Unknown Category")

# Initialize the ArticleClassifier with model and vectorizer paths
article_classifier = ArticleClassifier(
    model_path='../models/best_model.pkl', 
    vectorizer_path='../models/vectorizer.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict article type based on input text.
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Load model and vectorizer (only if not already loaded)
    article_classifier.load_model_and_vectorizer()

    # Get predicted article type
    articleType = article_classifier.predict_category(text)
    
    return jsonify({'ArticleType': articleType})

if __name__ == '__main__':
    app.run(debug=True)
