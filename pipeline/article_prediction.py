import pickle
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

# Class to handle category mapping
@dataclass
class CategoryMapping:
    Article_type: dict = None
    reverse_category_mapping: dict = None

    def __post_init__(self):
        # Initialize article type mappings
        self.Article_type = {
            "Commercial": 0,
            "Military": 1,
            "Executives": 2,
            "Others": 3,
            "Support & Services": 4,
            "Financing": 5,
            "Training": 6
        }
        # Create reverse mapping
        self.reverse_category_mapping = {v: k for k, v in self.Article_type.items()}

    def get_category_name(self, category_id):
        return self.reverse_category_mapping.get(category_id, "Unknown Category")


# Class to handle loading the model and vectorizer
class ModelLoader:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load_model_and_vectorizer(self):
        if self.model is None or self.vectorizer is None:
            print("Loading model and vectorizer...")
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.vectorizer = pickle.load(open(self.vectorizer_path, 'rb'))
        else:
            print("Model and vectorizer already loaded.")
        return self.model, self.vectorizer


# Class to handle URL fetching and text extraction
class TextExtractor:
    @staticmethod
    def extract_text_from_url(url):
        """
        Fetch the URL content and extract text using BeautifulSoup.
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                text_content = " ".join([para.get_text() for para in paragraphs])
                return text_content
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None


# Class to handle predictions
class ArticlePredictor:
    def __init__(self, model_loader, category_mapping):
        self.model_loader = model_loader
        self.category_mapping = category_mapping

    def predict_category(self, text):
        """
        Given the text, predict the article category using the trained model and vectorizer.
        """
        # Load model and vectorizer
        model, vectorizer = self.model_loader.load_model_and_vectorizer()

        # Transform the input text into a TF-IDF vector
        input_vector = vectorizer.transform([text])

        # Predict the category using the loaded model
        prediction_result = model.predict(input_vector)

        # Map the result to the article type
        return self.category_mapping.get_category_name(prediction_result[0])



# from flask import Flask, request, jsonify
# import requests
# from bs4 import BeautifulSoup
# import pickle
# from dataclasses import dataclass

# app = Flask(__name__)

# # Class to handle category mapping
# @dataclass
# class CategoryMapping:
#     Article_type: dict = None
#     reverse_category_mapping: dict = None

#     def __post_init__(self):
#         # Initialize article type mappings
#         self.Article_type = {
#             "Commercial": 0,
#             "Military": 1,
#             "Executives": 2,
#             "Others": 3,
#             "Support & Services": 4,
#             "Financing": 5,
#             "Training": 6
#         }
#         # Create reverse mapping
#         self.reverse_category_mapping = {v: k for k, v in self.Article_type.items()}

#     def get_category_name(self, category_id):
#         return self.reverse_category_mapping.get(category_id, "Unknown Category")

# # Class to handle loading the model and vectorizer
# class ModelLoader:
#     def __init__(self, model_path, vectorizer_path):
#         self.model_path = model_path
#         self.vectorizer_path = vectorizer_path
#         self.model = None
#         self.vectorizer = None

#     def load_model_and_vectorizer(self):
#         if self.model is None or self.vectorizer is None:
#             print("Loading model and vectorizer...")
#             self.model = pickle.load(open(self.model_path, 'rb'))
#             self.vectorizer = pickle.load(open(self.vectorizer_path, 'rb'))
#         else:
#             print("Model and vectorizer already loaded.")
#         return self.model, self.vectorizer


# # Class to handle URL fetching and text extraction
# class TextExtractor:
#     @staticmethod
#     def extract_text_from_url(url):
#         """
#         Fetch the URL content and extract text using BeautifulSoup.
#         """
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.content, 'html.parser')
#                 paragraphs = soup.find_all('p')
#                 text_content = " ".join([para.get_text() for para in paragraphs])
#                 return text_content
#             else:
#                 return None
#         except requests.exceptions.RequestException as e:
#             print(f"Error fetching {url}: {str(e)}")
#             return None


# # Class to handle predictions
# class ArticlePredictor:
#     def __init__(self, model_loader, category_mapping):
#         self.model_loader = model_loader
#         self.category_mapping = category_mapping

#     def predict_category(self, text):
#         """
#         Given the text, predict the article category using the trained model and vectorizer.
#         """
#         # Load model and vectorizer
#         model, vectorizer = self.model_loader.load_model_and_vectorizer()

#         # Transform the input text into a TF-IDF vector
#         input_vector = vectorizer.transform([text])

#         # Predict the category using the loaded model
#         prediction_result = model.predict(input_vector)

#         # Map the result to the article type
#         return self.category_mapping.get_category_name(prediction_result[0])


# # Instantiate the classes
# model_loader = ModelLoader(model_path='../models/best_model.pkl', vectorizer_path='../models/vectorizer.pkl')
# category_mapping = CategoryMapping()
# article_predictor = ArticlePredictor(model_loader=model_loader, category_mapping=category_mapping)
# text_extractor = TextExtractor()


# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     API endpoint to predict article type based on the URL.
   
#     """
#     data = request.json
#     url = data.get('url', '')

#     if not url:
#         return jsonify({'error': 'No URL provided'}), 400

#     # Fetch and extract content from the URL
#     text_content = text_extractor.extract_text_from_url(url)

#     if not text_content:
#         return jsonify({'error': f'Could not fetch content from URL: {url}'}), 400

#     # Predict the article type based on the content
#     article_type = article_predictor.predict_category(text_content)

#     return jsonify({
#         'URL': url,
#         'ArticleType': article_type
#     })


# if __name__ == '__main__':
#     app.run(debug=True)
