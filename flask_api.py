from flask import Flask, request, jsonify
from pipeline.article_prediction import ModelLoader, CategoryMapping, ArticlePredictor, TextExtractor  # Importing from the separate file

app = Flask(__name__)

# Instantiate the classes
model_loader = ModelLoader(model_path='./models/best_model.pkl', vectorizer_path='./models/vectorizer.pkl')
category_mapping = CategoryMapping()
article_predictor = ArticlePredictor(model_loader=model_loader, category_mapping=category_mapping)
text_extractor = TextExtractor()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict article type based on the URL.
    """
    data = request.json
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Fetch and extract content from the URL
    text_content = text_extractor.extract_text_from_url(url)

    if not text_content:
        return jsonify({'error': f'Could not fetch content from URL: {url}'}), 400

    # Predict the article type based on the content
    article_type = article_predictor.predict_category(text_content)

    return jsonify({
        'URL': url,
        'ArticleType': article_type
    })

if __name__ == '__main__':
    app.run(debug=True)
