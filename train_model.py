import pandas as pd
from scripts.DataLoader import DataLoader
from scripts.DataPreprocess import DataPreprocessor
from scripts.ModelTraining import ModelSelector, ModelTraining
import pickle
def main():
    # Load the article data
    file_path = "./data/articles.csv"
    loader = DataLoader(file_path)
    df = loader.load_data()

    # Define article type mapping
    Article_type = {
        "Commercial": 0,
        "Military": 1,
        "Executives": 2,
        "Others": 3,
        "Support & Services": 4,
        "Financing": 5,
        "Training": 6
    }

    # Instantiate the preprocessor and preprocess the data
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.clean_dataframe(df)
    cleaned_df = preprocessor.map_article_type(cleaned_df, Article_type)
    final_df = preprocessor.finalize_dataframe(cleaned_df)

    # Save the cleaned dataframe
    final_df.to_csv('./data/CleanedData.csv', index=False)

    # Define category mapping
    category_mapping = Article_type

    # Prepare features and target
    X = final_df['CombinedArticle']
    y = final_df['CleanedArticle_Type']

    # Instantiate the model trainer and selector
    trainer = ModelTraining()
    selector = ModelSelector()

    # Vectorize the text
    X_tfidf,vectorizer = trainer.vectorize_text(X)
    #X_Bert = trainer.tokenize_text(X)

 
    # Save Vect
    #pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb")) #Save vectorizer
    # Split the data
    X_train, X_test, y_train, y_test = trainer.split_data(X_tfidf, y)

    # Define and select the best model
    models = selector.get_model_params()
    best_model, best_model_name, results = selector.select_best_model(X_train, y_train,X_test,y_test, models)

    # Set the best model in the ModelTraining instance
    trainer.model = best_model

    # Save the best model
    trainer.save_model("models", "best_model.pkl")

    # Save results and accuracy
    trainer.save_results("metrics", results)
    train_accuracy, _ = selector.evaluate_model(best_model, X_train, y_train)
    test_accuracy, _ = selector.evaluate_model(best_model, X_test, y_test)
    trainer.save_accuracy_split("metrics", train_accuracy, test_accuracy)

if __name__ == "__main__":
    main()

