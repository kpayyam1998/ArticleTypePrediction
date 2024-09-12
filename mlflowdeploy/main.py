"""
This mlflow deployment script file is not fully completed yet.
"""

import pandas as pd
from scripts.DataLoader import DataLoader
from scripts.DataPreprocess import DataPreprocessor
from scripts.ModelTraining import ModelSelector, ModelTraining
from mlflowdeploy.MLFlowTracker import MLFlowTracker  
import pickle

def main():
    # Initialize MLFlow tracker
    tracker = MLFlowTracker(experiment_name="ArticleTypeClassification")
    
    # Start a new run
    tracker.start_run(run_name="Logistic Regression with TF-IDF")

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

    # Preprocess the data
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.clean_dataframe(df)
    cleaned_df = preprocessor.map_article_type(cleaned_df, Article_type)
    final_df = preprocessor.finalize_dataframe(cleaned_df)

    # Save the cleaned dataframe
    final_df.to_csv('./data/CleanedData.csv', index=False)
    tracker.log_artifact('./data/CleanedData.csv')  # Log cleaned data as an artifact

    # Prepare features and target
    X = final_df['CombinedArticle']
    y = final_df['CleanedArticle_Type']

    # Instantiate the model trainer and selector
    trainer = ModelTraining()
    selector = ModelSelector()

    # Vectorize the text
    X_tfidf, vectorizer = trainer.vectorize_text(X)

    # Log vectorizer as an artifact
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    tracker.log_artifact("models/vectorizer.pkl", "vectorizer")

    # Split the data
    X_train, X_test, y_train, y_test = trainer.split_data(X_tfidf, y)

    # Define and select the best model
    models = selector.get_model_params()
    best_model, best_model_name, results = selector.select_best_model(X_train, y_train, X_test, y_test, models)

    # Log model parameters and metrics
    tracker.log_params({"model_name": best_model_name})
    tracker.log_metrics(results)

    # Save the best model
    trainer.save_model("models", "best_model.pkl")
    tracker.log_model(best_model, "best_model")

    # Save results and accuracy
    train_accuracy, _ = selector.evaluate_model(best_model, X_train, y_train)
    test_accuracy, _ = selector.evaluate_model(best_model, X_test, y_test)
    trainer.save_accuracy_split("metrics", train_accuracy, test_accuracy)
    
    # Log additional metrics
    tracker.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

    # End the MLflow run
    tracker.end_run()

if __name__ == "__main__":
    main()




