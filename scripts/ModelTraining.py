import os
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
import torch

class ModelSelector:
    def __init__(self, random_state=33):
        self.random_state = random_state

    def get_model_params(self):
        """
        Define models and their parameter distributions for hyperparameter tuning.
        """
        models = {
            "Logistic Regression": (LogisticRegression(), {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }),
            "Decision Tree": (DecisionTreeClassifier(), {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }),
            "Random Forest": (RandomForestClassifier(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            })
            # We can add more params but it will take long time to train
            # "Gradient Boosting": (GradientBoostingClassifier(), {
            #     'n_estimators': [50, 100, 200],
            #     'learning_rate': [0.001, 0.01, 0.1, 1],
            #     'max_depth': [3, 5, 7]
            # }),
            # "AdaBoost": (AdaBoostClassifier(), {
            #     'n_estimators': [50, 100, 200],
            #     'learning_rate': [0.001, 0.01, 0.1, 1]
            # }),
            # "Naive Bayes": (MultinomialNB(), {
            #     'alpha': [0.001, 0.01, 0.1, 1]
            # })
        }
        return models

    def select_best_model(self, X_train, y_train,X_test, y_test, models, n_iter=100, cv=5):
        """
        Perform hyperparameter tuning and select the best model.
        """
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        results = {}

        for model_name, (model, params) in models.items():
            print(f"Training and tuning {model_name}...")
            random_search = RandomizedSearchCV(model, param_distributions=params, 
                                               n_iter=n_iter, cv=cv, random_state=self.random_state, n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_model_candidate = random_search.best_estimator_
            
            # Evaluate the model
            train_accuracy, _ = self.evaluate_model(best_model_candidate, X_train, y_train)
            test_accuracy, report = self.evaluate_model(best_model_candidate, X_test, y_test)
            
            results[model_name] = {'accuracy': test_accuracy, 'report': report, 'best_params': best_params}

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = best_model_candidate
                best_model_name = model_name

        return best_model, best_model_name, results

    def evaluate_model(self, model, X, y):
        """
        Evaluate the model's performance on the given data.
        """
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

class ModelTraining:
    def __init__(self, max_features=5000, stop_words='english', random_state=33):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.model = None
        self.random_state = random_state

    def vectorize_text(self, X):
        """
        Vectorize the text data using TF-IDF.
        """
        return self.vectorizer.fit_transform(X),self.vectorizer

        # or Below Bert BertTokenizer belwo we can use while loading its taking more time in my machine

    # def __init__(self):
    #     # Load pre-trained BERT model and tokenizer
    #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     self.bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # def tokenize_text(self, texts, max_length=128):
    #     """
    #     Tokenizes and converts input text to BERT embeddings
    #     """
    #     encoded_inputs = self.tokenizer(
    #         texts.tolist(),            # Convert DataFrame column to list
    #         padding=True,              # Pad to the longest sentence
    #         truncation=True,           # Truncate sentences to max_length
    #         max_length=max_length,     # Set maximum length
    #         return_tensors='pt'        # Return as PyTorch tensors
    #     )
        
    #     # Get the BERT embeddings from the model
    #     with torch.no_grad():  
    #         output = self.bert_model(**encoded_inputs)
        
    #     # Use the [CLS] token embeddings (1st token) as the representation
    #     cls_embeddings = output.last_hidden_state[:, 0, :]
        
    #     return cls_embeddings 

    def split_data(self, X, y, test_size=0.1):
        """
        Split the data into training and testing sets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def save_model(self, folder_path, filename="best_model.pkl"):
        """
        Save the trained model to a pickle file.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {file_path}")

    def save_results(self, folder_path, results, filename="model_results.txt"):
        """
        Save the model accuracy and classification reports to a text file.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w") as f:
            for model_name, result in results.items():
                f.write(f"{model_name} - Best Params: {result['best_params']}\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write("Classification Report:\n")
                f.write(result['report'])
                f.write("\n" + "="*50 + "\n")
        print(f"Results saved to {file_path}")

    def save_accuracy_split(self, folder_path, train_accuracy, test_accuracy, filename="accuracy.txt"):
        """
        Save the train and test accuracy to a text file.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w") as f:
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        print(f"Accuracy details saved to {file_path}")



# if __name__ == "__main__":
#     # Load your dataframe
#     df = pd.read_csv('../data/CleanedData.csv')

#     # Define category mapping
#     Article_type = {
#         "Commercial": 0,
#         "Military": 1,
#         "Executives": 2,
#         "Others": 3,
#         "Support & Services": 4,
#         "Financing": 5,
#         "Training": 6
#     }

#     # Example combined text data and target labels
#     X = df['CombinedArticle']
#     y = df['CleanedArticle_Type']

#     #print(X)
#     # Instantiate the classifier
#     classifier = ModelTraining()

#     # Step 1: Vectorize the text data
#     X_tfidf = classifier.vectorize_text(X)

#     # Step 2: Split the data
#     X_train, X_test, y_train, y_test = classifier.split_data(X_tfidf, y)

#     # Step 3: Train the model
#     classifier.train_model(X_train, y_train)

#     # Step 4: Evaluate the model
#     test_accuracy, report = classifier.evaluate_model(X_test, y_test)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print("Classification Report:\n", report)

#     # Calculate training accuracy
#     train_accuracy, _ = classifier.evaluate_model(X_train, y_train)
#     print(f"Training Accuracy: {train_accuracy:.4f}")

#     # Step 5: Save the evaluation results and accuracy
#     result_folder = "Metrics"  # Define the folder to save the results
#     classifier.save_results(result_folder, test_accuracy, report)
#     classifier.save_accuracy_split(result_folder, train_accuracy, test_accuracy)

