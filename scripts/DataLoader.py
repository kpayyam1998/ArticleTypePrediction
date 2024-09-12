import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize the DataLoader with the file path.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Load the article dataset from a CSV file.

       
        """
        data = pd.read_csv(self.file_path,encoding='cp1252')
        print(f"Data loaded from {self.file_path}, shape: {data.shape}")
        return data

    def explore_data(self, data):
        """
        Print basic information about the dataset, such as missing values, 
        column names, and a preview of the data.
        
        """
        print("Data Info:")
        print(data.info())  # Get summary of the dataset
        
        print("\nMissing values per column:")
        print(data.isnull().sum())  # Check for missing values
        
        
        print("\nPreview of the data:")
        print(data.shape)

        print("\nPreview of the data:")
        print(data.head())  # Show first few rows of the dataset

        # We can perform more exploration operations here
# # Example usage:
# if __name__ == "__main__":
#     file_path = "../data/articles.csv"  # 
#     loader = DataLoader(file_path)
    
#     # Load the article data
#     article_data = loader.load_data()
    
#     # Explore the data
#     loader.explore_data(article_data)
