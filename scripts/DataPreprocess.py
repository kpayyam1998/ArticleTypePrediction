import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

#from DataLoader import DataLoader
# Ensure NLTK dependencies are downloaded
# nltk.download()
# nltk.download('stopwords')
# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4') If already downloaded then no need to mention it

import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class for preprocessing textual data for machine learning tasks.
    """

    def __init__(self):
        # Initialize stopwords for text preprocessing
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """
        Preprocess the input text by applying various text cleaning steps.

        Args:
        - text (str): The raw text input.

        Returns:
        - str: The preprocessed text.
        """
        # Step 1: Remove HTML tags
        text = self.html_tag_remove(text)

        # Step 2: Convert to lowercase
        text = text.lower()

        # Step 3: Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Step 4: Remove special characters and punctuation
        text = re.sub(r'\W', ' ', text)

        # Step 5: Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 6: Tokenize the text
        words = word_tokenize(text)

        # Step 7: Remove stopwords
        words = [word for word in words if word not in self.stop_words]

        # Rejoin the cleaned words into a single string
        cleaned_text = ' '.join(words)

        return cleaned_text

    def html_tag_remove(self, text):
        """
        Remove HTML tags from the given text.

        Returns:
        - str: The text without HTML tags.
        """
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text()
        return clean_text

    def map_article_type(self, df, article_type_mapping):
        """
        Map article types to numerical values based on a given mapping.

        Returns:
        - pandas.DataFrame
        """
        df['CleanedArticle_Type'] = df['Article_Type'].map(article_type_mapping)
        return df

    def clean_dataframe(self, df):
        """
        Preprocess the data in the dataframe by cleaning text columns and dropping unnecessary columns.

        Returns:
        - pandas.DataFrame: The cleaned dataframe with relevant columns.
        """
        # Apply text preprocessing to the relevant columns
        df['CleanedHeading'] = df['Heading'].apply(lambda x: self.preprocess(x))
        df['CleanedArticleDesc'] = df['Article.Description'].apply(lambda x: self.preprocess(x))
        df['ClenedFullArticle'] = df['Full_Article'].apply(lambda x: self.preprocess(x))

        # Drop unnecessary columns
        df_cleaned = df.drop(columns=['Id', 'Heading', 'Article.Banner.Image', 'Article.Description', 'Full_Article', 'Outlets'], axis=1)

        return df_cleaned

    def finalize_dataframe(self, df):
        """
        Finalize the dataframe by dropping cleaned text columns and keeping only the necessary ones.

        Returns:
        - pandas.DataFrame: The final dataframe with only necessary columns.
        """
        df['CombinedArticle']=df["CleanedHeading"]+df["CleanedArticleDesc"]+df["ClenedFullArticle"]
        df = df.drop(columns=['Article_Type','Tonality','CleanedHeading', 'CleanedArticleDesc', 'ClenedFullArticle'], axis=1)

        df['CleanedArticle_Type']=df['CleanedArticle_Type'].fillna(value=1)
        df['CombinedArticle'] = df['CombinedArticle'].replace(to_replace=r'^\s*$', value="No Text", regex=True) # Instead of no text we can give custom text as well
        
        return df

# if __name__ == "__main__":
#     import pandas as pd
#     # Load the article data
#     file_path = "../data/articles.csv"  
#     loader = DataLoader(file_path)
    
#     # Load the article data
#     df = loader.load_data()

#     # Example article type mapping
#     Article_type = {
#         "Commercial": 0,
#         "Military": 1,
#         "Executives": 2,
#         "Others": 3,
#         "Support & Services": 4,
#         "Financing": 5,
#         "Training": 6
#     }

#     # Instantiate the preprocessor
#     preprocessor = DataPreprocessor()

#     # Preprocess the dataframe
#     cleaned_df = preprocessor.clean_dataframe(df)
    
#     # Map article types to numerical values
#     cleaned_df = preprocessor.map_article_type(cleaned_df, Article_type)

#     # Finalize the dataframe by dropping extra columns
#     final_df = preprocessor.finalize_dataframe(cleaned_df)

#     # Save the cleaned dataframe if needed
#     final_df.to_csv('../data/CleanedData.csv', index=False)
