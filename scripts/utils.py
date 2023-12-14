import logging
import re
import pandas as pd
import os
import nltk
from sklearn.model_selection import train_test_split
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from joblib import load, dump
import pickle
from sklearn.preprocessing import LabelEncoder
import logging
import warnings
from utils import *
from sklearn.model_selection import train_test_split
nlp = spacy.load('en_core_web_sm')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Function
def load_data(file_path, chunk_size=1000):
    """
    Load data from a CSV file in chunks to manage memory usage.
    Args:
    file_path: relative or absolute path to the CSV file.
    chunk_size (Optional): size of chunks to load. 
    Returns:
    DataFrame
    """
    try:
        logging.info("Starting execution of load_data")
        chunk_list = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, ignore_index=True)
        logging.info("Data loading completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data from %s: %s:", file_path, e)
        return None
    
# Function Custom Text Cleaning
def custom_clean_text(text):
    """
    Perform custom cleaning of text data for Seq2Seq model preparation.
    text (str): The text to be cleaned.
    Returns:
    str: Cleaned text.
    """
    try:
        # Remove URLs, user mentions, hashtags, and special characters
        text = re.sub(r'http\S+|@\w+|#\w+|[^\w\s]', '', text)
        # Convert to lowercase and remove extra spaces
        text = ' '.join(text.lower().strip().split())
        return text
    except Exception as e:
        logging.error(f"Error in cleaning text for Seq2Seq: {e}")
        return None


# Function to Save Model:
def save_model(model, model_path):
    """
    Saves the given model to the specified path using pickle.
    model: The machine learning model to be saved.
    model_path (str): The path where the model should be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model using pickle
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
 

# Function to save data
def save_data(df, folder_path, file_name):
    """
    Saves the given DataFrame to a CSV file within the specified folder.
    Args:
    df (pd.DataFrame): The DataFrame to be saved.
    folder_path (str): The path to the folder where the CSV file will be saved.
    file_name (str): The name of the CSV file.
    Returns:
    str: The full path to the saved CSV file, or None if an error occurred.
    """
    try:
        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Full path for the CSV file
        full_path = os.path.join(folder_path, file_name)

        # Save the DataFrame to CSV
        df.to_csv(full_path, index=False)
        
        logging.info(f"DataFrame saved successfully to {full_path}")
        return full_path
    except Exception as e:
        logging.error(f"Error saving the DataFrame to CSV: {e}")
        return None

    
# Modeling functions
def split_data(X, y, test_size=0.2):
    """
    Split the data into train and test sets.
    Args:
    X: Features (1-D array-like).
    y: Dependent variable (1-D array-like).
    test_size (float, optional): Proportion of the dataset to include in the test split (default is 0.2).
    Returns:
    X_train, X_test, y_train, y_test: Split data.
    """
    try: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logging.info("Data successfully split into train and test sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise

def preprocess_text(text):
    """
    Perform lemmatization and POS tagging.

    Args:
    text (str): Text to be processed.

    Returns:
    dict: Dictionary containing processed text and POS tags.
    """
    if not isinstance(text, str):
        logging.warning("Non-string input encountered. Converting to string.")
        text = str(text)

    try:
        doc = nlp(text)
        processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        pos_tags = [token.pos_ for token in doc]

        return {
            'processed_text': processed_text,
            'pos_tags': pos_tags
        }
    except Exception as e:
        logging.error(f"Error in preprocess_text function: {e}")
        return {'processed_text': '', 'pos_tags': []}

def extract_entities(text):
    """
    Extract named entities using spaCy NER.

    Args:
    text (str): Text from which to extract entities.

    Returns:
    list: List of named entities.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# function to evaluate models
def evaluate_model(model, model_name, X, y):
    """
    Evaluate a model using cross-validation and return its performance metrics.
    Args:
    model: The machine learning model to be evaluated.
    model_name (str): Name of the model.
    X: Features for model training.
    y: Target variable.
    Returns:
    pd.DataFrame: A DataFrame with performance metrics for the model.
    """
    try:
        scores = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)

        # Calculating the mean of each metric
        mean_scores = {metric: scores[f'test_{metric}'].mean() for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}
        
        # Creating a DataFrame row
        df_row = pd.DataFrame({
            'Model': [model_name],
            'F1 Score': [mean_scores['f1_macro']],
            'Accuracy': [mean_scores['accuracy']],
            'Recall': [mean_scores['recall_macro']],
            'Precision': [mean_scores['precision_macro']]
        })

        logging.info(f"{model_name} evaluation completed successfully.")
        return df_row
    except Exception as e:
        logging.error(f"Error in evaluating model {model_name}: {e}")
        return pd.DataFrame({
            'Model': [model_name],
            'F1 Score': [None],
            'Accuracy': [None],
            'Recall': [None],
            'Precision': [None]
            })