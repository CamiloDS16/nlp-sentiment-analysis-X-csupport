import logging
import re
import pandas as pd
import os
import nltk
from sklearn.model_selection import train_test_split

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
    
# Function to classify sentiment with threshold score
def classify_sentiment(score):
    """
    Classify the sentiment of the tweets based on a sentiment score.
    Returns 'Positive', 'Negative', or 'Neutral' based on the score.
    score (float): The sentiment score to classify.
    Returns:
    str: The sentiment classification ('Positive', 'Negative', 'Neutral').
    """
    try:
        if score > 0.2:
            return 'Positive'
        elif score < -0.2:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        logging.error(f"Error in classifying sentiment: {e}")
        # Optionally, return a default classification or raise the exception
        return 'Neutral'
    

# Function to Save Model:
def save_model(model, model_path):
    """
    Saves the given model to the specified path.
    model: The machine learning model to be saved.
    model_path (str): The path where the model should be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        model.save(model_path)
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

# Function to Count Ents   
def count_entity_type(entities, entity_type):
    """
    Count the number of entities of a specific type.
    Args:
        entities (list of tuples): List of entity tuples (entity, entity type).
        entity_type (str): The entity type to count.
    Returns:
        int: Count of the specified entity type.
    """
    try:
        return sum(entity[1] == entity_type for entity in entities if isinstance(entity, tuple))
    except Exception as e:
        logging.error(f'Error in counting entities of type {entity_type}: {e}')
        return 0
    

# feature engineering
def feature_engineering(df):
    """
    Create features for modeling tasks.
    Args:
        df (pd.DataFrame): DataFrame which the function will create features for.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    try:
        df['entity_count'] = df['entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['text_length'] = df['processed_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)

        # Check if 'pos_tags' column is present and is a list before processing
        if 'pos_tags' in df.columns:
            df['unique_pos_count'] = df['pos_tags'].apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
        else:
            df['unique_pos_count'] = 0

        df['sentence_complexity'] = df['dep_parse'].apply(lambda x: len(set(x)) / len(x) if isinstance(x, list) and x else 0)
        df['vocab_diversity'] = df['processed_text'].apply(lambda x: len(set(x.split())) / len(x.split()) if isinstance(x, str) and x.split() else 0)

        # Check if 'entities' column is present and is a list before applying count_entity_type
        df['product_entity_count'] = df['entities'].apply(lambda x: count_entity_type(x, 'PRODUCT') if isinstance(x, list) else 0)

        logging.info("Feature Engineering function applied successfully")
        return df
    except Exception as e:
        logging.error(f'Error in feature engineering: {e}')
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
