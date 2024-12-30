import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import pandas as pd
import pickle

download('punkt')  # For tokenization
download('stopwords')  # For stopwords

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # lowercase 
    text = text.lower()

    tokens = word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # lemmatization (using SpaCy)
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    # return to string 
    return " ".join(lemmatized_tokens)



def save_pickle(obj, file_path):
    """
    Saves an object to a pickle file.

    Parameters:
    obj: The object to be saved (e.g., model, list, dictionary).
    file_path (str): The path where the pickle file will be saved.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the pickle file: {e}")

def load_pickle(file_path):
    # Open the file in read-binary mode and load the content
    with open(file_path, 'rb') as file:
        file_ = pickle.load(file)
        return file_