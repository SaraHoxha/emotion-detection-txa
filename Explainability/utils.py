import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import pandas as pd

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

