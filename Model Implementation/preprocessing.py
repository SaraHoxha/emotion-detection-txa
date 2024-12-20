import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import tensorflow
import numpy as np
import contractions
import re
from tensorflow.keras.utils import pad_sequences 
from tensorflow.keras.layers import TextVectorization
import contractions
#from keras_preprocessing.text import Tokenizer
def preprocess(text):
    """
    Preprocesses text by converting to lowercase and removing non-alphanumeric characters.
    """
    text = contractions.fix(text)
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def tokenize_and_pad(train_data, max_length=100, num_words = 100000):
    """
    Tokenizes and pads training data using TensorFlow.Keras Tokenizer.
    
    Args:
        train_data: A list or pandas Series of text data for TextVectorization.
        num_words: Maximum number of words to keep in the vocabulary.
        max_length: Maximum sequence length after padding/truncating.

    Returns:
        padded_sequences: Numpy array of tokenized and padded sequences.
        tokenizer: The TextVectorization fitted on the training data.
        vocab_size: Size of the vocabulary.
    """
    # Tokenize data
    tokenized_data = [word_tokenize(sentence.lower()) for sentence in train_data['text'].tolist()]
    
    # Calculate lengths of tokenized sequences
    tokenized_lengths = [len(tokens) for tokens in tokenized_data]

    # Set max_length to the 95th percentile of sequence lengths
    max_length = int(np.percentile(tokenized_lengths, 95))
    
    # Initialize and fit tokenizer on training data
    text_vectorizer = TextVectorization(
        max_tokens=num_words,
        output_mode='int',
        output_sequence_length=max_length
    )   
    text_vectorizer.adapt(train_data['text'])

    # Convert text to sequences
    vectorized_sequences = text_vectorizer(train_data['text'])
    
    # Pad sequences to the specified max_length
    padded_sequences = pad_sequences(vectorized_sequences, maxlen=max_length, padding="post", truncating="post")
    
    vocab_size = len(text_vectorizer.get_vocabulary())

    return padded_sequences, text_vectorizer, vocab_size