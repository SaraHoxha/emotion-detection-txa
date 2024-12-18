import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import numpy as np
import contractions
import re
from keras.preprocessing.sequence import pad_sequences
import contractions
from keras.preprocessing.text import Tokenizer
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
    Tokenizes and pads training data using Keras' Tokenizer.
    
    Args:
        train_data: A list or pandas Series of text data for training.
        num_words: Maximum number of words to keep in the vocabulary.
        max_length: Maximum sequence length after padding/truncating.

    Returns:
        padded_sequences: Numpy array of tokenized and padded sequences.
        tokenizer: The Keras Tokenizer fitted on the training data.
        vocab_size: Size of the vocabulary (num_words + OOV + padding token).
    """
    UNK_TOKEN = "<unk>" 
    # Tokenize data
    tokenized_data = [word_tokenize(sentence.lower()) for sentence in train_data['text'].tolist()]
    
    # Calculate lengths of tokenized sequences
    tokenized_lengths = [len(tokens) for tokens in tokenized_data]

    # Set max_length to the 95th percentile of sequence lengths
    max_length = int(np.percentile(tokenized_lengths, 95))
    
    # Initialize and fit tokenizer on training data
    tokenizer = Tokenizer(num_words=num_words, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts(train_data['text'])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(train_data['text'])
    
    # Pad sequences to the specified max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token

    return padded_sequences, tokenizer, vocab_size