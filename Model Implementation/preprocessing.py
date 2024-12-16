import torch
import re
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download('punkt_tab')
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
import contractions

def preprocess(text):
    """
    Preprocesses text by converting to lowercase and removing non-alphanumeric characters.
    """
    text = contractions.fix(text)
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def tokenize_and_pad(train_data, train_vocab=None):
    """
    Tokenizes and pads training data.
    Args:
        train_data: A DataFrame with a 'text' column containing text data.
        train_vocab: A dictionary mapping words to indices for tokenization (optional for training).
    
    Returns:
        padded_sequences: Tensor of tokenized and padded sequences.
        vocab: Vocabulary dictionary mapping words to indices.
        vocab_size: Size of the vocabulary.
    """
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    
    # Tokenize data
    tokenized_data = [word_tokenize(sentence.lower()) for sentence in train_data['text'].tolist()]
    
    # Calculate lengths of tokenized sequences
    tokenized_lengths = [len(tokens) for tokens in tokenized_data]

    # Set max_length to the 95th percentile of sequence lengths
    max_length = int(np.percentile(tokenized_lengths, 95))
    
    # If no vocabulary is provided, create one from the training data
    if train_vocab is None:
        token_counts = Counter()
        for tokens in tokenized_data:
            token_counts.update(tokens)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(token_counts.items())}
        vocab[UNK_TOKEN] = 0
        vocab[PAD_TOKEN] = 1
    else:
        # Use provided vocabulary for tokenization
        vocab = train_vocab
    
    # Numericalize tokens
    numericalized = [
        [vocab.get(word, vocab[UNK_TOKEN]) for word in tokens]
        for tokens in tokenized_data
    ]
    numericalized = [torch.tensor(seq) for seq in numericalized]

    # Pad sequences
    padded_sequences = pad_sequence(numericalized, batch_first=True, padding_value=vocab[PAD_TOKEN])

    # Truncate or pad to the specified max_length
    if padded_sequences.size(1) < max_length:
        padding = torch.full((padded_sequences.size(0), max_length - padded_sequences.size(1)), vocab[PAD_TOKEN])
        padded_sequences = torch.cat([padded_sequences, padding], dim=1)
    else:
        padded_sequences = padded_sequences[:, :max_length]

    vocab_size = len(vocab)
    
    return padded_sequences, vocab, vocab_size
