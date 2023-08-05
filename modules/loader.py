from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import re
import string
from string import digits
from collections import Counter
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
import torch
import matplotlib.pyplot as plt
from .dataset import IMDBDataset
stop_words = set(stopwords.words('english'))
# Define the tokenizer
tokenizer = get_tokenizer('basic_english')
def stringprocess(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"http\S+","URL", text)
    text = re.sub(r"@", "", text)
    text = re.sub(r"[^A-Za-z0-9()!?\'\`\"]", " ", text)
    text = re.sub("\s{2,}", " ", text)
    text = text.strip(' ')
    text = text.lower()

    return text

def tokenprocess(text):
    text_tokens = tokenizer(text)
    # Filter tokens based on their frequency
    filtered_tokens = [token for token in text_tokens if token not in stop_words]
    return filtered_tokens

def collate_fn(batch):
    # Sort the batch in descending order of input sequence lengths
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Get the lengths of each input sequence
    input_lengths = [len(x) for x in inputs]

    # Pad the input sequences to the length of the longest sequence
    padded_inputs = pad_sequence(inputs, batch_first=True)

    return padded_inputs, torch.tensor(labels, dtype=torch.float32), input_lengths

def load_dataset(data_folder = 'dataset/IMDB Dataset.csv', batch_size = 32):
    df = pd.read_csv(data_folder)
    # Create a mapping dictionary
    label_mapping = {'positive': 1, 'negative': 0}

    # Convert labels using the mapping dictionary
    df['sentiment'] = df['sentiment'].map(label_mapping)

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=1234)


    # Example usage
    print("Train set size:", len(train_df))
    print("Test set size:", len(test_df))
    stop_words = set(stopwords.words('english'))


    X = df["review"]

    X = X.apply(stringprocess)
    word_tokens = list(X.apply(tokenprocess))

    word_tokens_flat = [item for sublist in word_tokens for item in sublist]

    # Collect unique tokens from the dataset
    vocab = set()
    for data_point in word_tokens:
        vocab.update(data_point)

    # Step 1: Determine word frequencies
    word_frequency = {}
    for word in word_tokens_flat:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1

    # Step 2: Define threshold frequency
    threshold = 4

    # Step 3: Create filtered list
    vocab = [word for word in vocab if word_frequency[word] >= threshold]

    # Convert the set of unique tokens to a list
    vocab = list(vocab)
    vocab = ['<pad>'] + vocab

    print(len(vocab))

    # Example usage: Print the vocabulary
    print(vocab[:50])

    # Count the number of tokens per data point
    token_counts = []
    for data_point in word_tokens:
        token_count = len(data_point)
        token_counts.append(token_count)


    # # Load GloVe embeddings
    # # Load a subset of GloVe embeddings
    glove = GloVe(name='6B', dim=300)

    # # Create a matrix to store GloVe embeddings
    embedding_matrix = np.zeros((len(vocab), 300))


    # # Fill the embedding matrix
    for i, token in enumerate(vocab):
        embedding_matrix[i] = glove[token]

    print("---------Saved pretrained embedding---------")
    np.save('embeddings.npy', embedding_matrix)


    train_dataset = IMDBDataset(train_df, tokenizer, vocab)
    test_dataset = IMDBDataset(test_df, tokenizer, vocab)


    # Create a DataLoader for batching and shuffling
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return train_dataloader, test_dataloader