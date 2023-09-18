from .BaseLoader import BaseLoader
from ..dataset import IMDBDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

class IMDBLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def load_dataset(self, data_folder = 'dataset/IMDB Dataset.csv', batch_size = 32, pretrained = True):
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


        X = df["review"]

        X = X.apply(self.stringprocess)
        word_tokens = list(X.apply(self.tokenprocess))

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

        if(pretrained == False):
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


        train_dataset = IMDBDataset(train_df, self.tokenizer, vocab)
        test_dataset = IMDBDataset(test_df, self.tokenizer, vocab)


        # Create a DataLoader for batching and shuffling
        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=False)
        return train_dataloader, test_dataloader