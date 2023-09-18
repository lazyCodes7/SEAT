from nltk.corpus import stopwords
import re
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
class BaseLoader(ABC):
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = get_tokenizer('basic_english')

    def stringprocess(self, text):
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

    def tokenprocess(self, text):
        text_tokens = self.tokenizer(text)
        # Filter tokens based on their frequency
        filtered_tokens = [token for token in text_tokens if token not in self.stop_words]
        return filtered_tokens

    def collate_fn(self, batch):
        # Sort the batch in descending order of input sequence lengths
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        # Separate inputs and labels
        inputs, labels = zip(*batch)

        # Get the lengths of each input sequence
        input_lengths = [len(x) for x in inputs]

        # Pad the input sequences to the length of the longest sequence
        padded_inputs = pad_sequence(inputs, batch_first=True)

        return padded_inputs, torch.tensor(labels, dtype=torch.float32), input_lengths

    @abstractmethod
    def load_dataset(self):
        pass
