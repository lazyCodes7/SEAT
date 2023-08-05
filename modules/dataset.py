import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
import nltk
import re
nltk.download('stopwords')
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
class IMDBDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_length=500):
        self.data = df['review']
        self.targets = df['sentiment']
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vocab_dict = {token: index for index, token in enumerate(vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data and target for the given index
        data_point = self.data.iloc[index]
        data_point = stringprocess(data_point)
        word_tokens = tokenprocess(data_point)
        target = self.targets.iloc[index]

        # Truncate the data point to the specified max length
        truncated_data = word_tokens[:self.max_length]
        data_ids = [self.vocab_dict[word] for word in truncated_data if self.vocab_dict.get(word) is not None]

        return torch.tensor(data_ids), target