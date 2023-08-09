import sys
from typing import Any
sys.path.append('../')
from modules.loader import load_dataset
from modules.lstm import AttentionLSTM
import torch.nn as nn
import torch
from seat import SEAT
import lightning as pl

'''
Function for initializing weights for a particular module
'''
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])



if __name__ == "__main__":
    train_loader, test_loader = load_dataset('IMDB Dataset.csv')

    embed_size, num_hiddens, num_layers, device = 300, 128, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = AttentionLSTM(
            vocab_size = len(train_loader.dataset.vocab_dict),
            emb_dim = embed_size,
            hidden_size = num_hiddens,
            num_classes = 1,
            dropout = 0.4,
    )
    net.apply(init_weights)
    net.load_state_dict(torch.load('imdb_bilstm_tanh_attention_glove_300d.pt', map_location = device))
    net = net.to(device)
    seat = SEAT(num_hiddens, net, device)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(seat, train_dataloaders=train_loader)
    trainer.validate(seat, dataloaders=test_loader)
