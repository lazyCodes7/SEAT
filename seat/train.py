import sys
from typing import Any
sys.path.append('../')
from modules.loader import load_dataset
from modules.lstm import AttentionLSTM
import torch.nn as nn
import torch
from seat import SEAT
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--device', type=str, required=True, default = 'cpu')
    parser.add_argument('-m', '--model_path', type=str, required=False, default='imdb_bilstm_tanh_attention_glove_300d.pt')
    parser.add_argument('-nh', '--hiddens', type=int, required=False, default=128)
    parser.add_argument('-es', '--embed_size', type=int, required=False, default=300)
    parser.add_argument('-l', '--layers', type=int, required=False, default=1)
    parser.add_argument('-e', '--epoch', type=int, required=False, default=1)

    args = parser.parse_args()

    train_loader, test_loader = load_dataset('IMDB Dataset.csv')

    net = AttentionLSTM(
            vocab_size = len(train_loader.dataset.vocab_dict),
            emb_dim = args.embed_size,
            hidden_size = args.hiddens,
            num_classes = 1,
            dropout = 0.4,
    )
    net.apply(init_weights)
    net.load_state_dict(torch.load('imdb_bilstm_tanh_attention_glove_300d.pt', map_location = args.device), strict = False)
    net = net.to(args.device)
    seat = SEAT(args.hiddens, net, args.device)
    trainer = pl.Trainer(max_epochs=args.epoch)
    trainer.fit(seat, train_dataloaders=train_loader)
    trainer.validate(seat, dataloaders=test_loader)
