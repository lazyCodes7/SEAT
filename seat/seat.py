import sys
from typing import Any
sys.path.append('../')
from modules.loader import load_dataset
from modules.lstm import AttentionLSTM
import torch.nn as nn
import torch
from modules.attention import TanhAttention
import torch.optim as optim
from modules.loss import pgd_attack, topk_loss, similarity_loss, stability_loss
from tqdm.auto import tqdm
from modules.metrics import total_variation_distance_from_logits, JSD
import lightning as pl
class SEAT(pl.LightningModule):
    '''
    init(num_hiddens, net, device)

    Inputs:
        num_hiddens -> (int) hidden_size of the LSTM
        net -> (nn.Module) Pretrained LSTM network
        device -> (torch.device) Preferred device for training

    Function: Initializes the SEAT
    '''
    def __init__(self, num_hiddens, net, device) -> None:
        super().__init__()
        self.attention = TanhAttention(hidden_size=num_hiddens*2)
        self.jsd = JSD()
        self.net = net
        self.init_attention = net.attention
        self.alpha = 1e-4
        self.loss_fn = nn.BCELoss()
        self.step_size = 20
        self.eps = 0.1
        self.lambda1 = 1
        self.lambda2 = 1000
        self.compute = device
    
    def forward(self, text, label, seq_length, seat = True):
        if(seat):
            self.net.attention = self.attention
        return self.net(text, label, seq_length)

    def reset(self):
        self.net.attention = self.init_attention

    '''
    Function to train the proposed SEAT Attention

    Inputs -> Batch (text, label, length(text sequence))

    Outputs -> Loss for parameter update
    '''
    def training_step(self, batch):
        text, label, seq_length = batch
        #delta_o = torch.randn(seat_w.shape).to(device)
        delta = pgd_attack(self.net, text, label, seq_length, self.alpha, self.attention, self.loss_fn, self.step_size, self.eps, 32, self.compute)
        st_loss = stability_loss(self.net, text, seq_length, self.attention, delta, self.loss_fn, self.compute)
        sim_loss = similarity_loss(self.net, text, seq_length, self.attention, self.loss_fn)
        tk_loss = topk_loss(self.net, text, seq_length, self.attention, k=7)
        loss = st_loss + self.lambda1*sim_loss + self.lambda2*tk_loss
        return loss
    
    '''
    Function to evaluate the SEAT vs the Vanilla/Base attention

    Inputs -> Batch (text, label, length(text sequence))

    Outputs -> JSD and TVD scores comparing the performance of SEAT with Vanilla attention
    '''
    def validation_step(self, batch):
        with torch.no_grad():
            text, label, seq_length = batch
            attn_vanilla, output_vanilla = self.net.atten_forward(text, seq_length)
            init_attention = self.net.attention
            self.net.attention = self.attention
            attn_perturbed, output_perturbed = self.net.perturb(text, seq_length, device = self.compute)
            attn_seat, output_seat = self.net.atten_forward(text, seq_length)

            jsd_score_perturb = self.jsd(attn_seat, attn_perturbed).item()
            jsd_score_seed = self.jsd(attn_seat, attn_vanilla).item()
            tvd_score_perturb = total_variation_distance_from_logits(output_seat, output_perturbed).item()
            tvd_score_seed = total_variation_distance_from_logits(output_seat, output_vanilla).item()
            self.net.attention = init_attention


            self.log("jsd_on_word_perturb", jsd_score_perturb),
            self.log("jsd_base_vs_seat", jsd_score_seed),
            self.log("tvd_on_word_perturb", tvd_score_perturb),
            self.log("tvd_base_vs_seat", tvd_score_seed)
    
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.attention.parameters(), lr=0.01) 
        return optimizer
            
        


