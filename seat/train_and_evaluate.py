import sys
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

'''
Objective function to train the proposed SEAT Attention
'''
def seat_objective_fn(text, label, seq_length, loss_fn, step_size, eps, alpha, lambda1, lambda2, iters = 100):
    #delta_o = torch.randn(seat_w.shape).to(device)
    delta = pgd_attack(net, text, label, seq_length, alpha, seat_w, loss_fn, step_size, eps, 32)
    st_loss = stability_loss(net, text, seq_length, seat_w, delta, loss_fn)
    sim_loss = similarity_loss(net, text, seq_length, seat_w, loss_fn)
    tk_loss = topk_loss(net, text, seq_length, seat_w, k=7)
    loss = st_loss + lambda1*sim_loss + lambda2*tk_loss
    return loss
def evaluate(net, seat_w):
  jsd = JSD()
  with tqdm(test_loader, unit="batch") as tepoch:
    jsd_score_perturb = 0
    jsd_score_seed = 0
    tvd_score_perturb = 0
    tvd_score_seed = 0
    tepoch.set_description(f"SEAT Evaluation")
    with torch.no_grad():
      for idx, (text, label, seq_length) in enumerate(tepoch):
        text = text.to(device)
        label = label.to(device)
        attn_vanilla, output_vanilla = net.atten_forward(text, seq_length)
        init_attention = net.attention
        net.attention = seat_w
        attn_perturbed, output_perturbed = net.perturb(text, seq_length, device = device)
        attn_seat, output_seat = net.atten_forward(text, seq_length)

        jsd_score_perturb += jsd(attn_seat, attn_perturbed).item()
        jsd_score_seed += jsd(attn_seat, attn_vanilla).item()
        tvd_score_perturb += total_variation_distance_from_logits(output_seat, output_perturbed).item()
        tvd_score_seed += total_variation_distance_from_logits(output_seat, output_vanilla).item()
        net.attention = init_attention


        #print(seat_w.state_dict())

        #print(seat_w.attn1.weight)
        tepoch.set_postfix(
            jsd_on_word_perturb = jsd_score_perturb/(idx+1),
            jsd_base_vs_seat = jsd_score_seed/(idx+1),
            tvd_on_word_perturb = tvd_score_perturb/(idx+1),
            tvd_base_vs_seat = tvd_score_seed/(idx+1)
        )
if __name__ == "__main__":
    alpha = 1e-4
    loss_fn = nn.BCELoss()
    step_size = 20
    eps = 0.1
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
    seat_w = TanhAttention(hidden_size = num_hiddens*2).to(device)
    optimizer = optim.Adam(seat_w.parameters(), lr = 0.01)

    for epoch in range(20):
        lambda1 = 1
        lambda2 = 1000
        running_loss = 0
        with tqdm(test_loader, unit="batch") as tepoch:
            tepoch.set_description(f"SEAT Training")
            for idx, (text, label, seq_length) in enumerate(tepoch):
                optimizer.zero_grad()
                text = text.to(device)
                label = label.to(device)
                #print(seat_w.attn1.weight)
                loss = seat_objective_fn(text, label, seq_length)
                loss.backward()

                running_loss += loss.item()

                optimizer.step()

                #print(seat_w.state_dict())
            
                #print(seat_w.attn1.weight)
                tepoch.set_postfix(loss = running_loss/(idx+1))
        torch.save(seat_w.state_dict(), 'seat_epoch_{}.pth'.format(epoch+1))
        evaluate(net, seat_w)