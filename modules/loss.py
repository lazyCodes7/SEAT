import torch
import torch.nn as nn
from .attention import TanhAttention
def pgd_attack(model, text, label, seq_length, alpha, loss_fn, step_size, eps, batch_size, device):
    """
        Projected Gradient Descent to find optimal delta to perturb the attention

        Inputs:
            model -> nn.Module
            text, label, seq_length - > Inputs for the model
            alpha -> Used to control the perturbation added to the attention
            loss_fn -> For measuring impact of adding perturbation to SEAT
            step_size -> No of iters of PGD
            eps -> For controlling the range of delta
            batch_size -> Batch size of a single batch
            device -> CPU/CUDA

        Outputs:
            The optimal delta that can be later used to measure the stability of the proposed attention

    """
    max_seq = max(seq_length)
    delta = torch.zeros_like(model.attention.attn2.weight).to(device)
    init_weight = model.seat_attention.attn2.weight.clone().to(device)
    for i in range(step_size):
        # Apply attention + delta on a specific layer
        output = model(text, seq_length, use_seat = True)
        model.seat_attention.attn2.weight = nn.Parameter(init_weight + delta)
        output_d = model(text, seq_length, use_seat = True)
        loss = loss_fn(output, output_d)
        loss.backward()
        delta.data = (delta + alpha*model.seat_attention.attn2.weight.grad.data.sign()).clamp(-eps, eps)
        model.zero_grad()
        model.seat_attention.attn2.weight = nn.Parameter(init_weight)

    model.seat_attention.attn2.weight = nn.Parameter(init_weight)
    return delta

def stability_loss(model, text, seq_length, delta, loss_fn, device):
    """
    Measures the stability of attention under perturbation
        Inputs:
            model -> nn.Module
            text, label, seq_length - > Inputs for the model
            delta -> the perturbation to be added to SEAT
            loss_fn -> For measuring impact of adding perturbation to SEAT
            device -> CPU/CUDA

        Outputs:
            Loss or stability of SEAT under perturbation

    """
    init_weight = model.seat_attention.attn2.weight.clone().to(device)
    # Apply seat attention on a specific layer
    output = model(text, seq_length, use_seat = True)
    # Apply seat attention + delta on a specific layer
    model.seat_attention.attn2.weight = nn.Parameter(init_weight + delta)
    output_d = model(text, seq_length, use_seat = True)
    model.seat_attention.attn2.weight = nn.Parameter(init_weight)
    return loss_fn(output, output_d)

def similarity_loss(model, text, seq_length, loss_fn):
    """
    Measures the similarity of prediction
        Inputs:
            model -> nn.Module
            text, seq_length - > Inputs for the model
            loss_fn -> For measuring similarity between SEAT and the vanilla attention
        
        Outputs:
            Loss or similarity between seat and vanilla attention

    
    """
    output = model(text, seq_length)
    # Apply SEAT on a specific layer
    output_s = model(text, seq_length, use_seat = True)
    return loss_fn(output, output_s)

def topk_loss(model, text, seq_length, k=7):
    """
    Measures the similarity of prediction
        Inputs:
            model -> nn.Module
            text, seq_length - > Inputs for the model
        
        Outputs:
            Top-k overlap between SEAT and vanilla attention
    """
    criterion = nn.L1Loss()
    attn_base, _ = model.atten_forward(text, seq_length)
    top_k_attention_vectors, _ = torch.topk(attn_base, k = k)
    attn_seat, _ = model.atten_forward(text, seq_length, use_seat = True)
    top_k_seat_vectors, _ = torch.topk(attn_seat, k = k)
    return_loss = (1/(2*k))*(criterion(top_k_seat_vectors, top_k_attention_vectors) + criterion(top_k_attention_vectors, top_k_seat_vectors))
    return return_loss/text.shape[0]
