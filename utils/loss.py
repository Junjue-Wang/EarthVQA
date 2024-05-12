import torch
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def ce_loss(pred, batch_label, alphas=None, alpha_c=1.0):
    loss_dict = {}
    pred_len = pred.size(1)
    
    label_clipped = batch_label[:, :pred_len] # [pred_len, batch_size]
    pred_clipped = pred[:, :pred_len, :].permute(0, 2, 1) # [pred_len, vocab_size ,batch_size]
    loss = F.cross_entropy(pred_clipped, label_clipped) # 0 refers to pad

    # cal performance
    valid_mask = (label_clipped != 0)
    pred_res = pred_clipped.softmax(dim=1).argmax(dim=1)
    loss_dict['train_acc'] = (pred_res[valid_mask] == label_clipped[valid_mask]).sum() / valid_mask.sum()
    if alphas is not None:
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    loss_dict['ce_loss'] = loss

    return loss_dict
