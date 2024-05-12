# Date: 2024.05.12
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def som(loss, ratio):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss

    return torch.sum(top_loss[loss_mask]) / (loss_mask.sum() + 1e-6)



class NumericalLoss(nn.Module):
    def __init__(self, uper_numer_idx, alpha=1.0, keep_ratio=1.0, gamma=0.5, min_keep=1, scale_factor=0.5):
        super(NumericalLoss, self).__init__()
        self.uper_numer_idx = uper_numer_idx
        self.alpha = alpha
        self.gamma = gamma
        self.min_keep = min_keep
        self.keep_ratio = keep_ratio
        self.scale_factor = scale_factor
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        '''

        Args:
            inputs: B x num_class
            targets: B x 1

        Returns:

        '''
        celoss_v = self.ce_loss(inputs, targets.long())
        
        pred_probs = inputs.softmax(dim=1)
        b, c = pred_probs.shape
        one_hot = F.one_hot(targets, num_classes=c)
        masked = targets < self.uper_numer_idx
        numer_losses = torch.tensor([0.], requires_grad=True).to(device)
        cls_losses = torch.tensor([0.], requires_grad=True).to(device)
        loss_dict = dict()
        
        
        if masked.float().sum() > 0:
            numer_gt, numer_probs, numer_onehot = targets[masked], pred_probs[masked], one_hot[masked]
            weights = self.alpha * (numer_probs.argmax(dim=1) - numer_gt).abs() ** self.gamma
            numer_losses = -(1 + weights) * torch.log((numer_probs * numer_onehot).sum(dim=1))
        
        if (~masked).float().sum() > 0:
            
            cls_one_hot, cls_probs, cls_gt = one_hot[~masked], pred_probs[~masked], targets[~masked]
            cls_losses = - torch.log((cls_probs * cls_one_hot).sum(dim=1))
            pre_cls_num = len(cls_losses)
            # SOM 
            num_inst = cls_losses.numel()
            if int(self.keep_ratio * num_inst) >= self.min_keep:
                num_hns = int(self.keep_ratio * num_inst)
                # 2. select loss
                cls_losses, _ = cls_losses.reshape(-1).topk(num_hns, -1)
                valid_num = torch.tensor([num_hns], requires_grad=False).to(device).float()
            else:
                valid_num = torch.tensor([self.min_keep], requires_grad=False).to(device).float()

        numer_size = numer_losses.numel() if masked.float().sum() > 0 else torch.tensor([0.], requires_grad=False).to(device)
        cls_size = valid_num if (~masked).float().sum() > 0 else torch.tensor([0.], requires_grad=False).to(device)
        
        
        numerical_loss = (cls_losses.sum() + numer_losses.sum()) / (numer_size + cls_size + 1e-9)
        loss_dict['numerical_loss'] = numerical_loss 
        loss_dict['scaler'] =  celoss_v / numerical_loss
        loss_dict['cls_l'] = cls_losses.sum() / (cls_size + 1e-9)
        loss_dict['count_l'] = numer_losses.sum() / (numer_size + 1e-9)
        loss_dict['ohem_num'] = cls_size
        loss_dict['ohem_pre_num'] = (~masked).float().sum()
        loss_dict['number_pre_num'] = masked.float().sum()
        return loss_dict



def cross_entropy_loss(logit, label):
    """
    get cross entropy loss
    Args:
        logit: logit
        label: true label
    Returns:
    """
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logit, label)
    return loss


class InverseWeightCrossEntroyLoss(nn.Module):
    def __init__(self,
                 class_num,
                 ignore_index=255
                 ):
        super(InverseWeightCrossEntroyLoss, self).__init__()
        self.class_num = class_num
        self.ignore_index=ignore_index

    def forward(self, logit, label):
        """
       get inverse cross entropy loss
        Args:
            logit: a tensor, [batch_size, num_class, image_size, image_size]
            label: a tensor, [batch_size, image_size, image_size]
        Returns:
        """
        inverse_weight = self.get_inverse_weight(label)
        cross_entropy = nn.CrossEntropyLoss(weight=inverse_weight,
                                            ignore_index=self.ignore_index).cuda()
        inv_w_loss = cross_entropy(logit, label)
        return inv_w_loss

    def get_inverse_weight(self, label):
        mask = (label >= 0) & (label < self.class_num)
        label = label[mask]
        # reduce dim
        total_num = len(label)
        # get unique label, convert unique label to list
        percentage = torch.bincount(label, minlength=self.class_num) / float(total_num)
        # get inverse
        w_for_each_class = 1 / torch.log(1.02 + percentage)
        # convert to tensor
        return w_for_each_class.float()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, ignore_index=255, reduction=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        alpha = torch.from_numpy(self.alpha).to(y_pred.device).view(1, -1)

        p = torch.softmax(y_pred, dim=1)

        ignore_mask = (y_true == self.ignore_index)

        # one hot encoding
        y_index = torch.clone(y_true).to(y_pred.device)
        y_index[ignore_mask] = 0
        one_hot_y_true = torch.zeros(y_pred.shape, dtype=torch.float).to(y_pred.device)
        one_hot_y_true.scatter_(1, y_index.unsqueeze_(dim=1).long(), torch.ones(one_hot_y_true.shape).to(y_pred.device))

        pt = (p * one_hot_y_true).sum(dim=1)
        modular_factor = (1 - pt).pow(self.gamma)

        cls_balance_factor = (alpha.float() * one_hot_y_true.float()).sum(dim=1)

        modular_factor.mul_(cls_balance_factor)

        losses = F.cross_entropy(y_pred, y_true, ignore_index=self.ignore_index, reduction='none')
        losses.mul_(modular_factor)

        if self.reduction:
            valid_mask = (y_true != self.ignore_index).float()
            mean_loss = losses.sum() / valid_mask.sum()
            return mean_loss
        return losses



class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def _dice_coeff(self, pred, target):
        """
        Args:
            pred: [N, 1] within [0, 1]
            target: [N, 1]
        Returns:
        """

        smooth = self.smooth
        inter = torch.sum(pred * target)
        z = pred.sum() + target.sum() + smooth
        return (2 * inter + smooth) / z

    def forward(self, pred, target):
        return 1. - self._dice_coeff(pred, target)






def ohem_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor,
                       ignore_index: int = -1,
                       thresh: float = 0.7,
                       min_kept: int = 100000):
    # y_pred: [N, C, H, W]
    # y_true: [N, H, W]
    # seg_weight: [N, H, W]
    y_true = y_true.unsqueeze(1)
    with torch.no_grad():
        assert y_pred.shape[2:] == y_true.shape[2:]
        assert y_true.shape[1] == 1
        seg_label = y_true.squeeze(1).long()
        batch_kept = min_kept * seg_label.size(0)
        valid_mask = seg_label != ignore_index
        seg_weight = y_pred.new_zeros(size=seg_label.size())
        valid_seg_weight = seg_weight[valid_mask]

        seg_prob = F.softmax(y_pred, dim=1)

        tmp_seg_label = seg_label.clone().unsqueeze(1)
        tmp_seg_label[tmp_seg_label == ignore_index] = 0
        seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
        sort_prob, sort_indices = seg_prob[valid_mask].sort()

        if sort_prob.numel() > 0:
            min_threshold = sort_prob[min(batch_kept,
                                          sort_prob.numel() - 1)]
        else:
            min_threshold = 0.0
        threshold = max(min_threshold, thresh)
        valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.

    seg_weight[valid_mask] = valid_seg_weight

    losses = F.cross_entropy(y_pred, y_true.squeeze(1), ignore_index=ignore_index, reduction='none')
    losses = losses * seg_weight

    return losses.sum() / seg_weight.sum()

