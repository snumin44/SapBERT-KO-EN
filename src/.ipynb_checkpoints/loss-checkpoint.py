import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSimilarityLoss(nn.Module):

    def __init__(self, threshold, scale_pos, scale_neg, margin, hard_pair_mining, device):
        
        super(MultiSimilarityLoss, self).__init__()
        self.threshold = threshold       
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.margin = margin
        self.hard_pair_mining = hard_pair_mining
        self.device = device
        
    def forward(self, sim_matrix, labels):
        
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels,labels.t()).float().to(self.device)
        neg_mask = torch.logical_not(pos_mask).float().to(self.device)
                
        pos_matrix = sim_matrix.masked_fill(pos_mask==0.0, 0.0)
        neg_matrix = sim_matrix.masked_fill(neg_mask==0.0, 0.0)
                
        if self.hard_pair_mining:
            
            # pos_pair - self.margin < max(neg_pair)
            neg_max = torch.max(neg_matrix[neg_matrix > 0.0])
            pos_mask = (pos_matrix - self.margin < neg_max).float()
            
            # neg_pair + self.margin > min(pos_pair)
            pos_min = torch.min(pos_matrix[pos_matrix > 0.0])
            neg_mask = (neg_matrix + self.margin > pos_min).float()

        pos_exp = torch.exp(-self.scale_pos * (pos_matrix - self.threshold))
        pos_exp = torch.where(pos_mask > 0.0, pos_exp, torch.zeros_like(pos_exp))

        neg_exp = torch.exp(self.scale_neg * (neg_matrix - self.threshold))
        neg_exp = torch.where(neg_mask > 0.0, neg_exp, torch.zeros_like(neg_exp))

        pos_term = 1 / self.scale_pos * torch.log(1.0 + torch.sum(pos_exp, dim=1) + 1e-8)
        neg_term = 1 / self.scale_neg * torch.log(1.0 + torch.sum(neg_exp, dim=1) + 1e-8)
        loss = torch.mean(pos_term + neg_term)

        return loss