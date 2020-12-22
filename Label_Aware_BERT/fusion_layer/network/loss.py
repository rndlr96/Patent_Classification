import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['FocalLoss', 'Normalized_FocalLoss']

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, smooth=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        if self.smooth != 0:
            targets = (1-self.smooth) * targets + self.smooth / inputs.size(1)
            
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_term = (1-pt).pow(self.gamma)
        F_loss = self.alpha * focal_term * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:

class Normalized_FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, smooth=0):
        super(Normalized_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        if self.smooth != 0:
            targets = (1-self.smooth) * targets + self.smooth / inputs.size(1)
            
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_term = (1-pt).pow(self.gamma)
        normalize = 1/focal_term.mean()
        F_loss = normalize * self.alpha * focal_term * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss