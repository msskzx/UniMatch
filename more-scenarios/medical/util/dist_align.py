"""
FixMatch, ReMixMatch PyTorch adaptation for image segmentation instead of classification
"""

import torch
import torch.nn.functional as F

def renorm(v):
    return v / torch.sum(v, dim=-1, keepdim=True)

class PMovingAverage:
    """
    Distribution Moving Average for Image Segmentation
    """
    def __init__(self, name, nclass, buf_size):
        self.name = name
        self.nclass = nclass
        self.buf_size = buf_size
        # Initialize buffer with uniform distribution
        self.ma = torch.ones(buf_size, nclass) / nclass

    def __call__(self):
        v = torch.mean(self.ma, dim=0)
        return v / torch.sum(v)

    def update(self, entry):
        entry_one_hot = F.one_hot(entry, num_classes=self.nclass).float()
        entry_summed = torch.sum(entry_one_hot, dim=(0, 1, 2))
        entry_mean = entry_summed / torch.sum(entry_summed)
        # Update the buffer by removing the oldest entry and adding the new one
        self.ma = torch.cat([self.ma[1:], entry_mean.unsqueeze(0)], dim=0)


class PData:
    """
    Distribution Management for Image Segmentation
    """
    def __init__(self, dataset):
        self.has_update = False
        
        if dataset.p_unlabeled is not None:
            self.p_data = torch.tensor(dataset.p_unlabeled, dtype=torch.float32)
        elif dataset.p_labeled is not None:
            self.p_data = torch.tensor(dataset.p_labeled, dtype=torch.float32)
        else:
            # Initialize with uniform distribution
            self.p_data = renorm(torch.ones(dataset.nclass, dtype=torch.float32))
            self.has_update = True

    def __call__(self):
        return self.p_data / torch.sum(self.p_data)

    def update(self, entry, decay=0.999):
        entry_one_hot = F.one_hot(entry, num_classes=self.p_data.shape[0]).float()
        entry_summed = torch.sum(entry_one_hot, dim=(0, 1, 2))
        entry_mean = entry_summed / torch.sum(entry_summed)
        self.p_data = self.p_data * decay + entry_mean * (1 - decay)
