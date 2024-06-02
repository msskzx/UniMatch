"""
FixMatch, ReMixMatch PyTorch adaptation for image segmentation instead of classification
"""

import torch
import torch.nn.functional as F


class PMovingAverage:
    """
    Model Distribution Moving Average for Image Segmentation
    """
    def __init__(self, name, buf_size, crop_size, nclass=4):
        self.name = name
        self.nclass = nclass
        self.buf_size = buf_size
        self.crop_size = crop_size
        self.ma = torch.ones(buf_size, crop_size, crop_size) / (crop_size * crop_size)

    def __call__(self):
        v = torch.mean(self.ma, dim=0)
        return v / torch.sum(v, keepdim=True)

    def update(self, entry):
        print(entry.shape)
        entry_mean = torch.mean(entry, dim=0)
        entry_mean = entry_mean.unsqueeze(0)
        print(entry_mean.shape)
        print(self.ma.shape)
        self.ma = torch.cat([self.ma[1:], entry_mean], dim=0)


class PData:
    """
    Data Distribution for Image Segmentation
    """
    def __init__(self, dataset, crop_size, nclass=4):
        self.has_update = False
        
        if dataset.p_unlabeled is not None:
            self.p_data = torch.tensor(dataset.p_unlabeled, dtype=torch.float32)
        elif dataset.p_labeled is not None:
            self.p_data = torch.tensor(dataset.p_labeled, dtype=torch.float32)
        else:
            self.p_data = torch.ones(crop_size, crop_size, dtype=torch.float32).cuda()
            self.p_data = self.p_data / torch.sum(self.p_data)
            self.has_update = True

    def __call__(self):
        return self.p_data / torch.sum(self.p_data)

    def update(self, entry, decay=0.999):
        entry_one_hot = F.one_hot(entry, num_classes=self.p_data.shape[0]).float()
        entry_summed = torch.sum(entry_one_hot, dim=(0, 2, 3))
        entry_mean = entry_summed / torch.sum(entry_summed)
        entry_mean = entry_mean.unsqueeze(0)
        self.p_data = self.p_data * decay + entry_mean * (1 - decay)


def guess_label(logits_y):
    p_model_y = sum(F.softmax(x, dim=1) for x in logits_y) / len(logits_y)
    p_model_y = p_model_y.repeat(len(logits_y), 1, 1, 1)
    return p_model_y