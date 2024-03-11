import torch
import cv2
from util.classes import MASK
import pandas as pd


def transform(in_img, normalize=False):
    """
    rotate the images 90 degrees counter clockwise and normalize to range [0, 1]
    
    Arguments:
    in_img -- img or mask to be rotated
    normalize -- normalize if img (not mask)

    return: transformed img
    """
    num_slices, height, width = in_img.shape
    img = torch.empty((num_slices, width, height), dtype=torch.float)
    for slice_idx in range(in_img.shape[0]):
        img[slice_idx] = torch.from_numpy(cv2.rotate(in_img[slice_idx], cv2.ROTATE_90_COUNTERCLOCKWISE))

        # Normalize the pixel values to the range [0, 1]
        if normalize:
            img[slice_idx] = (img[slice_idx] - torch.min(img[slice_idx])) / (torch.max(img[slice_idx]) - torch.min(img[slice_idx]))
    
    return img

    
def swap_classes(mask):
    """
    swap rv and lv classes in mask to match acdc dataset
    
    Arguments:
    mask -- mask to be converted
    """
    mask_lv = mask == MASK['rv']
    mask_rv = mask == MASK['lv']
    mask[mask_lv] = MASK['lv']
    mask[mask_rv] = MASK['rv']
    return mask

def get_patient_ids_frames(split, mode):
    df = pd.read_csv(split)
    res = []
    for idx, row in df.iterrows():
        if mode in ['train', 'train_l', 'train_u']:
            res.append((row['eid'], row['frame'], row['slice_idx']))
        else:
            res.append((row['eid'], row['frame']))

    return res