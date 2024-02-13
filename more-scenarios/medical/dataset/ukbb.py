import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage.interpolation import zoom
import cv2
from util.classes import CLASSES, MASK


class UKBBDataset(Dataset):
    def __init__(self, name, root_dir, mode, patient_ids, crop_size=None, id_path=None, nsample=None, transform=None):
        self.name = name
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.patient_ids = patient_ids
        # TODO load all slice ids for each patient

    def __getitem__(self, item):
        # TODO load from patient id and slice id
        patient_id = self.patient_ids[item]
        return self.get_patient_images(patient_id)

    def __len__(self):
        return len(self.patient_ids)


    def ccw_rotate(self, in_img, normalize=False):
        num_slices, height, width = in_img.shape
        img = torch.empty((num_slices, width, height), dtype=torch.float)
        for slice_idx in range(in_img.shape[0]):
            img[slice_idx] = torch.from_numpy(cv2.rotate(in_img[slice_idx], cv2.ROTATE_90_COUNTERCLOCKWISE))
            if normalize:
                # Normalize the pixel values to the range [0, 1]
                img[slice_idx] = (img[slice_idx] - torch.min(img[slice_idx])) / (torch.max(img[slice_idx]) - torch.min(img[slice_idx]))
        return img

    
    def swap_classes(self, mask):
        mask_lv = mask == MASK['rv']
        mask_rv = mask == MASK['lv']
        mask[mask_lv] = MASK['lv']
        mask[mask_rv] = MASK['rv']
        return mask

    def get_patient_images(self, patient_id):
        # TODO use ED images as well

        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, "sa_ES.nii.gz")
        in_img = nib.load(image_path).get_fdata()[:]
        in_img = np.transpose(in_img, (2, 0, 1))
        # Rotate the img using OpenCV to match the training data preprocessing
        img = self.ccw_rotate(in_img, normalize=True)

        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, "seg_sa_ES.nii.gz")
        in_mask = nib.load(massk_path).get_fdata()[:]
        in_mask = np.transpose(in_mask, (2, 0, 1))
        # Rotate the mask using OpenCV to match the training data preprocessing
        mask = self.ccw_rotate(in_mask, normalize=False)
        # swap classes to match acdc
        mask = self.swap_classes(mask)
        
        patient = {
            'patient_id': patient_id,
            'image': img,
            'mask': mask,
        }

        return patient
