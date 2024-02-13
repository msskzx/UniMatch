import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np
import cv2
from util.classes import MASK


class UKBBDataset(Dataset):
    def __init__(self, name, root_dir, mode, patient_ids_frames, crop_size=None, id_path=None, nsample=None, transform=None):
        """
        Arguments:
        str -- database name
        root_dir -- database path
        patient_ids_frames -- array of patient id and frame tuples (patient_id, frame)
        """
        self.name = name
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.patient_ids_frames = patient_ids_frames

    def __getitem__(self, item):
        """
        return:
        patient_id_frame -- tuple
        img -- example shape (1, 10, 204, 208) 1 is batch 10 represents slices
        mask -- same shape as img
        """
        patient_id_frame = self.patient_ids_frames[item]
        return self.get_patient_images(patient_id_frame)

    def __len__(self):
        return len(self.patient_ids_frames)


    def ccw_rotate(self, in_img, normalize=False):
        """
        rotate the images 90 degrees counter clockwise
        
        Arguments:
        in_img -- img or mask to be rotated
        normalize -- normalize to range [0, 1] if img
        """
        num_slices, height, width = in_img.shape
        img = torch.empty((num_slices, width, height), dtype=torch.float)
        for slice_idx in range(in_img.shape[0]):
            img[slice_idx] = torch.from_numpy(cv2.rotate(in_img[slice_idx], cv2.ROTATE_90_COUNTERCLOCKWISE))
            if normalize:
                # Normalize the pixel values to the range [0, 1]
                img[slice_idx] = (img[slice_idx] - torch.min(img[slice_idx])) / (torch.max(img[slice_idx]) - torch.min(img[slice_idx]))
        return img

    
    def swap_classes(self, mask):
        """
        swap rv and lv classes in mask to match acdc dataset
        """
        mask_lv = mask == MASK['rv']
        mask_rv = mask == MASK['lv']
        mask[mask_lv] = MASK['lv']
        mask[mask_rv] = MASK['rv']
        return mask

    def get_patient_images(self, patient_id_frame):
        patient_id, frame = patient_id_frame

        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, f"{frame}.nii.gz")
        in_img = nib.load(image_path).get_fdata()[:]
        in_img = np.transpose(in_img, (2, 0, 1))
        # Rotate the img using OpenCV to match the training data preprocessing
        img = self.ccw_rotate(in_img, normalize=True)

        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, f"seg_{frame}.nii.gz")
        in_mask = nib.load(massk_path).get_fdata()[:]
        in_mask = np.transpose(in_mask, (2, 0, 1))
        # Rotate the mask using OpenCV to match the training data preprocessing
        mask = self.ccw_rotate(in_mask, normalize=False)
        # swap classes to match acdc
        mask = self.swap_classes(mask)
        
        return {
            'patient_id': patient_id,
            'frame': frame,
            'img': img,
            'mask': mask,
        }
