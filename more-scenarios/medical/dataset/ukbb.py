import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage.interpolation import zoom
import cv2

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

    def get_patient_images(self, patient_id):
        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, "sa_ES.nii.gz")
        image = nib.load(image_path).get_fdata()
        #slice_id = image.shape[2] // 2
        slice_id = 3
        image = image[:, :, slice_id]
        
        # Rotate the image using OpenCV
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Convert the rotated image to uint8 (necessary for correct conversion to tensor)
        image_uint8 = image.astype(np.uint8)
        # Convert the uint8 image to a PyTorch tensor
        image_tensor = torch.tensor(image_uint8).float().unsqueeze(0)
        # Normalize the pixel values to the range [0, 1]
        # TODO use z-score normalization
        image = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))

        # Save or further process the rotated i
        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, "seg_sa_ES.nii.gz")
        mask = nib.load(massk_path).get_fdata()[:,:, slice_id]
        # Rotate the mask 90 degrees counterclockwise using OpenCV
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = torch.tensor(mask).long().unsqueeze(0)

        # change classes
        # RV should be white instead of dark gray
        # LV should be dark gray instead of white
        mask_1 = mask == 1
        mask_3 = mask == 3

        mask[mask_1] = 3
        mask[mask_3] = 1

        if self.mode == 'train':
            x, y = image.shape
            image = zoom(image, (self.crop_size / x, self.crop_size / y), order=0)
            mask = zoom(mask, (self.crop_size / x, self.crop_size / y), order=0)
        
        patient = {
            'patient_id': patient_id,
            'slice_id': slice_id,
            'image': image,
            'mask': mask,
        }

        return patient
