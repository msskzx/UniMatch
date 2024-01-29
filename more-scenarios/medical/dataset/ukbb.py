import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np

class UKBBDataset(Dataset):
    def __init__(self, name, root_dir, mode, patient_ids, crop_size=None, id_path=None, nsample=None, transform=None):
        self.name = name
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.patient_ids = patient_ids

    def __getitem__(self, item):
        patient_id = self.patient_ids[item]
        return self.get_patient_images(patient_id)

    def __len__(self):
        return len(self.patient_ids)

    def get_patient_images(self, patient_id):
        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, "sa_ES.nii.gz")
        image = nib.load(image_path).get_fdata()

        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, "seg_sa_ED.nii.gz")
        mask = nib.load(massk_path).get_fdata()

        patient = {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).long(),
        }

        return patient
