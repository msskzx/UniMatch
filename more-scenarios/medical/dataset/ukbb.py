import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np
from util.dataset_utils import get_patient_ids_frames, transform, swap_classes, get_patient_ids_frames_from_csv
import random
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
from copy import deepcopy
from PIL import Image


class UKBBDataset(Dataset):
    def __init__(self, name, root_dir, mode, crop_size, split):
        """
        Arguments:
        str -- database name
        root_dir -- database path
        split -- split path
        """
        self.name = name
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.patient_ids_frames = get_patient_ids_frames_from_csv(split, mode)

    def __getitem__(self, item):
        """
        get data item with given item index

        Arugments:
        item -- item index

        return:
        patient_id_frame -- tuple (id, frame) could contain slice as well
        img -- img with shape (1, 10, 204, 208)
        mask -- same shape as img
        """
        return self.get_patient_images(self.patient_ids_frames[item])


    def __len__(self):
        """
        get length of dataset

        return: length (int)
        """
        return len(self.patient_ids_frames)


    def get_patient_images(self, patient_id_frame):
        """
        helper to get data element
        
        Arguments:
        patient_id_frame -- tuple (id, frame) could contain slice as well

        return:
        patient_id -- id
        frame -- ED or ES
        img -- img with shape (1, 10, 204, 208)
        mask -- same shape as img
        """
        patient_id = str(patient_id_frame[0])
        frame = str(patient_id_frame[1])

        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, f"{frame}.nii.gz")
        in_img = nib.load(image_path).get_fdata()[:]
        in_img = np.transpose(in_img, (2, 0, 1))
        # Rotate the img using OpenCV to match the training data preprocessing
        img = transform(in_img, normalize=True)

        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, f"seg_{frame}.nii.gz")
        in_mask = nib.load(massk_path).get_fdata()[:]
        in_mask = np.transpose(in_mask, (2, 0, 1))
        # Rotate the mask using OpenCV to match the training data preprocessing
        mask = transform(in_mask, normalize=False)
        # swap classes to match acdc
        mask = swap_classes(mask)
        
        if self.mode in ['val', 'test']:
            return {
                'patient_id': patient_id,
                'frame': frame,
                'img': img,
                'mask': mask,
            }
        
        # TODO optimize this step because you load the whole 3D img and preprocess then return one slice
        slice_idx = int(patient_id_frame[2])
        img = img[slice_idx]
        mask = mask[slice_idx]
    
        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        
        x, y = img.shape
        img = zoom(img, (self.crop_size / x, self.crop_size / y), order=0)
        mask = zoom(mask, (self.crop_size / x, self.crop_size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.crop_size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.crop_size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2
