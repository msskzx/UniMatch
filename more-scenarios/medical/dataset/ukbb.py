import os
from torch.utils.data import Dataset
import nibabel as nib
import torch
import numpy as np
import cv2
from util.classes import MASK
from util.dataset_utils import get_patient_ids_frames
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
        self.patient_ids_frames = get_patient_ids_frames(split)


    def __getitem__(self, item):
        """
        get data item with given item index

        Arugments:
        item -- item index

        return:
        patient_id_frame -- tuple (id, frame)
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


    def transform(self, in_img, normalize=False):
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

    
    def swap_classes(self, mask):
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


    def get_patient_images(self, patient_id_frame):
        """
        helper to get data element
        
        Arguments:
        patient_id_frame -- tuple (id, frame)

        return:
        patient_id -- id
        frame -- ED or ES
        img -- img with shape (1, 10, 204, 208)
        mask -- same shape as img
        """
        patient_id, frame = patient_id_frame

        # Load original images for the current patient
        image_path = os.path.join(self.root_dir, patient_id, f"{frame}.nii.gz")
        in_img = nib.load(image_path).get_fdata()[:]
        in_img = np.transpose(in_img, (2, 0, 1))
        # Rotate the img using OpenCV to match the training data preprocessing
        img = self.transform(in_img, normalize=True)

        # Load segmentation images for the current patient
        massk_path = os.path.join(self.root_dir, patient_id, f"seg_{frame}.nii.gz")
        in_mask = nib.load(massk_path).get_fdata()[:]
        in_mask = np.transpose(in_mask, (2, 0, 1))
        # Rotate the mask using OpenCV to match the training data preprocessing
        mask = self.transform(in_mask, normalize=False)
        # swap classes to match acdc
        mask = self.swap_classes(mask)
        
        if self.mode in ['val', 'test']:
            return {
                'patient_id': patient_id,
                'frame': frame,
                'img': img,
                'mask': mask,
            }
        
        # TODO should I look for a way to return only one slice
        # by preprocessing the data once and storing the slices
        # or i pass id, frame, slice instead
        # or use all the slices per img
        # should this be during training and testing or only training
        img = img[3]
        mask = mask[3]
    
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
