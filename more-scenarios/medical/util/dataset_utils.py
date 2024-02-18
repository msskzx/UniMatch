import os
import torch
import cv2
from util.classes import MASK
import nibabel as nib


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


def get_patient_ids_from_directory(directory, num_slices=8):
    """
    read all patients ids from directory

    Arguements:
    directory -- path

    return: list of ids
    """
    folders = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            files = os.listdir(os.path.join(directory, folder))
            # has imgs, masks
            if ('sa_ED.nii.gz' in files
                and 'seg_sa_ED.nii.gz' in files
                and 'sa_ES.nii.gz' in files
                and 'seg_sa_ES.nii.gz' in files):
                image_path = os.path.join(directory, folder, 'sa_ED.nii.gz')
                in_img = nib.load(image_path).get_fdata()[:]
                if in_img.shape[2] >= num_slices:
                    folders.append(folder)
    return folders


def save_patient_ids(patient_ids, output_file):
    """
    save the list of ids in split text file

    Arguements:
    patient_ids -- list of ids
    output_file -- path
    """
    with open(output_file, 'w') as file:
        for patient_id in patient_ids:
            file.write(f'{patient_id}-sa_ED\n{patient_id}-sa_ES\n')


def get_patient_ids_frames(split):
    """
    read patient ids, frames, slices (if available) from split file

    Arguments:
    split -- split file path

    return: list of tuples
    """
    with open(split, 'r') as f:
        str_ids_frames = f.read().splitlines()
    
    return [x.split('-') for x in str_ids_frames]


def generate_split(output_file, num_patients=7, num_slices=8, num_frames=2, start_idx = 0, mode='train'):
    patient_ids_frames = get_patient_ids_frames('splits/ukbb/all.txt')

    with open(output_file, 'w') as file:
        for i in range(num_patients * num_frames):
            patient_id, frame = patient_ids_frames[i+start_idx]
            
            if mode == 'train':
                for slice_idx in range(num_slices):
                    file.write(f'{patient_id}-{frame}-{slice_idx}\n')
            else:
                file.write(f'{patient_id}-{frame}\n')

