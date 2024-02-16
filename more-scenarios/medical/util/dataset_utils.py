import os
import torch
import cv2
from util.classes import MASK


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


def get_patient_ids_from_directory(directory):
    """
    read all patients ids from directory

    Arguements:
    directory -- path

    return: list of ids
    """
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]


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
    get patient ids, frames, slices (if available) from split file

    Arguments:
    split -- split file path

    return: list of tuples
    """
    with open(split, 'r') as f:
        str_ids_frames = f.read().splitlines()
    
    return [x.split('-') for x in str_ids_frames]


def generate_split(output_file, num_patients=7, num_slices=8, num_frames=2, start_idx = 0):
    patient_ids_frames = get_patient_ids_frames('splits/ukbb/all.txt')

    with open(output_file, 'w') as file:
        for i in range(num_patients * num_frames):
            for slice_idx in range(num_slices):
                # example slices 2-9
                patient_id, frame = patient_ids_frames[i+start_idx]
                file.write(f'{patient_id}-{frame}-{slice_idx+2}\n')