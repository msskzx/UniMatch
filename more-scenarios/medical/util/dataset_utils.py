import os
import torch
import cv2
from util.classes import MASK, ETHNNICITY_CODING
import nibabel as nib
import pandas as pd
import random

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
    read all patients ids from directory who have both
    short axis ED, ES img, mask pairs

    Arguements:
    directory -- path

    return: list of ids
    """
    folders = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            files = os.listdir(os.path.join(directory, folder))
            if ('sa_ED.nii.gz' in files
                and 'seg_sa_ED.nii.gz' in files
                and 'sa_ES.nii.gz' in files
                and 'seg_sa_ES.nii.gz' in files):
                folders.append(folder)
    return folders

def save_patient_ids(patient_ids, output_file, csv_file):
    """
    save the list of ids in split text file

    Arguements:
    patient_ids -- list of ids
    output_file -- path
    """
    ids = pd.DataFrame(columns=['eid'])

    with open(output_file, 'w') as file:
        for patient_id in patient_ids:
            ids.loc[len(ids)] = {'eid': patient_id}
            file.write(f'{patient_id}-sa_ED\n{patient_id}-sa_ES\n')
    
    ids.to_csv(csv_file, index=False)


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


def generate_split(output_file, data_root, all_split, num_patients=7, num_frames=2, start_idx=0, mode='train', shuffle=True):
    patient_ids_frames = get_patient_ids_frames(all_split)
    lines = []

    for i in range(num_patients * num_frames):
        patient_id, frame = patient_ids_frames[i+start_idx]
        
        if mode == 'train':
            image_path = os.path.join(data_root, patient_id, f"{frame}.nii.gz")
            in_img = nib.load(image_path).get_fdata()[:]
            num_slices = in_img.shape[2]
            for slice_idx in range(num_slices):
                lines.append(f'{patient_id}-{frame}-{slice_idx}\n')
        else:
            lines.append(f'{patient_id}-{frame}\n')
    
    if shuffle:
        lines = shuffle_split(lines)
    
    with open(output_file, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def shuffle_split(lines, seed=42):
    random.seed(seed)
    random.shuffle(lines)
    return lines


def control_ethnicity(df, n):
    res_df = pd.DataFrame()
    for k, v in ETHNNICITY_CODING.items():
        cond = df['ethnicity'].astype(str).str.startswith(k)
        cur_df = df[cond].sample(n=n, random_state=42)
        res_df = pd.concat([res_df, cur_df])
    return res_df


def control_sex(df, split_prcnt, len_dataset):
    n = int(len_dataset * split_prcnt / 2)
    df_male = df[df['sex'] == 1].sample(n=n, random_state=42)
    df_female = df[df['sex'] == 0].sample(n=n, random_state=42)
    return pd.concat([df_male, df_female])


def control_age(df, mean=51.0, std=7.0):
    lw_bound = mean - std
    up_bound = mean + std
    return df[(df['age'] > lw_bound) & (df['age'] < up_bound)]