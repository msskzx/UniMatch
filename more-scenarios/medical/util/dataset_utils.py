import os
import torch
import cv2
from util.classes import MASK, ETHNNICITY_CODING, FRAME
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


def get_patient_ids_frames_from_csv(split, mode):
    df = pd.read_csv(split)
    res = []
    for idx, row in df.iterrows():
        if mode in ['train', 'train_l', 'train_u']:
            res.append((row['eid'], row['frame'], row['slice_idx']))
        else:
            res.append((row['eid'], row['frame']))

    return res


def generate_split(input_file, output_file, cfg=None, mode='train', shuffle=True):
    """
    given csv file containing split generate a txt file with ids, frames, slices
    """
    df = pd.read_csv(input_file)
    data = {
        'eid': [],
        'frame': [],
    }
    if mode == 'train':
        data['slice_idx'] = []

    for idx, row in df.iterrows():  
        for k, frame in FRAME.items():
            if mode == 'train':
                image_path = os.path.join(cfg['data_root'], str(int(row['eid'])), f"{frame}.nii.gz")
                in_img = nib.load(image_path).get_fdata()[:]
                num_slices = in_img.shape[2]
                for slice_idx in range(num_slices):
                    data['eid'].append(str(int(row["eid"])))
                    data['frame'].append(frame)
                    data['slice_idx'].append(slice_idx)
            else:
                data['eid'].append(str(int(row["eid"])))
                data['frame'].append(frame)


    res_df = pd.DataFrame(data=data)

    if shuffle:
        df = df.sample(frac=1, random_state=42)
    
    if mode == 'train':
        split_un_labeled(res_df, cfg)

    res_df.to_csv(output_file)


def split_un_labeled(df, cfg, frac=0.1):
    df_l = df.sample(frac=frac, random_state=42)
    df_u = df.drop(df_l.index)
    df_l.to_csv('splits/ukbb/labeled.csv')
    df_u.to_csv('splits/ukbb/unlabeled.csv')


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


def run_get_all_eids_from_dir(cfg):
    """
    used only once to get all ids
    """
    patient_ids = get_patient_ids_from_directory(cfg['data_root'])
    save_patient_ids(patient_ids, cfg['all_split'], cfg['all_csv'])