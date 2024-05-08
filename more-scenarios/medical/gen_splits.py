from yaml import load, Loader
from util.classes import FRAME, ETHNNICITY_CODING, ETHNNICITY_CODING_REVERSED, EXPERIMENTS
import nibabel as nib
import pandas as pd
import os
from util.analysis_utils import prep_patients_df
 

def generate_split(input_file, output_file, cfg=None, mode='train', seed=42, shuffle=True):
    """
    given csv file containing eids generate a csv file with eids, frames, slices

    Arguments:
    input_file -- csv containing ukbb patients info
    output_file -- csv containing patients eids for this split
    cfg -- loaded cfg file
    mode -- mode (train, val, test)
    shuffle -- shuffle (True, False)
    """
    df = pd.read_csv(input_file)
    data = {
        'eid': [],
        'frame': [],
    }
    if mode == 'train':
        data['slice_idx'] = []

    for _, row in df.iterrows():  
        for _, frame in FRAME.items():
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
        split_un_labeled(res_df, cfg, seed=seed)

    res_df.to_csv(output_file, index=False)


def split_un_labeled(df, cfg, seed=42, frac=0.1):
    df_l = df.sample(frac=frac, random_state=42)
    df_u = df.drop(df_l.index)
    df_l.to_csv(f'splits/{cfg["dataset"]}/{cfg["split"]}/seed{seed}/labeled.csv', index=False)
    df_u.to_csv(f'splits/{cfg["dataset"]}/{cfg["split"]}/seed{seed}/unlabeled.csv', index=False)


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


def save_patient_ids(patient_ids, csv_file):
    """
    save the list of ids in split text file

    Arguements:
    patient_ids -- list of ids
    output_file -- path
    """
    ids = pd.DataFrame({'eids': patient_ids})
    ids.to_csv(csv_file, index=False)


def run_get_all_eids_from_dir(cfg):
    """
    used only once to get all ids
    """
    patient_ids = get_patient_ids_from_directory(cfg['data_root'])
    save_patient_ids(patient_ids, f'splits/{cfg["dataset"]}/all.csv')


def control_ethnicity(df, n, sex_ctrl=False, seed=42):
    """
    extract equal number of samples per ethnic group

    Arguments:
    df -- input df
    n -- number of samples per ethnic group

    return: ethnicity controlled df
    """
    res_df = pd.DataFrame()
    for k, _ in ETHNNICITY_CODING.items():
        cond = df['ethnicity'].astype(str).str.startswith(k)
        cur_df = df[cond]
        if sex_ctrl:
            male_df = cur_df[cur_df['sex'] == 1].sample(n=n//2, random_state=seed)
            female_df = cur_df[cur_df['sex'] == 0].sample(n=n//2, random_state=seed)
            cur_df = pd.concat([male_df, female_df])
        else:
            cur_df = cur_df.sample(n=n, random_state=seed)
        res_df = pd.concat([res_df, cur_df])
    return res_df


def control_sex(df, split_prcnt, seed=42):
    """
    extract equal number of samples per sex
    
    Arguments:
    df -- input df
    split_prcnt -- split percentage (train, val, test)
    
    return: sex controlled df
    """
    # number of samples per sex
    n = int(len(df) * split_prcnt / 2)
    male_count = len(df[df['sex'] == 1])
    female_count = len(df) - male_count
    n = min(n, male_count, female_count)

    # sample
    df_male = df[df['sex'] == 1].sample(n=n, random_state=seed)
    df_female = df[df['sex'] == 0].sample(n=n, random_state=seed)
    return pd.concat([df_male, df_female])


def control_age(df, mean=51.0, std=7.0):
    """
    extract samples within certain age range
    
    Arguments:
    df -- input df
    mean -- age mean to extract around
    std -- standard deviation used to define range around mean
    
    return: sex controlled df
    """
    lw_bound = mean - std
    up_bound = mean + std
    return df[(df['age'] > lw_bound) & (df['age'] < up_bound)]


def get_ethnic_group(df, grp):
    """
    extract ethnic group from df given code

    Arguments:
    df -- input df
    grp -- ethnic group code

    return: ethnic group df
    """
    res_df = df[df['ethnicity'].astype(str).str.startswith(grp)]
    return res_df


def sample_from_dataset(sa_patients_df, wht_n=4000, seed=42):
    """
    sample from the whole data
    """
    all_wht_df = get_ethnic_group(sa_patients_df, ETHNNICITY_CODING_REVERSED['White'])
    all_as_df = get_ethnic_group(sa_patients_df, ETHNNICITY_CODING_REVERSED['Asian'])
    all_bl_df = get_ethnic_group(sa_patients_df, ETHNNICITY_CODING_REVERSED['Black'])

    sample_wht_df = all_wht_df.sample(n=wht_n, random_state=seed)
    dataset_df = pd.concat([sample_wht_df, all_as_df, all_bl_df])
    return dataset_df, sample_wht_df, all_as_df, all_bl_df


def gen_sex_ctrl(all_wht_df, all_as_df, all_bl_df, output_csv, split_prcnt, seed=42):
    """
    sample sex controlled df
    """
    # control for sex per ethnic group
    # I wonder why?
    wht_sex_ctrl_df = control_sex(all_wht_df, split_prcnt, seed=seed)
    as_sex_ctrl_df = control_sex(all_as_df, split_prcnt, seed=seed)
    bl_sex_ctrl_df = control_sex(all_bl_df, split_prcnt, seed=seed)
    sex_ctrl_df = pd.concat([wht_sex_ctrl_df, as_sex_ctrl_df, bl_sex_ctrl_df])

    # save to csv
    sex_ctrl_df.to_csv(output_csv, index=False)
    return sex_ctrl_df


def gen_ethn_ctrl(sample_wht_df, all_as_df, all_bl_df, output_csv, split_prcnt, sex_ctrl=False, seed=42):
    """
    sample ethnicity controlled df
    """
    dataset_df = pd.concat([sample_wht_df, all_as_df, all_bl_df])
    # control for ethnicity
    whtn = len(sample_wht_df) * split_prcnt
    asn = len(all_as_df) * split_prcnt
    bln = len(all_bl_df) * split_prcnt
    n = int(min(whtn, asn, bln))
    ethn_ctrl_df = control_ethnicity(dataset_df, n, sex_ctrl, seed=seed)
    ethn_ctrl_df.to_csv(output_csv, index=False)
    return ethn_ctrl_df


def gen_baseline_splits_csv():
    """
    Experiment 1 - Baseline
    extract train, val not controlled (4k)
    extract test controlled for age, sex, ethnicity (1k)
    """
    # load ukbb patients data
    og_df = pd.read_csv('/vol/aimspace/projects/ukbb/data/tabular/ukb668815_imaging.csv')

    # prep
    all_patients_df = prep_patients_df(og_df)

    # data with short axis images available
    all_sa_df = pd.read_csv('splits/ukbb/all.csv')

    # merge
    sa_patients_df = pd.merge(all_patients_df, all_sa_df, on='eid')

    # sample 4k white
    dataset_df, sample_wht_df, all_as_df, all_bl_df = sample_from_dataset(sa_patients_df, wht_n=4000)

    # control for age
    wht_age_ctrl_df = control_age(sample_wht_df)
    as_age_ctrl_df = control_age(all_as_df)
    bl_age_ctrl_df = control_age(all_bl_df)

    # gen 2 test split controlled for sex, ethnicity (both controlled for age)
    test_sex_ctrl_df = gen_sex_ctrl(wht_age_ctrl_df, as_age_ctrl_df, bl_age_ctrl_df, 'ukbb/test_sex_ctrl.csv', 0.2)
    test_ethn_ctrl_df = gen_ethn_ctrl(wht_age_ctrl_df, as_age_ctrl_df, bl_age_ctrl_df, 'ukbb/test_ethn_ctrl.csv', 0.2)
    whole_test_df = pd.concat([test_sex_ctrl_df, test_ethn_ctrl_df])

    train_val_df = dataset_df.drop(whole_test_df.index)
    val_df = train_val_df.sample(frac=0.13, random_state=42)
    train_df = train_val_df.drop(val_df.index)

    print(f'Train: {len(train_df)} subjects, {len(train_df)/len(dataset_df)*100.0:.2f}%')
    print(f'Validation: {len(val_df)} subjects, {len(val_df)/len(dataset_df)*100.0:.2f}%')
    print(f'Whole Test: {len(whole_test_df)} subjects, {len(whole_test_df)/len(dataset_df)*100.0:.2f}%')
    print(f'Test Sex Controlled: {len(test_sex_ctrl_df)} subjects, {len(test_sex_ctrl_df)/len(dataset_df)*100.0:.2f}%')
    print(f'Test Ethnicity Controlled: {len(test_ethn_ctrl_df)} subjects, {len(test_ethn_ctrl_df)/len(dataset_df)*100.0:.2f}%')

    train_df.to_csv('ukbb/train.csv', index=False)
    val_df.to_csv('ukbb/val.csv', index=False)


def gen_train_val_ctrl_csv(seed=42):
    """
    Experiment 2, 3, 4
    Train sets are controlled for age, sex, ethnicity
    """
    dataset = 'ukbb'
    # read already sampled train, val
    all_train_df = pd.read_csv(f'{dataset}/train.csv')
    all_val_df = pd.read_csv(f'{dataset}/val.csv')
    all_train_val_df = pd.concat([all_train_df, all_val_df])

    # extract ethnic groups
    sample_wht_df = get_ethnic_group(all_train_val_df, ETHNNICITY_CODING_REVERSED['White'])
    all_as_df = get_ethnic_group(all_train_val_df, ETHNNICITY_CODING_REVERSED['Asian'])
    all_bl_df = get_ethnic_group(all_train_val_df, ETHNNICITY_CODING_REVERSED['Black'])

    # control for age
    wht_age_ctrl_df = control_age(sample_wht_df)
    as_age_ctrl_df = control_age(all_as_df)
    bl_age_ctrl_df = control_age(all_bl_df)
    
    # EXPERIMENT 2
    # control for sex
    train_val_sex_ctrl_df = gen_sex_ctrl(wht_age_ctrl_df, as_age_ctrl_df, bl_age_ctrl_df, f'{dataset}/exp2/seed{seed}/train_val.csv', split_prcnt=1.0, seed=seed)

    # split train, val
    val_sex_ctrl_df = train_val_sex_ctrl_df.sample(frac=0.13, random_state=seed)
    train_sex_ctrl_df = train_val_sex_ctrl_df.drop(val_sex_ctrl_df.index)

    # save
    val_sex_ctrl_df.to_csv(f'{dataset}/exp2/seed{seed}/val.csv', index=False)
    train_sex_ctrl_df.to_csv(f'{dataset}/exp2/seed{seed}/train.csv', index=False)

    # EXPERIMENT 3
    # control for ethnicity
    train_val_ethn_ctrl_df = gen_ethn_ctrl(wht_age_ctrl_df, as_age_ctrl_df, bl_age_ctrl_df, f'{dataset}/exp3/seed{seed}/train_val.csv', split_prcnt=1.0, seed=seed)

    # split train, val
    val_ethn_ctrl_df = train_val_ethn_ctrl_df.sample(frac=0.13, random_state=seed)
    train_ethn_ctrl_df = train_val_ethn_ctrl_df.drop(val_ethn_ctrl_df.index)

    # save
    val_ethn_ctrl_df.to_csv(f'{dataset}/exp3/seed{seed}/val.csv', index=False)
    train_ethn_ctrl_df.to_csv(f'{dataset}/exp3/seed{seed}/train.csv', index=False)

    # EXPERIMENT 4
    # control for sex, ethnicity
    # extract ethnic groups
    wht_ethn_ctrl_df = get_ethnic_group(train_val_sex_ctrl_df, ETHNNICITY_CODING_REVERSED['White'])
    as_ethn_ctrl_df = get_ethnic_group(train_val_sex_ctrl_df, ETHNNICITY_CODING_REVERSED['Asian'])
    bl_ethn_ctrl_df = get_ethnic_group(train_val_sex_ctrl_df, ETHNNICITY_CODING_REVERSED['Black'])
    train_val_sex_ethn_ctrl_df = gen_ethn_ctrl(wht_ethn_ctrl_df, as_ethn_ctrl_df, bl_ethn_ctrl_df, f'{dataset}/exp4/seed{seed}/train_val.csv', split_prcnt=1.0, sex_ctrl=True, seed=seed)

    # split train, val
    val_sex_ethn_ctrl_df = train_val_sex_ethn_ctrl_df.sample(frac=0.13, random_state=seed)
    train_sex_ethn_ctrl_df = train_val_sex_ethn_ctrl_df.drop(val_sex_ethn_ctrl_df.index)

    # save
    val_sex_ethn_ctrl_df.to_csv(f'{dataset}/exp4/seed{seed}/val.csv', index=False)
    train_sex_ethn_ctrl_df.to_csv(f'{dataset}/exp4/seed{seed}/train.csv', index=False)


def main(seed=43):
    dataset = 'ukbb'
    mode = 'train'

    # EXPERIMENT 1
    # generate baseline splits csv - expirement 1
    #gen_baseline_splits_csv()

    # read generated csv files that contain ids only to produce splits that contain ids, frames, slices
    #cfg = load(open('configs/ukbb/train/ukbb.yaml', 'r'), Loader=Loader)
    #generate_split(input_file='ukbb/val.csv', output_file='splits/ukbb/val.csv', mode='val', shuffle=True)
    #generate_split(input_file='ukbb/test_sex_ctrl.csv', output_file='splits/ukbb/test_sex_ctrl.csv', mode='test', shuffle=False)
    #generate_split(input_file='ukbb/test_ethn_ctrl.csv', output_file='splits/ukbb/test_ethn_ctrl.csv', mode='test', shuffle=False)
    #generate_split(input_file='ukbb/train.csv', output_file='splits/ukbb/train.csv', mode='train', cfg=cfg, shuffle=True)

    # ---
    # EXPERIMENTS 2, 3, 4

    # make dirs
    for exp, split in EXPERIMENTS.items():
        dataset_exp_path = f'{dataset}/exp{exp}/seed{seed}'
        if not os.path.exists(dataset_exp_path):
            os.makedirs(dataset_exp_path)

        split_exp_path = f'splits/{dataset}/{split}/seed{seed}'
        if not os.path.exists(split_exp_path):
            os.makedirs(split_exp_path)

    # prepare train and validation data distributions
    gen_train_val_ctrl_csv(seed)

    for exp, split in EXPERIMENTS.items():
        # EXPERIMENT k
        cfg = load(open(f'configs/{dataset}/{mode}/exp{exp}/config.yaml', 'r'), Loader=Loader)
        generate_split(input_file=f'{dataset}/exp{exp}/seed{seed}/train.csv', output_file=f'splits/{dataset}/{split}/seed{seed}/train.csv', mode='train', cfg=cfg, seed=seed, shuffle=True)
        generate_split(input_file=f'{dataset}/exp{exp}/seed{seed}/val.csv', output_file=f'splits/{dataset}/{split}/seed{seed}/val.csv', mode='val', seed=seed, shuffle=True)


if __name__ == '__main__':
    main()
