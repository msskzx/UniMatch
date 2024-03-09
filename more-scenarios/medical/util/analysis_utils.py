import pandas as pd
from util.classes import ETHNNICITY_CODING
import seaborn as sns
import matplotlib.pyplot as plt
from yaml import load, Loader
from util.classes import PRIMARY_DEMOGRAPHICS
import re

def merge_result_info(df, patients_df):
    return pd.merge(df, patients_df, left_on='patient_id', right_on='eid', how='inner')

def get_patients_info():
    og_df = pd.read_csv('/vol/aimspace/projects/ukbb/data/tabular/ukb668815_imaging.csv')
    
    # extract cols of interest
    all_cols = og_df.columns
    cols = []
    for index, s in enumerate(all_cols):
        for k, v in PRIMARY_DEMOGRAPHICS.items():
            pattern = r'{}-(0.0)'.format(re.escape(k))
            if re.match(pattern, s):
                cols.append(s)

    # rename cols instead of codes
    new_cols = []
    for old_col in cols:
        new_cols.append(PRIMARY_DEMOGRAPHICS[old_col.split('-')[0]])
    cols.append('eid')
    new_cols.append('eid')
    df = og_df[cols]
    df.columns = new_cols

    return df


def get_group_results(df):
    sex_df = df.groupby('sex')['dice_mean'].mean()
    print(f'Dice Mean for Male: {sex_df[1]}')
    print(f'Dice Mean for Female: {sex_df[0]}')

    for i, k in enumerate(ETHNNICITY_CODING):
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethn_df = df[cond]

        mean = ethn_df.groupby('ethnicity')['dice_mean'].mean().reset_index()
        print(f'Mean Dice for {ETHNNICITY_CODING[k]}:\n{mean}')
        print(f'Mean: {mean["dice_mean"][0]}')


def get_results(df):
    print('Number of Patients:', df['eid'].nunique())
    print('Total number of slices:', len(df))

    print('---')
    print("DICE Mean: ", df['dice_mean'].mean())
    print("RV Mean: ", df['dice_rv'].mean())
    print("MYO Mean: ", df['dice_myo'].mean())
    print("LV Mean: ", df['dice_lv'].mean())
    print('---')

    thresh = 50.0
    filtered_rows = df[df['dice_mean'] < thresh]
    print(f'Number of slices with DICE < {thresh}%:', len(filtered_rows))

    count_df = filtered_rows[['slice_idx', 'eid']].groupby('slice_idx').count().reset_index()
    count_df.columns = ['slice_idx', 'count']
    sns.barplot(x='slice_idx', y='count', data=count_df)
    plt.xlabel('Slice Index')
    plt.ylabel('Count')
    plt.title(f'Slice Index Distribution with DICE < {thresh}%')
    plt.show()


def plot_age_dist(df):
    plt.figure(figsize=(8, 6))
    for i, k in enumerate(ETHNNICITY_CODING):
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethnic_group_df = df[cond]
        sns.kdeplot(ethnic_group_df['age'], label=ETHNNICITY_CODING[k], fill=True)
        mean = ethnic_group_df['age'].mean()
        std = ethnic_group_df['age'].std()
        
        print(f'Mean age for {ETHNNICITY_CODING[k]}:', mean)
        print(f'STD age for {ETHNNICITY_CODING[k]}:', std)

    plt.title('Density Plot of Age Distribution per Ethnic Group')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plot_sex_dist(df):
    patient_count = {}
    ctgrz = ['Total', 'Male', 'Female']
    fig, axs = plt.subplots(1, len(ETHNNICITY_CODING), figsize=(9, 3))
    i = 0

    for k, v in ETHNNICITY_CODING.items():
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethnic_group_df = df[cond]
        count = ethnic_group_df.shape[0]
        male = (ethnic_group_df['sex'] == 1).sum()
        female = count - male
        print(f'Coding {k} means {v} and has {count} patients, male {male}, female {female}')
        patient_count[k] = [count, male, female]
        palette = sns.color_palette("pastel")[7:]
        sns.barplot(hue=ctgrz, y=patient_count[k], ax=axs[i], palette=palette)
        axs[i].set_title(ETHNNICITY_CODING[k])
        i += 1
    plt.show