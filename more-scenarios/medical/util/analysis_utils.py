import pandas as pd
from util.classes import ETHNNICITY_CODING
import seaborn as sns
import matplotlib.pyplot as plt
from util.classes import PRIMARY_DEMOGRAPHICS
import re


one_palette = sns.color_palette("pastel")[0]
two_palette = sns.color_palette("pastel")[:2]
three_palette = sns.color_palette("pastel")[:3]
four_palette = sns.color_palette("pastel")[0:4]


def get_all_results(df, cfg):
    res = {}

    dice = get_dice(df, cfg)

    sex_dice, ethn_dice = get_group_results(df, cfg)

    low_scores_prcnt = get_slices(df, cfg)

    res = {
        'experiment': [cfg['exp']],
        'model': [cfg['model']],
        'train_set': [cfg['train_set']],
        'test_set': [cfg['test_set']],
        'epochs': [cfg['epochs']],
        'dice_mean': [dice['mean']],
        'rv_mean': [dice['rv']],
        'myo_mean': [dice['myo']],
        'lv_mean': [dice['lv']],
        'male_dice': [sex_dice['male']],
        'female_dice': [sex_dice['female']],
        'white_dice': [ethn_dice['White']],
        'asian_dice': [ethn_dice['Asian']],
        'black_dice': [ethn_dice['Black']],
        'low_scores_prcnt': [low_scores_prcnt]
    }

    return pd.DataFrame(res)


def merge_results_patients(df, patients_df):
    return pd.merge(df, patients_df, left_on='patient_id', right_on='eid', how='inner')


def prep_patients_df(og_df):   
    # extract cols of interest
    all_cols = og_df.columns
    cols = []
    for _, s in enumerate(all_cols):
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


def plot_exps(df):
    plot_all_dice(df, x='experiment', y='dice_mean', hue='experiment', palette=four_palette, title='Mean Dice')

    plot_all_dice(df, x='experiment', y='male_dice', hue='experiment', palette=four_palette, title='Male Dice')

    plot_all_dice(df, x='experiment', y='female_dice', hue='experiment', palette=four_palette, title='Female Dice')

    plot_all_dice(df, x='experiment', y='white_dice', hue='experiment', palette=four_palette, title='White Dice')

    plot_all_dice(df, x='experiment', y='asian_dice', hue='experiment', palette=four_palette, title='Asian Dice')

    plot_all_dice(df, x='experiment', y='black_dice', hue='experiment', palette=four_palette, title='Black Dice')

    plot_all_dice(df, x='experiment', y='low_scores_prcnt', hue='experiment', palette=four_palette, title='Slices Scores < 50%')



def plot_all_dice(df, x, y, hue, palette, title):
    min_val = df[y].min() - 0.5
    max_val = df[y].max() + 0.5
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette).set_ylim(min_val, max_val)
    plt.xticks(df[x].tolist())
    plt.xlabel('Experiment')
    plt.ylabel('Mean Dice')
    plt.title(f'All Experiments - {title}')
    plt.show()


def get_group_results(df, cfg):
    sex_df = df.groupby('sex')['dice_mean'].mean()
    sex_dice = {
        'male': sex_df[1],
        'female': sex_df[0],
    }

    print(f'Dice Mean for Male: {sex_df[1]:.2f}')
    print(f'Dice Mean for Female: {sex_df[0]:.2f}')
    plot_dice(sex_dice, two_palette, cfg)

    ethn_dice = {}

    for _, k in enumerate(ETHNNICITY_CODING):
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethn_df = df[cond]

        mean = ethn_df.groupby('ethnicity')['dice_mean'].mean().reset_index()
        print(f'Mean Dice for {ETHNNICITY_CODING[k]}:\n{mean}')
        print(f'Mean: {mean["dice_mean"][0]}')
        ethn_dice[ETHNNICITY_CODING[k]] = mean['dice_mean'][0]

    plot_dice(ethn_dice, three_palette, cfg)

    return sex_dice, ethn_dice


def get_dice(df, cfg):
    print('Number of Patients:', df['eid'].nunique())
    print('Total number of slices:', len(df))

    dice = {
        'mean': df['dice_mean'].mean(),
        'rv': df['dice_rv'].mean(),
        'myo': df['dice_myo'].mean(),
        'lv': df['dice_lv'].mean(),
    }
    print(f"DICE Mean: {dice['mean']:.2f}")
    print(f"RV Mean: {dice['rv']:.2f}")
    print(f"MYO Mean: {dice['myo']:.2f}")
    print(f"LV Mean: {dice['lv']:.2f}")

    plot_dice(dice, four_palette, cfg)
    return dice


def plot_dice(dice, palette, cfg):
    categories = list(dice.keys())
    values = list(dice.values())
    sns.pointplot(x=categories, hue=categories, y=values, palette=palette).set_ylim(min(values)-0.5, max(values)+0.5)
    plt.xlabel('Class')
    plt.ylabel('Mean')
    plt.title(f'Experiment {cfg["exp"]} - Mean Dice per Class')
    plt.show()


def get_slices(df, cfg):
    thresh = 50.0
    filtered_rows = df[df['dice_mean'] < thresh]
    print(f'Experiment {cfg["exp"]} - Number of slices with DICE < {thresh}%: {len(filtered_rows)}')

    plot_slices(filtered_rows, thresh, cfg)
    return len(filtered_rows) / len(df) * 100.0


def plot_slices(df, thresh, cfg):
    sns.kdeplot(df['slice_idx'], color=one_palette, fill=True)
    plt.xlim(0, df['slice_idx'].max())
    plt.xlabel('Slice Index')
    plt.ylabel('Count')
    plt.title(f'Experiment {cfg["exp"]} - Slice Index Distribution with DICE < {thresh}%')
    plt.show()


def plot_age_dist(df):
    plt.figure(figsize=(8, 6))
    for _, k in enumerate(ETHNNICITY_CODING):
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethnic_group_df = df[cond]
        sns.kdeplot(ethnic_group_df['age'], label=ETHNNICITY_CODING[k], fill=True)
        mean = ethnic_group_df['age'].mean()
        std = ethnic_group_df['age'].std()
        
        print(f'Mean age for {ETHNNICITY_CODING[k]}: {mean:.2f}')
        print(f'STD age for {ETHNNICITY_CODING[k]}: {std:.2f}')

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