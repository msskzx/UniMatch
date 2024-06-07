import pandas as pd
from util.classes import ETHNNICITY_CODING
import seaborn as sns
import matplotlib.pyplot as plt
from util.classes import PRIMARY_DEMOGRAPHICS, EXPERIMENTS
import re


one_palette = sns.color_palette("pastel")[0]
two_palette = sns.color_palette("pastel")[8:]
three_palette = sns.color_palette("pastel")[:3]
three_palette_exps = sns.color_palette("pastel")[3:6]
four_palette = sns.color_palette("pastel")[2:6]


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
        'dice_mean': [dice['dice_mean']],
        'dice_lv': [dice['dice_lv']],
        'dice_rv': [dice['dice_rv']],
        'dice_myo': [dice['dice_myo']],
        'male': [sex_dice['male']],
        'male_lv': [sex_dice['male_lv']],
        'male_rv': [sex_dice['male_rv']],
        'male_myo': [sex_dice['male_myo']],
        'female': [sex_dice['female']],
        'female_lv': [sex_dice['female_lv']],
        'female_rv': [sex_dice['female_rv']],
        'female_myo': [sex_dice['female_myo']],
        'white': [ethn_dice['White']],
        'white_rv': [ethn_dice['White_rv']],
        'white_lv': [ethn_dice['White_lv']],
        'white_myo': [ethn_dice['White_myo']],
        'asian': [ethn_dice['Asian']],
        'asian_rv': [ethn_dice['Asian_rv']],
        'asian_lv': [ethn_dice['Asian_lv']],
        'asian_myo': [ethn_dice['Asian_myo']],
        'black': [ethn_dice['Black']],
        'black_rv': [ethn_dice['Black_rv']],
        'black_lv': [ethn_dice['Black_lv']],
        'black_myo': [ethn_dice['Black_myo']],
        'white_male': [ethn_dice['White_male']],
        'asian_male': [ethn_dice['Asian_male']],
        'black_male': [ethn_dice['Black_male']],
        'white_female': [ethn_dice['White_female']],
        'asian_female': [ethn_dice['Asian_female']],
        'black_female': [ethn_dice['Black_female']],
        'low_scores_prcnt': [low_scores_prcnt],
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


def boxplot_sex_dice(df, cls='', g1='', g2=''):
    if not g1:
        g1 = f'male{cls}'
    if not g2:
        g2 = f'female{cls}'

    for exp in EXPERIMENTS:
        df_melted = pd.melt(df[df['experiment'] == int(exp)], value_vars=[g1, g2], var_name='sex', value_name='sex_dice')
        boxplot_all_dice(df_melted, x='sex', y='sex_dice', hue='sex', palette=two_palette, xlabel='Sex', title=f'Experiment {exp} - Male vs. Female Mean Dice')


def boxplot_ethn_dice(df, cls =''):
    for exp in EXPERIMENTS:
        df_melted = pd.melt(df[df['experiment'] == int(exp)], value_vars=[f'white{cls}', f'asian{cls}', f'black{cls}'], var_name='ethnicity', value_name='ethnicity_dice')
        boxplot_all_dice(df_melted, x='ethnicity', y='ethnicity_dice', hue='ethnicity', palette=three_palette, xlabel='Ethnicity', title=f'Experiment {exp} - Ethnic Groups Dice')
    

def plot_exps(df):
    # overall segmentation
    boxplot_all_dice(df, x='experiment', y='dice_mean', palette=three_palette_exps, title='All Experiments - Overall Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='dice_rv', palette=three_palette_exps, title='All Experiments - RV Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='dice_lv', palette=three_palette_exps, title='All Experiments - LV Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='dice_myo', palette=three_palette_exps, title='All Experiments - MYO Mean Dice', rm_lgnd=True)

    # sex groups
    boxplot_all_dice(df, x='experiment', y='male', palette=three_palette_exps, title='All Experiments - Males Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='female', palette=three_palette_exps, title='All Experiments - Females Mean Dice', rm_lgnd=True)
    
    # ethnic groups
    boxplot_all_dice(df, x='experiment', y='white', palette=three_palette_exps, title='All Experiments - White Group Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='asian', palette=three_palette_exps, title='All Experiments - Asian Group Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='black', palette=three_palette_exps, title='All Experiments - Black Group Mean Dice', rm_lgnd=True)

    # intersectional groups dice mean
    boxplot_all_dice(df, x='experiment', y='white_male', palette=three_palette_exps, title='All Experiments - White Males Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='asian_male', palette=three_palette_exps, title='All Experiments - Asian Males Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='black_male', palette=three_palette_exps, title='All Experiments - Black Males Mean Dice', rm_lgnd=True)

    boxplot_all_dice(df, x='experiment', y='white_female', palette=three_palette_exps, title='All Experiments - White Females Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='asian_female', palette=three_palette_exps, title='All Experiments - Asian Females Mean Dice', rm_lgnd=True)
    boxplot_all_dice(df, x='experiment', y='black_female', palette=three_palette_exps, title='All Experiments - Black Females Mean Dice', rm_lgnd=True)

    boxplot_all_dice(df, x='experiment', y='low_scores_prcnt', palette=three_palette_exps, title='All Experiments - Slices Scores < 50%', rm_lgnd=True)


def boxplot_all_dice(df, x, y, palette, title, hue='experiment', xlabel='Experiment', ylabel='Mean Dice', rm_lgnd=False, min_y=None, max_y=None):
    plt.subplots(figsize=(4, 4))
    if min_y:
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette, width=0.1).set_ylim(min_y, max_y)
    else:
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette, width=0.1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rm_lgnd:
        plt.legend().remove()
    plt.title(title, fontsize=11)
    plt.show()


def scatterplot_all_dice(df, x, y, hue, palette, title):
    min_val = df[y].min() - 0.5
    max_val = df[y].max() + 0.5
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette).set_ylim(min_val, max_val)
    plt.xticks(df[x].tolist())
    plt.xlabel('Experiment')
    plt.ylabel('Mean Dice')
    plt.title(f'All Experiments - {title}')
    plt.show()


def get_group_results(df, cfg, plot=False):
    sex_df = df.groupby('sex').agg({
        'dice_mean': 'mean',
        'dice_lv': 'mean',
        'dice_rv': 'mean',
        'dice_myo': 'mean',
    })

    sex_dice = {
        'male': sex_df.loc[1, 'dice_mean'],
        'male_lv': sex_df.loc[1, 'dice_lv'],
        'male_rv': sex_df.loc[1, 'dice_rv'],
        'male_myo': sex_df.loc[1, 'dice_myo'],
        'female': sex_df.loc[0, 'dice_mean'],
        'female_lv': sex_df.loc[0, 'dice_lv'],
        'female_rv': sex_df.loc[0, 'dice_rv'],
        'female_myo': sex_df.loc[0, 'dice_myo'],
    }

    if plot:
        plot_dice(sex_dice, two_palette, cfg)

    ethn_dice = {}

    for _, k in enumerate(ETHNNICITY_CODING):
        cond = df['ethnicity'].astype(str).str.startswith(k)
        ethn_df = df[cond]

        # TODO save subgroups for ethnic groups
        ethn_gr = ethn_df.groupby('ethnicity').agg({
            'dice_mean': 'mean',
            'dice_rv': 'mean',
            'dice_lv': 'mean',
            'dice_myo': 'mean',
        })

        ethn_subgr = ethn_df.groupby('sex').agg({
            'dice_mean': 'mean',
        })

        ethn_dice[ETHNNICITY_CODING[k]] = ethn_gr['dice_mean'].mean()
        ethn_dice[f'{ETHNNICITY_CODING[k]}_rv'] = ethn_gr['dice_rv'].mean()
        ethn_dice[f'{ETHNNICITY_CODING[k]}_lv'] = ethn_gr['dice_lv'].mean()
        ethn_dice[f'{ETHNNICITY_CODING[k]}_myo'] = ethn_gr['dice_myo'].mean()
        ethn_dice[f'{ETHNNICITY_CODING[k]}_male'] = ethn_subgr.loc[1, 'dice_mean']
        ethn_dice[f'{ETHNNICITY_CODING[k]}_female'] = ethn_subgr.loc[0, 'dice_mean']

    if plot:
        plot_dice(ethn_dice, three_palette, cfg)

    return sex_dice, ethn_dice


def get_dice(df, cfg, plot=False, stdout=False):
    dice = {
        'dice_mean': df['dice_mean'].mean(),
        'dice_rv': df['dice_rv'].mean(),
        'dice_myo': df['dice_myo'].mean(),
        'dice_lv': df['dice_lv'].mean(),
    }

    if stdout:
        print('Number of Patients:', df['eid'].nunique())
        print('Total number of slices:', len(df))
        print(f"DICE Mean: {dice['dice_mean']:.2f}")
        print(f"RV Mean: {dice['dice_rv']:.2f}")
        print(f"MYO Mean: {dice['dice_myo']:.2f}")
        print(f"LV Mean: {dice['dice_lv']:.2f}")

    if plot:
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


def get_slices(df, cfg, plot=False, stdout=False):
    thresh = 50.0
    filtered_rows = df[df['dice_mean'] < thresh]
    
    if stdout:
        print(f'Experiment {cfg["exp"]} - Number of slices with DICE < {thresh}%: {len(filtered_rows)}')

    if plot:
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