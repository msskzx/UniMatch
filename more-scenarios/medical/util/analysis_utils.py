import pandas as pd
from util.classes import ETHNNICITY_CODING
import seaborn as sns
import matplotlib.pyplot as plt


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

    # Add title and labels
    plt.title('Density Plot of Age Distribution per Ethnic Group')
    plt.xlabel('Age')
    plt.ylabel('Density')

    # Add legend
    plt.legend()

    # Show the plot
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