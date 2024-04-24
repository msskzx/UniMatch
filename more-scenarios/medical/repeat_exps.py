from yaml import load, Loader, dump
import gen_splits
import os
import subprocess

from util.classes import EXPERIMENTS


def edit_config_slurm(dataset, mode, exp, seed, split, prev_split, cfg_file):
    # edit config
    cfg_path = f'configs/{dataset}/{mode}/exp{exp}/{cfg_file}.yaml'
    cfg = load(open(cfg_path, 'r'), Loader=Loader)
    cfg['seed'] = seed
    with open(cfg_path, 'w') as file:
        dump(cfg, file)

    # edit slurm_ukbb_train.sh
    script_path = f'scripts/slurm_{dataset}_{mode}.sh'
    with open(script_path, 'r') as file:
        lines = file.readlines()

    # replace prev split
    indices = [2, 3, 4, 22]
    if mode == 'train':
        indices.extend([32, 44])
    for idx in indices:
        lines[idx-1] = lines[idx-1].replace(prev_split, split)
    exp_idx = 32 if mode == 'train' else 23
    lines[exp_idx] = f"exp_num='{exp}'"

    with open(script_path, 'w') as file:
        file.writelines(lines)


def remove_prev_model(dataset, split):
    files = ['best.pth', 'latest.pth', 'events.*']
    for file in files:
        file_path = f'exp/{dataset}/unimatch/unet/{split}/{file}'
        if os.path.exists(file_path):
            os.remove(file_path)


def run_slurm(mode):
    try:
        subprocess.run(f'sbatch scripts/slurm_ukbb_{mode}.sh', shell=True, check=True)
        print('Script execution completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')


def main():
    """
    repeat experiments 10 times while sampling again from distribution
    """
    #seeds = [42, 43, 47, 53, 57, 61, 71, 73, 79, 83]
    # test
    seeds = [47]

    # experiments to repeat
    dataset = 'ukbb'
    testsets = ['sex', 'ethn']
    max_score = 0

    for seed in seeds:
        # generate train data for the exps
        gen_splits.main(seed=seed)

        # TODO get prev split from cfg instead
        prev_split = 18
        for exp, split in EXPERIMENTS.items():
            # edit config, slurm script for train
            mode = 'train'
            edit_config_slurm(dataset, mode, exp, seed, split, prev_split, cfg_file='config')

            # delete prev model
            remove_prev_model(dataset, split)
            
            # train model
            run_slurm(mode)
            # TODO save score in a txt file

            # TODO get score from txt file
            score = 0
            if score > max_score:
                max_score = score
                # TODO copy best.pth
                s = f'exp/{dataset}/unimatch/unet/{exp}/best.pth'
                gs = f'exp/{dataset}/unimatch/unet/{exp}/repeat_best.pth'
            
            mode = 'test'
            for testset in testsets:
                # edit config, slurm script
                edit_config_slurm(dataset, mode, exp, seed, split, prev_split, cfg_file=testset)
                # run test
                run_slurm(mode)

            


            prev_split = split



if __name__ == '__main__':
    main()