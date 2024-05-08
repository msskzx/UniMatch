from yaml import load, Loader, dump
import gen_splits
import os
import subprocess

from util.classes import EXPERIMENTS


def edit_config_slurm(dataset, mode, exp, seed, split, cfg_file):
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
    prev_split = cfg['split']
    for idx in indices:
        lines[idx-1] = lines[idx-1].replace(prev_split, split)
    exp_idx = 32 if mode == 'train' else 23
    lines[exp_idx] = f"exp_num='{exp}'"

    with open(script_path, 'w') as file:
        file.writelines(lines)


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
    dataset = 'ukbb'
    seeds = [42, 43, 47, 53, 57, 61, 71, 73, 79, 83]

    # skip if files are already generated
    generated = True
    if not generated:
        for seed in seeds:
            # generate train data for the exps
            gen_splits.main(seed=seed)

    # skip if already trained the models
    trained = True
    if not trained:
        # train 3 models per sampling
        mode = 'train'
        for seed in seeds:
            for exp, split in EXPERIMENTS.items():
                # edit config, slurm script for train
                edit_config_slurm(dataset, mode, exp, seed, split, cfg_file='config')

                # train model
                run_slurm(mode)

    # skip if already tested the models
    tested = True
    if not tested:
        testsets = ['sex', 'ethn']
        mode = 'test'
        for seed in seeds:
            for exp, split in EXPERIMENTS.items():
                for testset in testsets:
                    # edit config, slurm script
                    edit_config_slurm(dataset, mode, exp, seed, split, prev_split, cfg_file=testset)
                    # run test
                    run_slurm(mode)

                prev_split = split
                
                # TODO flag the exp as done


if __name__ == '__main__':
    main()