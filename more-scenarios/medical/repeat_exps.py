from yaml import load, Loader, dump
import gen_splits
import os
import subprocess

def edit_config_slurm_train(dataset, exp, seed, prev_split, split):
    # edit config
    mode = 'train'
    cfg_path = f'configs/{dataset}/{mode}/exp{exp}/config.yaml'
    cfg = load(open(cfg_path, 'r'), Loader=Loader)
    cfg['seed'] = seed
    with open(cfg_path, 'w') as file:
        dump(cfg, file)

    # edit slurm_ukbb_train.sh
    with open('scripts/slurm_ukbb_train.sh', 'r') as file:
        lines = file.readlines()

    # replace prev split
    indices = [2, 3, 4, 22, 32, 44]
    for idx in indices:
        lines[idx-1] = lines[idx-1].replace(prev_split, split)

    lines[32] = f"exp_num='{exp}'"


def remove_prev_model(dataset, split):
    files = ['best.pth', 'latest.pth', 'events.*']
    for file in files:
        file_path = f'exp/{dataset}/{split}/{file}'
        if os.path.exists(file_path):
            os.remove(file_path)


def run_slurm(mode):
    try:
        subprocess.run(f'sbatch scripts/slurm_ukbb_{mode}.sh', shell=True, check=True)
        print('Script execution completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')


def edit_config_slurm_test(dataset, exp, seed, testset):
    mode = 'test'
    # edit config
    cfg_path = f'configs/{dataset}/{mode}/exp{exp}/{testset}.yaml'
    cfg = load(open(cfg_path, 'r'), Loader=Loader)
    cfg['seed'] = seed
    cfg['results_path'] = f'outputs/results/csv/{dataset}/exp{exp}/{testset}_seed_{seed}.csv'
    with open(cfg_path, 'w') as file:
        dump(cfg, file)


def main():
    # add 42, 43
    seeds = [47, 53, 57, 61, 71, 73, 79, 83]
    exps = {
        '2': 80,
        '3': 26,
        '4': 18
    }
    prev_split = 18
    dataset = 'ukbb'
    testsets = ['sex', 'ethn']

    for seed in seeds:
        # generate train data for the three exps
        gen_splits.main(seed=seed)

        for exp, split in exps.items():
            # edit config, slurm script for train
            edit_config_slurm_train(dataset, exp, seed, prev_split, split)

            # delete prev model
            remove_prev_model(dataset, split)
            
            # train model
            run_slurm(mode='train')
            
            for testset in testsets:
                # edit config, slurm for test
                edit_config_slurm_test(dataset, exp, seed, testset)

                # test
                run_slurm(mode='inference')

            # TODO save global best in repeated exps



if __name__ == '__main__':
    main()