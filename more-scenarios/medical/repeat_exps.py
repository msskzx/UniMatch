from yaml import load, Loader, dump
import gen_splits
import subprocess
from util.classes import EXPERIMENTS
import os

def edit_config_slurm(dataset, mode, exp, seed, cfg_file):
    """
    cfg_file: values in ['config', 'sex', 'ethn']
    """
    cfg_path = f'configs/{dataset}/{mode}/exp{exp}/{cfg_file}.yaml'

    cfg = load(open(cfg_path, 'r'), Loader=Loader)
    # edit config
    cfg['seed'] = seed
    # only in test configs
    if 'control' in cfg:
        cfg['control'] = cfg_file

    cfg_out_dir = f'configs/{dataset}/{mode}/exp{exp}/seed{seed}'
    cfg_out_path = os.path.join(cfg_out_dir, f'{cfg_file}.yaml')
    if not os.path.exists(cfg_out_dir):
        os.makedirs(cfg_out_dir)
    with open(cfg_out_path, 'w') as file:
        dump(cfg, file)

    # edit slurm_ukbb_train.sh or test
    script_path = f'scripts/slurm_{dataset}_{mode}.sh'
    with open(script_path, 'r') as file:
        lines = file.readlines()

    # use new split number and exp number
    for i in range(len(lines)):
        # change exp num for sbatch info
        if i < 5 and 'exp' in lines[i]:
            idx = lines[i].find('exp')
            lines[i] = lines[i].replace(lines[i][idx+3], str(exp))

        # change exp num
        if 'exp_num=' in lines[i]:
            lines[i] = f"exp_num='{exp}'\n"
        
        # change exp num    
        if 'seed=' in lines[i]:
            lines[i] = f"seed='{seed}'\n"
        
        if 'control=' in lines[i]:
            lines[i] = f"control='{cfg_file}'\n"

    with open(script_path, 'w') as file:
        file.writelines(lines)


def run_slurm(mode):
    try:
        subprocess.run(f'sbatch scripts/slurm_ukbb_{mode}.sh', shell=True, check=True)
        print('Script execution completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e} sbatch scripts/slurm_ukbb_{mode}.sh')


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
    trained = False
    if not trained:
        # train 3 models per sampling
        mode = 'train'
        for seed in seeds:
            for exp, _ in EXPERIMENTS.items():
                # edit config, slurm script for train
                edit_config_slurm(dataset, mode, exp, seed, cfg_file='config')
                # train model
                run_slurm(mode)

    # skip if already tested the models
    tested = False
    if not tested:
        testsets = ['sex', 'ethn']
        mode = 'test'
        for seed in seeds:
            for exp, _ in EXPERIMENTS.items():
                for testset in testsets:
                    # edit config, slurm script
                    edit_config_slurm(dataset, mode, exp, seed, cfg_file=testset)
                    # run test
                    run_slurm(mode)


if __name__ == '__main__':
    main()