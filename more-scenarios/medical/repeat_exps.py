from yaml import load, Loader, dump
import gen_splits
import subprocess
from util.classes import EXPERIMENTS, SEEDS


def edit_slurm(dataset, method, mode, exp, seed, cfg_file):
    """
    cfg_file: values in ['config', 'sex', 'ethn']
    """
    script_path = f'slurm/{dataset}/{method}/{mode}.sh'
    with open(script_path, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        if 'python -u' in lines[i]:
            break

        # change exp num for sbatch info
        if i < 5 and 'exp' in lines[i]:
            idx = lines[i].find('exp')
            lines[i] = lines[i].replace(lines[i][idx+3], str(exp))

        # change exp num
        if 'exp=' in lines[i]:
            lines[i] = f"exp='{exp}'\n"
        
        # change exp num    
        if 'seed=' in lines[i]:
            lines[i] = f"seed='{seed}'\n"
        
        if 'control=' in lines[i]:
            lines[i] = f"control='{cfg_file}'\n"

        if 'port=' in lines[i]:
            lines[i] = f"port='80{seed}'\n"
        
    with open(script_path, 'w') as file:
        file.writelines(lines)


def run_slurm(dataset, method, mode):
    try:
        subprocess.run(f'sbatch slurm/{dataset}/{method}/{mode}.sh', shell=True, check=True)
        print('Script execution completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')


def main(dataset='ukbb', method='supervised', generate_splits=False, train=False, test=False):
    """
    repeat experiments 10 times while sampling again from distribution
    """
    # skip if files are already generated train splits
    if generate_splits:
        for seed in SEEDS:
            gen_splits.main(seed=seed)

    # skip if already trained the models
    if train:
        mode = 'train'
        for seed in SEEDS:
            for exp in [4]:
                edit_slurm(dataset=dataset, method=method, mode=mode, exp=exp, seed=seed, cfg_file='config')
                run_slurm(dataset=dataset, method=method, mode=mode)

    # skip if already tested the models
    if test:
        testsets = ['ethn']
        mode = 'test'
        for seed in SEEDS:
            for exp in [4]:
                for testset in testsets:
                    edit_slurm(dataset=dataset, method=method, mode=mode, exp=exp, seed=seed, cfg_file=testset)
                    run_slurm(dataset=dataset, method=method, mode=mode)

if __name__ == '__main__':
    main(test=True)