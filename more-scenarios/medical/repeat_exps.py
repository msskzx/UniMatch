import gen_splits


def main():
    # add 42, 43
    seeds = [47, 53, 57, 61, 71, 73, 79, 83]
    for seed in seeds:
        # generate train data
        gen_splits.main(seed=seed)

        models_splits = [18, 26, 80]
        # TODO for model in models:

            # TODO edit config file 

            # TODO edit slurm_ukbb_train.sh

            # TODO train model by running sbatch scripts/slurm_ukbb_train.sh

            # TODO infere using model

            # TODO save global best in repeated exps


if __name__ == '__main__':
    main()