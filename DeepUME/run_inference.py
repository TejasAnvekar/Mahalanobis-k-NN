import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with MAH or OG model.")
    parser.add_argument("--mah", action="store_true", help="Use the MAH model instead of the OG model")
    args = parser.parse_args()

    MAH = args.mah

    path_OG = "checkpoints/deepume/models/model.best.t7"
    path_MAH = "checkpoints/deepume_mah/models/model.best.t7"
    DATASETS = ['FAUST', 'ModelNet40', 'Stanford']
    NOISE = ['sampling', 'zero_intersec', 'bernoulli', 'gaussian', '']

    for dataset in DATASETS:
        for noise in NOISE:
            if MAH:
                run_command = (
                    f"python main_mah.py --exp_name=MAH_{noise}_{dataset} "
                    f"--eval --pretrained={path_MAH} --noise={noise} --test_dataset={dataset}"
                )
            else:
                run_command = (
                    f"python main.py --exp_name=OG_{noise}_{dataset} "
                    f"--eval --pretrained={path_OG} --noise={noise} --test_dataset={dataset}"
                )
            os.system(run_command)
