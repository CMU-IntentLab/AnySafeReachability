import subprocess

import numpy as np

margins = np.linspace(0.1, 0.5, 5)
alphas = [16, 32, 64, 128, 256]
num_examples_per_class = [10, 20, 50, 100, None]  # None means all examples
ul_ratios = [1.0, 2.0, 3.0, 10.0]
betas = [0.1, 0.2, 0.3, 0.4, 0.5]
temps = [0.05, 0.1, 0.2, 0.3, 0.4]

# Dont use unlabelled examples
for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Running with lr={lr}, batch_size={bs}")
        subprocess.run(
            [
                "python",
                "train_failure_classifier_pa.py",
                "--num-examples-per-class",
                str(ex),
                "--epochs",
                "100",
            ]
        )

# Use unlabelled examples
for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Running with lr={lr}, batch_size={bs}")
        subprocess.run(
            [
                "python",
                "train_failure_classifier_pa.py",
                "--use-unlabeled-data",
                "--num-examples-per-class",
                str(ex),
                "--epochs",
                "100",
                "--unlabeled-ratio",
                "3.0",
            ]
        )
