#!/bin/bash

for i in {0..2}
do
    cd dino_wm
    python train_dino_classifier.py "$i"
    cd ..
    cd scripts
    python run_training_ddpg-dinowm.py --latent-safe --class-id "$i"
    cd ..
done