#!/bin/bash

python scripts/generate_data_traj_cont.py

python scripts/dreamer_offline.py

python scripts/wm_analysis.py 

python scripts/run_training_ddpg-wm.py 