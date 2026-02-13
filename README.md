# 🏎️ Dubins Car


## 📦 Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/CMU-IntentLab/AnySafeReachability.git
git checkout dubins
cd AnySafeReachability

# Create and activate conda environment
conda env create -f environment.yml
conda activate anysafe

# Login to wandb
wandb login
```

## 🗂️ Dataset Generation

Generate synthetic data for training the world model.

```bash
python scripts/generate_data_traj_cont.py
```

Generates **`wm_demos_dubins_sc_F_arrow_0.15_128.pkl`** file. (Name may change depending on config)


## 🧠 World Model Training

Train the latent dynamics world model using the generated dataset.

```bash
python scripts/dreamer_offline.py
```

Produces **`rssm_ckpt.pt`** file.

## 🏋️ Train semantic encoder

```bash
python scripts/train_failure_classifier_sem_dubins.py
```
Produces **`encoder_task_dubins-wm.pth.pt`** file.

Download pre-trained world model and semantic encoder:
```bash
pip install gdown
mkdir -p logs/checkpoints_sem/
gdown "https://drive.google.com/file/d/1tCRG6cpBcrLcbA0ckf9Hquk3Bekhg_S6/view?usp=sharing" \
  -O logs/checkpoints_sem/encoder_task_dubins-wm.pth
```

## 🛡️ Constraint-Conditioned Latent Reachability Analysis

Run RL for constraint-conditioned latent-space reachability:

```bash
python scripts/run_training_ddpg_wm.py 
```

Download pre-trained filter:
```bash
pip install gdown
mkdir -p "logs/dreamer_dubins/PyHJ/sim_cos_sim_dist_type_ds_V(z, z_c_sem)_const_embd_512/epoch_id_6"
gdown "https://drive.google.com/file/d/1yyOhKWvPm3-M_Kk8i12VxM8UN-L7aGCr/view?usp=sharing" \
  -O "logs/dreamer_dubins/PyHJ/sim_cos_sim_dist_type_ds_V(z, z_c_sem)_const_embd_512/epoch_id_6/policy.pth"
```

## 🎯 Threshold Calibration

Run conformal prediction script
```bash
python scripts/dubins_cp.py
```
Update desired value in **`configs.yaml`**

## 📊 Evaluation Tools

Run for analyzing world model
```bash
python scripts/wm_analysis.py # Creates wandb run with visualizations
```

Evaluate value funtion and policy
```bash
python scripts/eval_dubins_ddpg_wm.py # Saves results as metrics.txt
```

## ⚖️ Baselines
For Privileged Safe:
```bash
# Reachability training
python scripts/run_training_sac_nodist.py

# Evaluation
python scripts/eval_dubins_sac.py
```

For Latent Safe:
```bash
# Follow dataset generation and WM training from above

# Reachability training
python scripts/run_training_ddpg_wm.py --env-dist-type v --safety-margin-type learned --safety-margin-threshold 0
```


## 🙏 Acknowledgements

This implementation builds on the following open-source projects:

1. [dreamerv3-pytorch](https://github.com/NM512/dreamerv3-torch)
2. [HJReachability](https://github.com/HJReachability/safety_rl/)
3. [latent-safety](https://github.com/CMU-IntentLab/latent-safety.git)
4. [UNISafe](https://github.com/CMU-IntentLab/UNISafe.git)

If you build upon this work, please consider citing our research.

📄 Citation

```
@article{agrawal2025anysafe,
  title={AnySafe: Adapting Latent Safety Filters at Runtime via Safety Constraint Parameterization in the Latent Space},
  author={Agrawal, Sankalp and Seo, Junwon and Nakamura, Kensuke and Tian, Ran and Bajcsy, Andrea},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
