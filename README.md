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
```

## 🗂️ Dataset Generation

Generate synthetic expert and random trajectory data for training the world model.

```bash
python scripts/generate_data_traj_cont.py
```

Generates ... file.


## 🧠 World Model Training

Train the latent dynamics world model using the generated dataset.

```bash
python scripts/dreamer_offline.py
```

Produces ... file.

Download pre-trained world model...
```bash
pip install gdown
gdown ...
```


## 🛡️ Constraint-Conditioned Latent Reachability Analysis

Run RL for constraint-conditioned latent-space reachability:

```bash
python scripts/run_training_ddpg-wm.py 
```

## 📊 Evaluation Tools

Run for analyzing world model
```bash
python scripts/wm_analysis.py 
```

## ⚖️ Baselines
For Privileged Safe:
```bash
# Reachability training
python scripts/run_training_sac_nodist.py # Generates ... file

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
  journal={arXiv preprint arXiv:2509.19555},
  year={2025}
}
```
