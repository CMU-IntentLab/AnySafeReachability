<div align="center">
    <h1>
        <span style="color: #ff9500; font-style: italic; font-weight: bold;">
            AnySafe:
        </span>
        Adapting Latent Safety Filters at Runtime via Safety Constraint Parameterization in the Latent Space (ICRA, 2026)
    </h1>
    <a href="https://any-safe.github.io/">Homepage</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://arxiv.org/abs/2509.19555">Paper</a>
    <!-- <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="[VIDEO URL]">Video</a> -->
    <br />
</div>

---

This is the official repository for [**AnySafe: Adapting Latent Safety Filters at Runtime via Safety Constraint Parameterization in the Latent Space**](https://any-safe.github.io/).

<p align="center">
 <img width="1200" src="main.png" style="background-color:white;" alt="Flow Diagram">
</p>

---

## 🗂️ Code Structure

```bash
git clone https://github.com/CMU-IntentLab/AnySafeReachability.git
cd AnySafeReachability
```

The project is organized into separate branches:

* **`dubins`**: 3D Dubins Car. [Link]()

```bash
git checkout dubins
```

* **`franka`**: Implementation for real world experiment with Franka Panda arm. [Link]()

```bash
git checkout franka
```

This repository provides the implementation of **Constraint-Conditioned Latent Safety Filters** for adapting safety behavior at runtime in robotics tasks both in simulation and hardware.

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/CMU-IntentLab/AnySafeReachability.git
git checkout franka
cd AnySafe Reachability

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate anysafe
```

---

# Full AnySafe Pipeline
## 🌍 Step 1: Train World Model
1. Train decoder
```bash
python train_dino_decoder.py
```
The best decoder model is saved as `checkpoints/testing_decoder.pth`

2. Train transistion model
```bash
python train_dino_wm.py
```
The best transistion model is saved as `checkpoints/best_testing.pth`

## 🧩 Step 2: Train Semantic Encoder
```bash
python dino_wm/train_failure_classifier.py
```
The best transistion model is saved as `checkpoints_sem/encoder_priv.pth`

You can download the pre-trained world model for the "Sweeper" task: [sweeper model](LINK).
```bash
# Download sweeper world model with semantic encoder
pip install gdown
gdown LINK
```

## 🛡️ Step 3: Constraint-Conditioned Reachability
```bash
python scripts/run_training_ddpg-dinowm.py # save policy
```

## 🎯 Step 4: Conformal Prediction
```bash
python scripts/sweeper_cp.py
```

## 📈 Evaluation Tools
Coming soon!


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