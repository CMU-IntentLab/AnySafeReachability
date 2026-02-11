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

## Code Structure

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

## Installation

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


## Quick Start: Download Pretrained Models and Dataset
You can download pretrained models: [pretrained models](LINK).
```bash
# Download pretrained world model
gdown LINK

# Download pretrained constraint-conditioned filter

# This will create:
# - dreamer.pt (pretrained world model)
# - filter/ (reachability filter directory)
#   └── model/ (filter checkpoints at different training steps)
```

## Train World Model
You can download the pre-collected dataset for the "Sweeper" task: [sweeper dataset](LINK).
```bash
# Download sweeper dataset
pip install gdown
gdown LINK
unzip pretrained_models.zip
```

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


## Train Semantic Encoder
```bash
python train_dino_wm.py
```
The best transistion model is saved as `checkpoints/best_testing.pth`
