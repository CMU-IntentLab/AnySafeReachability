import argparse
import os
import random
import sys

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import *
from utils import load_state_dict_flexible

from dino_wm.models.dino_models import VideoTransformer, select_xyyaw_from_state
from dino_wm.utils.test_loader import SplitTrajectoryDataset

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
    ]
)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


def make_parser():
    parser = argparse.ArgumentParser()
    # export directory, training and val datasets, test datasets
    parser.add_argument(
        "--embedding-size",
        default=512,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch-size",
        default=150,
        type=int,
        dest="sz_batch",
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        default=60,
        type=int,
        dest="nb_epochs",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gpu-id", default=0, type=int, help="ID of GPU that is used for training."
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument("--loss", default="priv", help="Criterion for training")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--lr-decay-step", default=10, type=int, help="Learning decay step setting"
    )
    parser.add_argument(
        "--lr-decay-gamma", default=0.5, type=float, help="Learning decay gamma setting"
    )
    parser.add_argument(
        "--dont-save-model",
        dest="save_model",
        action="store_false",
        help="Don't save model",
    )
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Dataset Loader and Sampler
BS = args.sz_batch  # batch size
BL = 1

hdf5_file = "/home/sunny/data/sweeper/train/consolidated.h5"
hdf5_file_test = "/home/sunny/data/sweeper/test/consolidated.h5"

if args.use_unlabeled_data:
    train_data_unlabeled = SplitTrajectoryDataset(
        hdf5_file,
        BL,
        split="train",
        num_test=0,
        provide_labels=False,  # Unlabeled data
        num_examples_per_class=int(args.num_examples_per_class * args.unlabeled_ratio)
        if args.unlabeled_ratio != -1.0
        else None,
    )
test_data = SplitTrajectoryDataset(
    hdf5_file_test,
    BL,
    split="train",
    num_test=0,
    provide_labels=True,
    num_examples_per_class=None,  # Don't limit number of examples per class for evaluation
)

test_loader = DataLoader(
    test_data, batch_size=BS, shuffle=True, num_workers=args.nb_workers
)

device = "cuda:0"

model = VideoTransformer(
    image_size=(224, 224),
    dim=384,  # DINO feature dimension
    ac_dim=10,  # Action embedding dimension
    state_dim=3,  # State dimension
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=3,
    dropout=0.1,
    # nb_classes=nb_classes,
).to(device)
load_state_dict_flexible(model, "../checkpoints_sem/encoder_priv.pth")

model.eval()
X = []
y_priv = []

radius = 40


def mapping_fn(X):
    # Maps a distance to a cosine similarity
    # Distance of 1.0 -> cosine sim of -1.0
    # Distance of 0.0 -> cosine sim of 1.0
    return -2 * (X / 250) + 1


with torch.no_grad():
    pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Evaluation",
        position=2,
        leave=False,
    )
    for batch_idx, data in pbar:
        labels_gt = data["failure"][:, -1:].to(device, dtype=torch.float32)  # [B, 2]
        # TODO: look into masking if needed

        inputs1 = (  # [B, 1, 256, 384]
            data["cam_zed_embd"][:, -1:].to(device)
        )
        states = select_xyyaw_from_state(data["state"][:, -1:]).to(device)  # [B, 1, 3]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            semantic_features = model.semantic_embed(  # [embedding_dim]
                inp1=inputs1, state=states
            )

        # Normalize all vectors for cosine similarity
        semantic_features_norm = F.normalize(semantic_features, dim=-1)  # (10, 1, 512)

        # Broadcastable shapes: (10, 1, 1, 512) and (1, 1, 2, 512)
        semantic_features_exp = einops.rearrange(
            semantic_features_norm, "B T Z -> B T 1 Z"
        )  # (BS, T, 1, 512)

        X.append(semantic_features.cpu().numpy())
        y_priv.append(labels_gt.cpu().numpy())

    X = einops.rearrange(np.concatenate(X, axis=0), "B T Z -> (B T) Z")
    y_priv = np.concatenate(y_priv, axis=0).squeeze()

    # Cosine similarity
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos_sim_matrix = X_norm @ X_norm.T

    # Euclidean distance
    diffs = y_priv[:, None] - y_priv[None, :]  # shape (N, N, 2)
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    # Mask away self similarity and double comparison
    mask = np.triu(np.ones(cos_sim_matrix.shape), k=1).astype(bool)

    cos_sim_matrix = cos_sim_matrix[mask]
    dist_matrix = dist_matrix[mask]

    # Find pairs with distance less than radius
    cos_sim_matrix = cos_sim_matrix[dist_matrix < radius]
    cos_sim_matrix = np.clip(cos_sim_matrix, -1.0, 1.0)
    cos_sim_matrix = cos_sim_matrix.astype(np.float32)

    # Find 1-alpha percentile of cosine similarities
    alphas = [0.10, 0.05, 0.01]
    naive_threshold = mapping_fn(radius)
    print(f"Thresholds for radius {radius} with naive threshold: {-naive_threshold}")
    for alpha in alphas:
        threshold = np.percentile(-cos_sim_matrix, 100 * (1 - alpha))
        print(f"Alpha: {alpha}, Threshold: {threshold}")
