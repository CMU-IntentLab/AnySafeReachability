import os
import random
import sys

import h5py
import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state

# Add directories to system path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
        os.path.join(base_dir, "model_based_irl_torch"),
        os.path.join(base_dir, "real_envs"),
    ]
)

# Load model
print(sys.path)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")


def transition_from_data(data, transition, device, use_amp=True):
    data1 = data["cam_zed_embd"].to(device)
    data2 = data["cam_rs_embd"].to(device)

    inputs1 = data1[:, :-1]
    inputs2 = data2[:, :-1]

    states = data["state"].to(device)[:, :-1]
    acs = normalize_acs(data["action"].to(device)[:, :-1], device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
        with torch.no_grad():
            pred1, pred2, pred_state, pred_fail, semantic_feat = transition(
                inputs1, inputs2, states, acs
            )

    return pred1, pred2, pred_state, pred_fail, semantic_feat


def data_from_traj(traj):
    data = {}
    segment_length = traj["actions"].shape[0]
    # data["robot0_eye_in_hand_image"] = torch.tensor(
    #     np.array(traj["camera_0"][:]) * 255.0, dtype=torch.uint8
    # )
    data["agentview_image"] = torch.tensor(
        np.array(traj["camera_1"][:]) * 255.0, dtype=torch.uint8
    )
    # data["cam_rs_embd"] = torch.tensor(
    #     np.array(traj["cam_rs_embd"][:]), dtype=torch.float32
    # )
    data["cam_zed_embd"] = torch.tensor(
        np.array(traj["cam_zed_embd"][:]), dtype=torch.float32
    )
    data["state"] = torch.tensor(np.array(traj["states"][:]), dtype=torch.float32)
    data["action"] = torch.tensor(np.array(traj["actions"][:]), dtype=torch.float32)
    if "labels" in traj.keys():
        data["failure"] = torch.tensor(np.array(traj["labels"][:]), dtype=torch.float32)
    data["is_first"] = torch.zeros(segment_length)
    data["is_last"] = torch.zeros(segment_length)
    data["is_first"][0] = 1.0
    data["is_terminal"] = data["is_last"]
    data["discount"] = torch.ones(segment_length, dtype=torch.float32)
    return data


def confusion(pred, fail_data):
    safe_data = torch.where(fail_data == 0.0)
    unsafe_data = torch.where(fail_data == 1.0)

    pos = pred[safe_data]
    neg = pred[unsafe_data]

    TP = torch.sum(pos > 0).item()
    FN = torch.sum(pos < 0).item()
    FP = torch.sum(neg > 0).item()
    TN = torch.sum(neg < 0).item()

    return torch.tensor([TP, FN, FP, TN])


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform1 = transforms.Compose(
    [
        transforms.Resize(520),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda:0"

hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
database = {}
with h5py.File(hdf5_file, "r") as hf:
    trajectory_ids = list(hf.keys())
    database = {
        i: data_from_traj(hf[traj_id]) for i, traj_id in enumerate(trajectory_ids)
    }

BL = 4
EVAL_H = 13
open_loop = True
use_proxy_anchor = True
transition = VideoTransformer(
    image_size=(224, 224),
    dim=384,
    ac_dim=10,
    state_dim=3,
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=BL - 1,
    dropout=0.1,
).to(device)
# load_state_dict_flexible(transition, "../checkpoints_pa/encoder_0.1.pth")
# load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

# transition.load_state_dict(
#     torch.load("../checkpoints/best_classifier.pth"), strict=False
# )
transition.load_state_dict(
    torch.load("checkpoints_pa/encoder_mrg_0.1_alpha_32_num_ex_all_ul_F.pth"),
    strict=False,
)

transition.eval()

decoder = VQVAE().to(device)
decoder.load_state_dict(torch.load("checkpoints/testing_decoder.pth"))
decoder.eval()

scale = 1.0
# Define your table as a list of rows
confusion_matrices = {
    k: torch.zeros(4, dtype=torch.int64, device=device) for k in range(3)
}

bars = [
    tqdm(total=0, bar_format="{desc}", position=2, leave=True),
    tqdm(total=0, bar_format="{desc}", position=3, leave=True),
    tqdm(total=0, bar_format="{desc}", position=4, leave=True),
]

num_traj = min(10, len(database))  # Limit to 10 trajectories for testing
for traj_id in tqdm(range(num_traj), desc="Processing Trajectories", position=0):
    data = database[traj_id]

    none_list = [-1.0 for _ in range(BL - 1)]

    traj_length = data["action"].shape[0]
    # all_acs: [1 64 A]
    all_acs = data["action"][:].unsqueeze(0).to(device)
    all_acs = normalize_acs(all_acs, device)

    for start_idx in tqdm(
        range(0, traj_length - EVAL_H + 1),
        desc="Processing Segments",
        position=1,
    ):
        # Imagination Rollouts
        # inputs1: [1, BL-1, 256, 384], acs: [1, BL-1, 7], states: [1, BL-1, 8]
        inputs1 = (
            data["cam_zed_embd"][start_idx : start_idx + BL - 1, :]
            .to(device)
            .unsqueeze(0)
        )
        acs = data["action"][start_idx : start_idx + BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        inputs_states = select_xyyaw_from_state(
            data["state"][start_idx : start_idx + BL - 1, :].to(device)
        ).unsqueeze(0)

        pred_fails = []

        # Imagination Loop
        for t in range(EVAL_H - BL + 1):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    # pred1: [1, EVAL_H, N, P], pred_state: [1, EVAL_H, S], pred_fail: [1, EVAL_H, 1]
                    # semantic_features: [1, EVAL_H, Z], latent: [1, EVAL_H, N, (P + A + S)]
                    pred1, pred_state, ___, semantic_features, latent = transition(
                        inputs1,
                        # wrist_hist,
                        inputs_states,
                        acs,
                        return_latent=True,
                    )
                    if use_proxy_anchor:
                        proxies = transition.proxies.to(device)  # [M Z]

                        queries_norm = F.normalize(
                            semantic_features, p=2, dim=-1
                        )  # [N, T, Z]
                        proxies_norm = F.normalize(proxies, p=2, dim=-1)  # [M, Z]

                        # Compute cosine similarity
                        # [N, T, Z] @ [Z M] -> [N, T, M]
                        pred_fail = -queries_norm @ proxies_norm.T
                    else:
                        pred_fail = transition.multi_class_head(latent)

                    pred_fails.append(pred_fail.squeeze().cpu().numpy())

                # getting next inputs
                # acs: [1 EVAL_H A]
                if start_idx + t + BL < len(all_acs[0]):  # if not on last step
                    acs = torch.cat(
                        [
                            acs[[0], 1:],
                            all_acs[0, start_idx + BL - 1 + t]
                            .unsqueeze(0)
                            .unsqueeze(0),
                        ],
                        dim=1,
                    )
                # inputs1: [1 EVAL_H N P]
                # inputs_states: [1 EVAL_H S]
                if open_loop:
                    inputs1 = torch.cat(
                        [inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1
                    )
                    states = torch.cat(
                        [
                            inputs_states[[0], 1:],
                            pred_state[:, -1].unsqueeze(1),
                        ],
                        dim=1,
                    )
                else:
                    inputs1 = (
                        data["cam_zed_embd"][t : start_idx + t + BL - 1, :]
                        .to(device)
                        .unsqueeze(0)
                    )
                    inputs_states = select_xyyaw_from_state(
                        data["state"][t : start_idx + t + BL - 1, :]
                        .to(device)
                        .unsqueeze(0)
                    )

        pred_fails = torch.tensor(
            pred_fails, device=device
        )  # [(EVAL_H-BL+1), T, nb_classes]
        gt_labels = (data["failure"][start_idx + BL - 1 : start_idx + EVAL_H]).numpy()

        for _class in range(3):
            # pred_fail_class: [EVAL_H-BL+1] Only doing last timestep for now
            pred_fail_class = pred_fails[..., -1, _class]
            gt_labels_class = torch.tensor((gt_labels - 1) == _class, device=device)

            confusion_matrices[_class] += confusion(
                pred_fail_class, gt_labels_class
            ).to(device)
            bars[_class].set_description_str(
                f"Class {_class}: {confusion_matrices[_class].reshape((2, 2)).T.tolist()}"
            )

# Aggregate metrics
for _class, values in confusion_matrices.items():
    TP, FN, FP, TN = values.tolist()
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TPR
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Class {_class} Metrics:")
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    print(f"FPR: {FPR:.4f}, FNR: {FNR:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n"
    )
