import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# add to os sys path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../model_based_irl_torch")
)
sys.path.append(dreamer_dir)
env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../real_envs"))
sys.path.append(env_dir)
print(dreamer_dir)
print(sys.path)
from torch.utils.data import DataLoader
from tqdm import tqdm

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

import random

import torch
from dino_decoder import VQVAE
from torchvision import transforms

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
        transforms.CenterCrop(518),  # should be multiple of model patch_size
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)


DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        # transforms.CenterCrop(224), #should be multiple of model patch_size
        transforms.ToTensor(),
    ]
)
norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

import torch
from dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from test_loader import SplitTrajectoryDataset


def fail_loss(pred, fail_data):
    safe_data = torch.where(fail_data == 0.0)
    unsafe_data = torch.where(fail_data == 1.0)
    unsafe_data_weak = torch.where(fail_data == 2.0)

    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]

    gamma = 0.75
    lx_loss = (
        (1 / pos.size(0)) * torch.sum(torch.relu(gamma - pos))
        if pos.size(0) > 0
        else 0.0
    )  # penalizes safe for being negative
    lx_loss += (
        (1 / neg.size(0)) * torch.sum(torch.relu(gamma + neg))
        if neg.size(0) > 0
        else 0.0
    )  # penalizes unsafe for being positive
    lx_loss += (
        (1 / neg_weak.size(0)) * torch.sum(torch.relu(neg_weak))
        if neg_weak.size(0) > 0
        else 0.0
    )  # penalizes unsafe for being positive

    return lx_loss


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


if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16
    BL = 4
    hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
    use_proxy_anchor = True

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))

    device = "cuda:0"
    H = 3

    # decoder = Decoder().to(device)
    # decoder.load_state_dict(torch.load('checkpoints/best_decoder_10m.pth'))
    # decoder.eval()

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=3,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    # transition.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/claude_zero_wfail20500_rotvec.pth'))
    # transition.load_state_dict(torch.load("checkpoints/best_multi_classifier.pth"))
    transition.load_state_dict(
        torch.load("checkpoints_pa/encoder_mrg_0.1_alpha_32_num_ex_all_ul_F.pth")
    )

    decoder = VQVAE().to(device)
    # decoder.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/best_decoder_10m.pth'))
    decoder.load_state_dict(torch.load("checkpoints/testing_decoder.pth"))
    decoder.eval()

    data = next(expert_loader)

    data1 = data[
        "cam_zed_embd"
    ].to(
        device
    )  # [transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
    # data2 =  data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]

    data_state = select_xyyaw_from_state(data["state"].to(device))
    states = data_state[:, :-1]
    output_state = data_state[:, 1:]

    data_acs = data["action"].to(device)
    acs = data_acs[:, :-1]
    acs = normalize_acs(acs, device=device)

    print(data.keys())

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        pred1, pred_state, pred_fail, ___ = transition(inputs1, states, acs)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    best_eval = float("inf")
    best_fail = float("inf")
    iters = []
    eval_iter = 144
    transition.eval()

    confusion_matrices = {
        k: torch.zeros(4, dtype=torch.int64, device=device) for k in range(3)
    }
    for i in tqdm(range(eval_iter), desc="Training", unit="iter"):
        # if i % 30 == 0:
        #    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        #    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        #    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        data1 = data["cam_zed_embd"].to(device)

        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        data_state = select_xyyaw_from_state(data["state"].to(device))
        states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data["action"].to(device)
        acs = data_acs[:, :-1]
        acs = normalize_acs(acs, device)

        with torch.no_grad():
            pred1, pred_state, ___, semantic_feat, latent = transition(
                inputs1, states, acs, return_latent=True
            )
            latent = torch.mean(latent, dim=-2)
            if use_proxy_anchor:
                proxies = transition.proxies.to(device)  # [M Z]

                queries_norm = F.normalize(semantic_feat, p=2, dim=-1)  # [N, T, Z]
                proxies_norm = F.normalize(proxies, p=2, dim=-1)  # [M, Z]

                # Compute cosine similarity
                # [N, T, Z] @ [Z M] -> [N, T, M]
                pred_fail = -queries_norm @ proxies_norm.T
            else:
                pred_fail = transition.multi_class_head(latent)

            gt_labels = data["failure"][:, 1:].cpu().numpy()

        for _class in range(3):
            pred_fail_class = pred_fail[..., _class]
            gt_labels_class = torch.tensor((gt_labels - 1) == _class, device=device)

            confusion_matrices[_class] += confusion(
                pred_fail_class, gt_labels_class
            ).to(device)

        for _class in range(3):
            print(f"Class {_class} Confusion Matrix:")
            print(confusion_matrices[_class].reshape(2, 2).T)

# Get aggregate metrics
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

    print(f"Class {_class} Aggregate Metrics:")
    print(f"FPR: {FPR:.4f}, FNR: {FNR:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}"
    )
