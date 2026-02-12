import os
import random
import sys

import einops
import h5py
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
from dino_wm.models.dino_decoder import VQVAE
from dino_wm.models.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from torchvision import transforms
from tqdm import tqdm

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
    data["state"] = torch.tensor(
        np.array(select_xyyaw_from_state(traj["states"][:])), dtype=torch.float32
    )
    data["action"] = torch.tensor(np.array(traj["actions"][:]), dtype=torch.float32)
    if "labels" in traj.keys():
        data["failure"] = torch.tensor(np.array(traj["labels"][:]), dtype=torch.float32)
    data["is_first"] = torch.zeros(segment_length)
    data["is_last"] = torch.zeros(segment_length)
    data["is_first"][0] = 1.0
    data["is_terminal"] = data["is_last"]
    data["discount"] = torch.ones(segment_length, dtype=torch.float32)
    return data


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


def crop_top_middle(image):
    top = 30
    left = 28
    height = 192
    width = 192
    return F.crop(image, top, left, height, width)


crop_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Lambda(lambda img: crop_top_middle(img)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda:0"

    EVAL_H = 32
    hdf5_file = "/data/sunny/sweeper/test/consolidated.h5"
    # hdf5_file = "/home/sunny/data/sweeper/train/optimal/traj_0001.hdf5"
    database = {}
    with h5py.File(hdf5_file, "r") as hf:
        trajectory_ids = list(hf.keys())
        print(f"Total trajectories in dataset: {len(trajectory_ids)}")
        total = 0
        for i, traj_id in enumerate(trajectory_ids):
            if "labels" not in hf[traj_id].keys():
                # print(f"Skipping {traj_id} as it has no labels")
                continue

            labels = data_from_traj(hf[traj_id])["failure"]
            a = labels[:-1]
            b = labels[1:]
            transitions = ((a == -1) & (b != -1)) | ((a != -1) & (b == -1))
            index = transitions.nonzero(as_tuple=True)[0]
            if len(index) == 0:  # No failure transitions, skip
                continue

            if (
                index < (EVAL_H // 2)
            ).all():  # If all the transitions are before EVAL_H, skip
                print(f"Skipping {traj_id} as transitions are too early")
                continue

            if (
                (index + (EVAL_H // 2)) > len(labels)
            ).all():  # If any transition is too close to the end
                print(f"Skipping {traj_id} as transitions are too close to the end")
                continue

            if total > 10:
                print("Reached limit of 10 trajectories with failure transitions")
                break

            database[total] = data_from_traj(hf[traj_id])
            database[total]["index"] = index[index > (EVAL_H // 2)][
                0
            ].item()  # Store the index of the first failure
            total += 1

    print(f"Total trajectories with failure transitions: {len(database)}")

    BL = 4
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
    # load_state_dict_flexible(transition, "../checkpoints_sem/encoder_0.1.pth")
    # load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

    transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
    transition.eval()

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("../checkpoints/testing_decoder.pth"))
    decoder.eval()

    scale = 1.0
    for traj_id in tqdm(
        range(len(database)), desc="Processing Trajectories", position=0
    ):
        save_path = f"results/output_video_{traj_id}.mp4"
        with torch.no_grad():
            H = BL - 1
            data = database[traj_id]
            start_idx = data["index"] - EVAL_H // 2
            # H: 3, EVAL_H: 16
            # eval_data1: [1 EVAL_H N P]
            eval_data1 = data["cam_zed_embd"][start_idx : start_idx + EVAL_H].to(device)
            # inputs1: [1 H N P]
            inputs1 = eval_data1[0:H].unsqueeze(0).to(device)

            # all_acs: [1 EVAL_H A]
            all_acs = (
                data["action"][start_idx : start_idx + EVAL_H].unsqueeze(0).to(device)
            )
            all_acs = normalize_acs(all_acs, device)

            # acs: [1 H A]
            acs = data["action"][start_idx : H + start_idx].unsqueeze(0).to(device)
            acs = normalize_acs(acs, device)

            # inputs_states: [1 H S]
            inputs_states = (
                data["state"][start_idx : H + start_idx].unsqueeze(0).to(device)
            )
            im1s = (
                data["agentview_image"][start_idx : H + start_idx].squeeze().to(device)
                / 255.0
            )
            for k in range(EVAL_H - H):
                # inputs1: [1 H N P], inputs_states: [1 H S], acs: [1 H A]
                pred1, pred_state, _, ___ = transition(inputs1, inputs_states, acs)

                # pred_latent = pred1[:, [-1]]
                pred_ims, _ = decoder(pred1[:, [-1]])

                pred_ims = einops.rearrange(pred_ims, "t c h w -> t h w c", t=1)
                pred_im1 = pred_ims

                im1s = torch.cat([im1s, pred_im1], dim=0)

                # getting next inputs
                # acs: [1 H A]
                acs = torch.cat(
                    [
                        acs[[0], 1:],
                        all_acs[0, H + k].unsqueeze(0).unsqueeze(0),
                    ],
                    dim=1,
                )
                # inputs1: [1 H N P]
                inputs1 = torch.cat(
                    [inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1
                )
                # inputs_states: [1 H S]
                states = torch.cat(
                    [inputs_states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1
                )

            gt_im1 = (
                data["agentview_image"][start_idx : EVAL_H + start_idx]
                .squeeze()
                .to(device)
            )

            gt_imgs = torch.cat([gt_im1], dim=-2) / 255.0  # [T H W C]
            pred_imgs = torch.cat([im1s], dim=-2)

            vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
            vid = vid.detach().cpu().numpy()
            vid = (vid * 255).clip(0, 255).astype(np.uint8)
            # vid = einops.rearrange(vid, "t h w c -> t c h w")

        # Accuracy Metrics
        state_mse = torch.mean((inputs_states - states) ** 2)
        vid_mse = torch.mean(
            (
                im1s
                - data["agentview_image"][start_idx : EVAL_H + start_idx]
                .squeeze()
                .to(device)
                / 255.0
            )
            ** 2
        )

        print(f"State MSE: {state_mse.item():.4f}, Image MSE: {vid_mse.item():.4f}")

        # Save video/gif
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, vid, fps=10)
        print(f"Saved to {save_path}")
        print(f"Saved to {save_path}")
