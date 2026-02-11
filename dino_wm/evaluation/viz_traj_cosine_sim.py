import copy
import os
import random
import sys

import einops
import h5py
import imageio
import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from dino_wm.test_loader import SplitTrajectoryDataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
import os
import sys

from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from tqdm import *
from tqdm import tqdm
from utils import load_state_dict_flexible

from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

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


def evaluate_V(policy, latent, constraint, action, device):
    constraint = einops.repeat(
        torch.concat((constraint, torch.tensor([1.0], device=device))),
        "C -> B C",
        B=1,
    )  # 1 indicates constraint is active

    obs = {
        "state": latent[:, [-1]].mean(dim=2).reshape(1, -1),
        "constraints": constraint,
    }
    return (
        policy.critic(
            obs=obs,
            act=normalize_acs(
                action.to(device).unsqueeze(0)  # Next action
            ),
        )
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )


import imageio


def make_comparison_video(
    output_dict, keys_to_plot=None, save_path="output_video.mp4", fps=5
):
    """
    Creates a video comparing ground truth and imagination rollouts over time.
    Now supports custom keys to plot and evenly spaced rainbow colors.

    Parameters:
        output_dict (dict): Must include "ground_truth" and "imagination" with rollouts and constraint images.
        keys_to_plot (list or None): List of keys to plot. Default is all relevant keys.
        save_path (str): Output path.
        fps (int): Frames per second.
    """
    output = output_dict

    # Default keys if not provided
    all_keys = [
        "pred_fail",
        "cosine_sim_prox",
        "const1_cos_sim",
        "const2_cos_sim",
        "value_fn",
        "value_fn_ken",
        "gt_fail_label",
    ]
    if keys_to_plot is None:
        keys_to_plot = all_keys

    # Tanh activation on selected outputs
    for key in ["ground_truth", "imagination"]:
        for subkey in (
            ["pred_fail"]  # , "const1_cos_sim", "const2_cos_sim"]
            + [f"class_{_class}_prox" for _class in range(nb_classes)]
            + [f"class_{_class}_logit" for _class in range(nb_classes)]
        ):
            if subkey in output[key]:
                output[key][subkey] = np.tanh(
                    3 * np.array(output[key][subkey]).squeeze()
                )

    T = len(output["ground_truth"]["pred_fail"])

    lengths = {
        k: {s: {len(output[k][s])} for s in keys_to_plot}
        for k in ["ground_truth", "imagination"]
    }
    # Consistency check
    assert all(
        len(output[k][s]) == T
        for k in ["ground_truth", "imagination"]
        for s in keys_to_plot
    ), f"Inconsistent sequence lengths, received lengths: {lengths}"

    # Setup figure
    fig = plt.figure(figsize=(12, 8), dpi=100)
    plt.subplots_adjust(top=0.95)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.05)  # match image pixel height

    # Graph axes
    gt_graph_ax = fig.add_subplot(gs[0, 0:3])
    im_graph_ax = fig.add_subplot(gs[1, 0:3])

    # Generate colors from rainbow colormap
    cmap = mpl.colormaps["rainbow"]
    colors = (
        [cmap(i / (len(keys_to_plot) - 1)) for i in range(len(keys_to_plot))]
        if len(keys_to_plot) > 1
        else [cmap(0.5)]
    )

    # Image axes
    def init_img(ax, title, color=None):
        img_obj = ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        # add vertical lines at each third
        for x in x_class_boundaries:
            if x <= 0 or x >= 224:
                continue
            ax.axvline(x=x, color="black", linestyle="--", linewidth=1, alpha=0.5)
        for y in y_class_boundaries:
            if y <= 0 or y >= 224:
                continue
            ax.axhline(y=y, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(title)

        if color is not None:
            # Get the extent of the image (left, right, bottom, top)
            extent = img_obj.get_extent()

            # Outline thickness in data units (positive = outside, negative = inside)
            thickness = 3  # adjust for thickness

            # Colored outline (inside)
            rect = Rectangle(
                (extent[0] + thickness, extent[3] + thickness),  # bottom-left corner
                224 - 2 * thickness,  # width
                224 - 2 * thickness,  # height
                linewidth=thickness,
                edgecolor=color,
                facecolor="none",
            )
            # rect = Rectangle(
            #     (0, 0),  # bottom-left corner
            #     80,  # width
            #     80,  # height
            #     linewidth=thickness,
            #     edgecolor=color,
            #     facecolor=color,
            # )
            ax.add_patch(rect)

        ax.axis("off")
        return img_obj

    const1_color = (
        colors[keys_to_plot.index("const1_cos_sim")]
        if "const1_cos_sim" in keys_to_plot
        else None
    )
    const2_color = (
        colors[keys_to_plot.index("const2_cos_sim")]
        if "const2_cos_sim" in keys_to_plot
        else None
    )
    gt_front_img_ax = fig.add_subplot(gs[0, 3])
    gt_front_img = init_img(gt_front_img_ax, "Front View")
    gt_const1_img = init_img(
        fig.add_subplot(gs[0, 4]), "Constraint 1", color=const1_color
    )
    gt_const2_img = init_img(
        fig.add_subplot(gs[0, 5]), "Constraint 2", color=const2_color
    )

    im_front_img = init_img(fig.add_subplot(gs[1, 3]), "Front View")
    im_const1_img = init_img(
        fig.add_subplot(gs[1, 4]), "Constraint 1", color=const1_color
    )
    im_const2_img = init_img(
        fig.add_subplot(gs[1, 5]), "Constraint 2", color=const2_color
    )

    gt_label_rect = Rectangle(
        (0, 224),  # bottom-left corner
        224 // 3,  # width
        224 // 3,  # height
        linewidth=3,
        edgecolor=const1_color,
        facecolor="none",
    )
    gt_front_img_ax.add_patch(gt_label_rect)

    # Create legend handles
    legend_handles = []
    for key, color in zip(keys_to_plot, colors):
        label = key.replace("_", " ").title()
        handle = gt_graph_ax.plot([], [], color=color, label=label)[0]
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )

    # Initialize line objects
    def init_lines(ax):
        lines = {}
        for key, col in zip(keys_to_plot, colors):
            if key == "gt_fail_label":
                # Special case for gt_fail_label
                lines[key] = ax.plot([], [], color=col, linestyle="--", label=key)[0]
            else:
                lines[key] = ax.plot([], [], color=col, label=key)[0]
        return lines

    gt_lines = init_lines(gt_graph_ax)
    im_lines = init_lines(im_graph_ax)

    # Graph axes formatting
    for ax, title in zip(
        [gt_graph_ax, im_graph_ax],
        ["Ground Truth Graph", f"Imagination Graph (H = {EVAL_H})"],
    ):
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, T)
        ax.set_ylabel("l(z)")
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.set_box_aspect(224 / (3 * 224))  # 1 for square

    # Add vertical lines on imagine graph every EVAL_H timesteps
    for x in range(BL - 1, T, EVAL_H):
        im_graph_ax.axvline(x, color="gray", linestyle="--", linewidth=0.5)

    # Add horizontal lines for each class
    for key in keys_to_plot:
        if key in [f"class_{_class}_prox" for _class in range(nb_classes)]:
            y = -np.tanh(
                2 * transition.thresholds[keys_to_plot.index(key)].cpu().numpy()
            )
            gt_graph_ax.axhline(
                y, color=colors[keys_to_plot.index(key)], linestyle="--", linewidth=0.5
            )
            im_graph_ax.axhline(
                y, color=colors[keys_to_plot.index(key)], linestyle="--", linewidth=0.5
            )

    # add vertical lines on ground truth graph every label transition
    transitions = (
        np.where(np.diff(output["ground_truth"]["gt_fail_label"], axis=0) != 0)[0] + 1
    )
    for x in transitions:
        for graph in [gt_graph_ax, im_graph_ax]:
            graph.axvline(
                x,
                color=colors[keys_to_plot.index("gt_fail_label")]
                if "gt_fail_label" in keys_to_plot
                else "gray",
                linestyle="--",
                linewidth=1.0,
            )

    # Prepare images
    def prepare_img(img):
        if img.dtype == np.float16:
            img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        return img.astype(np.uint8)

    time = np.arange(T)
    frames = []

    # Generate frames
    for t in tqdm(range(T), desc="Generating Frames"):
        t_slice = slice(t + 1)

        # Graph updates
        for lines, key in [(gt_lines, "ground_truth"), (im_lines, "imagination")]:
            for k in keys_to_plot:
                lines[k].set_data(time[t_slice], output[key][k][t_slice])

        # Image updates
        gt_front_img.set_data(prepare_img(output["ground_truth"]["imgs_front"][t]))
        gt_const1_img.set_data(
            prepare_img(output["ground_truth"]["img_constraint1"][0])
        )
        gt_const2_img.set_data(
            prepare_img(output["ground_truth"]["img_constraint2"][0])
        )

        im_front_img.set_data(prepare_img(output["imagination"]["imgs_front"][t]))
        im_const1_img.set_data(prepare_img(output["imagination"]["img_constraint1"][0]))
        im_const2_img.set_data(prepare_img(output["imagination"]["img_constraint2"][0]))

        label = int(output["ground_truth"]["gt_fail_label"][t] * nb_classes)
        if label != -1 and f"class_{label}_prox" in keys_to_plot:
            label_y = label % (len(y_class_boundaries) - 1)
            label_x = label // (len(y_class_boundaries) - 1)

            gt_label_rect.set_xy(
                (x_class_boundaries[label_x], y_class_boundaries[label_y])
            )
            gt_label_rect.set_width(
                x_class_boundaries[label_x + 1] - x_class_boundaries[label_x] + 1
            )
            gt_label_rect.set_height(
                y_class_boundaries[label_y + 1] - y_class_boundaries[label_y] + 1
            )
            gt_label_rect.set_edgecolor(
                colors[keys_to_plot.index(f"class_{label}_prox")]
            )
            gt_label_rect.set_visible(True)
        else:
            gt_label_rect.set_visible(False)

        # Render
        canvas.draw()
        renderer = canvas.get_renderer()
        buf = np.asarray(renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())

    # Save video/gif
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved to {save_path}")


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

# labels is a tensor of shape (B, 2)
x_class_boundaries = np.array(
    [0, 224 // 3, 224 * 2 // 3, 224]
)  # x boundaries for 3 classes
y_class_boundaries = np.array(
    [224 // 3, 224 * 2 // 3, 224]
)  # y boundaries for 3 classes
# y_class_boundaries = np.array([0, 224])  # y boundaries for 3 classes
# 3 * 2 = 6 classes in total
nb_classes = (len(x_class_boundaries) - 1) * (len(y_class_boundaries) - 1)
label_to_str = {
    0: "Left Top",
    1: "Left Bottom",
    2: "Middle Top",
    3: "Middle Bottom",
    4: "Right Top",
    5: "Right Bottom",
}
# label_to_str = {
#     0: "Left",
#     1: "Middle",
#     2: "Right",
# }
cmap = plt.cm.rainbow
class_to_colors = {i: cmap(i / nb_classes) for i in range(nb_classes)}


def get_class_from_xy(labels):
    assert labels.shape[-1] == 2, (
        "Labels should have shape (B, 2), got shape {}".format(labels.shape)
    )
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)  # Convert to (1, 2) if single label
    assert labels.ndim == 2, "Labels should be a 2D tensor"
    labels = labels.to(device)
    x_labels = torch.bucketize(
        labels[..., 0], torch.tensor(x_class_boundaries, device=device)
    ).unsqueeze(1)
    y_labels = torch.bucketize(
        labels[..., 1], torch.tensor(y_class_boundaries, device=device)
    ).unsqueeze(1)

    class_labels = (x_labels - 1) * (len(y_class_boundaries) - 1) + (y_labels - 1)
    class_labels[torch.logical_or(x_labels <= 0, y_labels <= 0)] = -1
    class_labels[
        torch.logical_or(
            labels[..., 0] < x_class_boundaries[0],
            labels[..., 0] >= x_class_boundaries[-1],
        )
    ] = -1
    class_labels[
        torch.logical_or(
            labels[..., 1] < y_class_boundaries[0],
            labels[..., 1] >= y_class_boundaries[-1],
        )
    ] = -1

    return class_labels


if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda:0"

    hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
    # hdf5_file = "/home/sunny/data/sweeper/proxy_anchor/consolidated.h5"
    database = {}
    with h5py.File(hdf5_file, "r") as hf:
        trajectory_ids = list(hf.keys())
        random.shuffle(trajectory_ids)
        i = 0
        for traj_id in trajectory_ids:
            data = data_from_traj(hf[traj_id])
            if "failure" not in data.keys():
                continue
            database.update({i: data})
            i += 1
            if i > 50:
                break

    constraint_data = SplitTrajectoryDataset(
        "/home/sunny/data/sweeper/proxy_anchor/consolidated.h5",
        3,
        split="train",
        num_test=0,
        only_pass_labeled_examples=True,
    )
    const_data_loader = iter(DataLoader(constraint_data, batch_size=1, shuffle=True))

    BL = 4
    open_loop = True
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
        nb_classes=nb_classes,
    ).to(device)
    # load_state_dict_flexible(
    #     transition,
    #     "../checkpoints_pa/encoder_mrg_0.1_alpha_32_num_ex_all_ul_F.pth",
    # )
    load_state_dict_flexible(
        transition,
        "/home/sunny/AnySafe_Reachability/dino_wm/checkpoints_pa/encoder_priv.pth",
    )
    # load_state_dict_flexible(
    #     transition,
    #     "../checkpoints_pa/encoder_npair_mrg_0.1.pth",
    # )

    # nb_classes = transition.proxies.shape[0]
    # load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

    # transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
    # transition.load_state_dict(torch.load("../checkpoints/best_multi_classifier.pth"))
    transition.eval()

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU

    critic_net = Net(
        state_shape=(397,),
        obs_inputs=["state", "constraint"],
        action_shape=(3,),
        hidden_sizes=[512, 512, 512, 512],
        constraint_dim=512,
        constraint_embedding_dim=512,
        hidden_sizes_constraint=[],
        activation=critic_activation,
        concat=True,
        device=device,
    )

    critic = Critic(critic_net, device=critic_net.device).to(critic_net.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-3)

    from PyHJ.policy import avoid_DDPGPolicy_annealing_dinowm as DDPGPolicy

    print(
        "DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!"
    )

    actor_net = Net(
        (397,),
        obs_inputs=["state", "constraint"],
        hidden_sizes=[512, 512, 512, 512],
        activation=actor_activation,
        device=device,
        constraint_dim=512,
        constraint_embedding_dim=512,
        hidden_sizes_constraint=[],
    )
    actor = Actor(actor_net, (3,), max_action=1, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

    policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=0.005,
        gamma=0.9999,
        exploration_noise=GaussianNoise(sigma=0.1),
        reward_normalization=False,
        estimation_step=1,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=1,
    )
    policy.load_state_dict(
        torch.load(
            "/home/sunny/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy_priv.pth"
        )
    )
    split_policy = copy.deepcopy(policy)
    split_policy.load_state_dict(
        torch.load(
            "/home/sunny/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy_split.pth"
        )
    )

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("../checkpoints/testing_decoder.pth"))
    decoder.eval()

    def randomly_select_constraint(const_data_loader, class_id):
        try:
            data = next(const_data_loader)
        except StopIteration:
            const_data_loader = iter(
                DataLoader(constraint_data, batch_size=1, shuffle=True)
            )
            data = next(const_data_loader)

        if get_class_from_xy(data["failure"][0, -1]) != class_id:
            return randomly_select_constraint(const_data_loader, class_id)

        return data

    # select a random index
    data_const_1 = randomly_select_constraint(const_data_loader, 1)  # 3, 103
    data_const_2 = randomly_select_constraint(const_data_loader, 3)  # 1, 285

    # data_const_1 = {k: v[20:23].unsqueeze(0).to(device) for k, v in database[2].items()}
    # data_const_2 = {
    #     k: v[270:273].unsqueeze(0).to(device) for k, v in database[6].items()
    # }

    constraint1, constraint2 = {}, {}
    for data_const, constraint in [
        (data_const_1, constraint1),
        (data_const_2, constraint2),
    ]:
        constraint.update(
            {
                "front": data_const["agentview_image"][0, -1].to(
                    device
                ),  # [1, 3, 224, 224]
                "inputs1": (  # [1, 1, 256, 384]
                    data_const["cam_zed_embd"][[0], -1:].to(device)
                ),
                "semantic_feat": transition.semantic_embed(
                    inp1=data_const["cam_zed_embd"].to(device),
                    state=select_xyyaw_from_state(data_const["state"]).to(device),
                ).detach()[0, -1],
                "failure": data_const["failure"][0, -1].to(device),
            }
        )  # random class 0 frame

    def proxy_id_to_constraint(proxy_id):
        return torch.append(
            transition.proxies[proxy_id], torch.tensor([1.0], device=device)
        ).unsqueeze(0)

    scale = 1.0

    num_traj = min(10, len(database))
    for traj_id in tqdm(range(num_traj), desc="Processing Trajectories", position=0):
        data = database[traj_id]

        none_list = [-1.0 for _ in range(BL - 1)]

        output_dict = {
            # "imgs_wrist": [],
            "imgs_front": [
                img.cpu().numpy() for img in data["agentview_image"][: BL - 1]
            ],
            "img_constraint1": constraint1["front"].unsqueeze(0).cpu().numpy(),
            "img_constraint2": constraint2["front"].unsqueeze(0).cpu().numpy(),
            "const1_gt_label": copy.deepcopy(none_list),
            "const2_gt_label": copy.deepcopy(none_list),
            "const1_cos_sim": copy.deepcopy(none_list),
            "const2_cos_sim": copy.deepcopy(none_list),
            "const1_value_fn": copy.deepcopy(none_list),
            "const2_value_fn": copy.deepcopy(none_list),
            "pred_fail": copy.deepcopy(none_list),
            "gt_fail_label": get_class_from_xy(data["failure"][:]).cpu().numpy()
            / nb_classes,
            "split_value_fn": copy.deepcopy(none_list),
        }
        for class_id in range(nb_classes):
            output_dict[f"class_{class_id}_logit"] = copy.deepcopy(none_list)
            output_dict[f"class_{class_id}_prox"] = copy.deepcopy(none_list)
            output_dict[f"class_{class_id}_value_fn"] = copy.deepcopy(none_list)
        output = {
            "imagination": copy.deepcopy(output_dict),
            "ground_truth": copy.deepcopy(output_dict),
        }

        traj_length = data["action"].shape[0]

        # Imagination Rollouts
        # inputs1: [1, BL-1, 256, 384], acs: [1, BL-1, 7], states: [1, BL-1, 8]
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        # all_acs: [1 64 A]
        all_acs = data["action"][:].unsqueeze(0).to(device)
        all_acs = normalize_acs(all_acs, device)
        inputs_states = select_xyyaw_from_state(
            data["state"][0 : BL - 1, :].to(device)
        ).unsqueeze(0)

        EVAL_H = 10

        # Imagination Loop
        for t in tqdm(
            range(traj_length - BL + 1),
            desc=f"Imagining Trajectory {traj_id}",
            leave=False,
            position=1,
        ):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    # pred1: [1, H, N, P], pred_state: [1, H, S], pred_fail: [1, H, 1]
                    # semantic_features: [1, H, Z], latent: [1, H, N, (P + A + S)]
                    pred1, pred_state, pred_fail, semantic_features, latent = (
                        transition(
                            inputs1,
                            inputs_states,
                            acs,
                            return_latent=True,
                        )
                    )
                    # pred_labels = transition.multi_class_head(latent)
                    # pred_labels = torch.mean(pred_labels, dim=-2)

                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_features.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T
                    # Decode images
                    # pred_ims: [1, C, H, W] H is height not horizon
                    pred_ims, _ = decoder(pred1[:, [-1]])
                    pred_img1 = pred_ims.permute(0, 2, 3, 1)

                    # pred_img = einops.rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                    # getting next inputs
                # acs: [1 H A]
                if t + BL < len(all_acs[0]):  # if not on last step
                    acs = torch.cat(
                        [
                            acs[[0], 1:],
                            all_acs[0, BL - 1 + t].unsqueeze(0).unsqueeze(0),
                        ],
                        dim=1,
                    )
                # inputs1: [1 H N P]
                # inputs_states: [1 H S]
                if (t + 1) % EVAL_H == 0 or not open_loop:
                    # inputs1: [1, BL-1, 256, 384], acs: [1, BL-1, 7], states: [1, BL-1, 8]
                    inputs1 = (
                        data["cam_zed_embd"][t : t + BL - 1, :].to(device).unsqueeze(0)
                    )
                    inputs_states = select_xyyaw_from_state(
                        data["state"][t : t + BL - 1, :].to(device).unsqueeze(0)
                    )
                else:
                    inputs1 = torch.cat([inputs1[:, 1:], pred1[:, [-1]]], dim=1)
                    states = torch.cat(
                        [inputs_states[:, 1:], pred_state[:, [-1]]], dim=1
                    )

            # pred1_img: [1, H, W, C]
            output["imagination"]["imgs_front"].append(
                pred_img1[0].cpu().numpy() * 255.0,
            )
            output["imagination"]["pred_fail"].append(
                pred_fail.detach().squeeze().cpu().numpy()[-1]
            )
            output["imagination"]["split_value_fn"].append(
                evaluate_V(
                    policy=split_policy,
                    latent=latent,
                    constraint=transition.proxies[0],
                    action=data["action"][t + BL - 1, :],
                    device=device,
                )
            )
            for class_id in range(nb_classes):
                # output["imagination"][f"class_{class_id}_logit"].append(
                #     -pred_labels.detach().squeeze().cpu().numpy()[-1, class_id]
                # )
                output["imagination"][f"class_{class_id}_prox"].append(
                    -cos_sim_matrix[-1, class_id].item()
                    # - transition.thresholds[class_id].item()
                )

            if t + BL >= len(data["action"]):  # Last step
                index = t + BL - 1
            else:
                index = t + BL

            for _class in range(nb_classes):
                output["imagination"][f"class_{_class}_value_fn"].append(
                    evaluate_V(
                        policy=policy,
                        latent=latent,
                        constraint=transition.proxies[_class],
                        action=data["action"][index, :],
                        device=device,
                    )
                )

            for constraint, const_key in [
                (constraint1, "const1"),
                (constraint2, "const2"),
            ]:
                output["imagination"][f"{const_key}_cos_sim"].append(
                    -F.cosine_similarity(
                        semantic_features.squeeze()[-1],
                        constraint["semantic_feat"],
                        dim=0,
                    ).item()
                    # + 0.3
                )
                # Debugging: What is cosine similarity with proxy and constraint?
                F.cosine_similarity(
                    transition.proxies[:],
                    constraint["semantic_feat"].unsqueeze(0),
                )

                output["imagination"][f"{const_key}_value_fn"].append(
                    evaluate_V(
                        policy=policy,
                        latent=latent,
                        constraint=constraint["semantic_feat"],
                        action=data["action"][index, :],
                        device=device,
                    )
                )

                dist_const = torch.norm(
                    data["failure"][BL - 1 + t].cpu() - constraint["failure"].cpu(),
                    dim=-1,
                ).item()

                dist_const = -2 * (dist_const / 250) + 1

                output["imagination"][f"{const_key}_gt_label"].append(-dist_const)

        lengths = [
            len(output["imagination"][key]) for key in output["imagination"].keys()
        ]

        # Do ground truth images
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        states = select_xyyaw_from_state(
            data["state"][0 : BL - 1, :].to(device).unsqueeze(0)
        )

        output["ground_truth"]["imgs_front"] = [
            img for img in data["agentview_image"][:].cpu().numpy()
        ]

        for t in tqdm(
            range(traj_length - BL + 1), desc="GT Trajectory", position=1, leave=False
        ):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    # semantic_features: [1, (T-1), Z]
                    assert inputs1.shape == (1, BL - 1, 256, 384), (
                        f"Inputs1 shape mismatch, got {inputs1.shape}"
                    )
                    assert states.shape == (1, BL - 1, 3), (
                        f"States shape mismatch, got {states.shape}"
                    )
                    semantic_features = transition.semantic_embed(
                        inp1=inputs1, state=states
                    )

                    # latent: [1, (T-1), N, (P + A + S)]
                    latent = transition.forward_features(
                        video1=inputs1, states=states, actions=acs
                    )

                    # pred_fail: [1, (T-1), 1]
                    pred_fail = transition.fail_pred(inp1=inputs1, state=states)

                    # pred_labels: [1, (T-1), num_classes]
                    # pred_labels = transition.multi_class_head(latent)
                    # pred_labels = torch.mean(pred_labels, dim=-2)

                    # Calculate cos sim for failure margin
                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_features.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T

            # inputs2 = data["cam_rs_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            inputs1 = data["cam_zed_embd"][t : t + BL - 1, :].to(device).unsqueeze(0)
            acs = data["action"][t : t + BL - 1, :].to(device).unsqueeze(0)
            acs = normalize_acs(acs, device=device)
            states = select_xyyaw_from_state(
                data["state"][t : t + BL - 1, :].to(device)
            ).unsqueeze(0)

            # pred_fail: [1 (T-1), 1] -> [1]
            pred_fail = pred_fail.squeeze().cpu().numpy()[-1]

            # output["ground_truth"]["imgs_wrist"].append(
            #     data["robot0_eye_in_hand_image"][t + BL - 1]
            #     .unsqueeze(0)
            #     .cpu()
            #     .numpy()[-1]
            # )
            output["ground_truth"]["pred_fail"].append(pred_fail)
            output["ground_truth"]["split_value_fn"].append(
                evaluate_V(
                    policy=split_policy,
                    latent=latent,
                    constraint=transition.proxies[0] * 0.0,
                    action=data["action"][t + BL - 1, :],
                    device=device,
                )
            )
            for class_id in range(nb_classes):
                # output["ground_truth"][f"class_{class_id}_logit"].append(
                #     -pred_labels.detach().squeeze().cpu().numpy()[-1, class_id]
                # )
                output["ground_truth"][f"class_{class_id}_prox"].append(
                    -cos_sim_matrix[-1, class_id].item()
                    # - transition.thresholds[class_id].item()
                )

            if t + BL >= len(data["action"]):  # Last step
                index = t + BL - 1
            else:
                index = t + BL

            for _class in range(nb_classes):
                output["ground_truth"][f"class_{_class}_value_fn"].append(
                    evaluate_V(
                        policy=policy,
                        latent=latent,
                        constraint=transition.proxies[_class],
                        action=data["action"][index, :],
                        device=device,
                    )
                )
            for constraint, const_key in [
                (constraint1, "const1"),
                (constraint2, "const2"),
            ]:
                output["ground_truth"][f"{const_key}_cos_sim"].append(
                    -F.cosine_similarity(
                        semantic_features.squeeze()[-1],
                        constraint["semantic_feat"],
                        dim=0,
                    ).item()
                    # + 0.3
                )

                # Sanity Check: What is cosine similarity with proxy and constraint?
                F.cosine_similarity(
                    transition.proxies[2].unsqueeze(0),
                    constraint["semantic_feat"].unsqueeze(0),
                )

                output["ground_truth"][f"{const_key}_value_fn"].append(
                    evaluate_V(
                        policy=policy,
                        latent=latent,
                        constraint=constraint["semantic_feat"],
                        action=data["action"][index, :],
                        device=device,
                    )
                )

                dist_const = torch.norm(
                    data["failure"][BL - 1 + t].cpu() - constraint["failure"].cpu(),
                    dim=-1,
                ).item()

                dist_const = -2 * (dist_const / 250) + 1

                output["ground_truth"][f"{const_key}_gt_label"].append(-dist_const)
            # output["ground_truth"]["cosine_sim_prox"].append(cos_sim_fail * scale)
            # output["ground_truth"]["value_fn"].append(
            #     policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=acs,  # Next action
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # output["ground_truth"]["value_fn_ken"].append(
            #     ken_policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=acs,  # Next action
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # output["ground_truth"]["gt_fail_label"].append(
            #     -2 * data["failure"][t + BL - 1].cpu().numpy() + 1
            # )

        line_keys = [
            # "pred_fail",
            # "cosine_sim_prox",
            "const1_cos_sim",
            # "const2_cos_sim",
            "const1_gt_label",
            # "const2_gt_label",
            # "value_fn_ken",
            # "gt_fail_label",
            # "split_value_fn",
            "const1_value_fn",
            # "const2_value_fn",
        ]
        for _class in range(nb_classes):
            # line_keys.append(f"class_{_class}_logit")
            # line_keys.append(f"class_{_class}_prox")
            # line_keys.append(f"class_{_class}_value_fn")

            1 + 1

        # line_keys.append(f"class_{5}_prox")

        make_comparison_video(
            output_dict=output,
            save_path=f"results/output_video_{traj_id}.mp4",
            fps=10,
            keys_to_plot=line_keys,
        )
