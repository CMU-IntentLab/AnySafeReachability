import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms

# Global variables
current_idx = 0
images = []
labels = {}
current_traj = ""


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


def on_key_press(event):
    global current_idx, sep_labels

    # Label images with '0', '1' or '2'
    if event.key in {"0", "1"}:
        labels[current_idx] = int(event.key)
        print(f"Image {current_idx} labeled as {labels[current_idx]}")

        # Move to the next image
        current_idx += 1

        if current_idx < len(images):
            update_plot()
        else:
            print("All images labeled for this trajectory! Close the window to exit.")
            fig.canvas.mpl_disconnect(key_press_cid)
            plt.pause(0.5)
            plt.close(fig)

    # elif event.key == "backspace":
    #     labels[current_idx] = -1.0
    #     if current_idx < len(images):
    #         update_plot()
    #     else:
    #         print("All images labeled for this trajectory! Close the window to exit.")
    #         fig.canvas.mpl_disconnect(key_press_cid)
    #         plt.pause(0.5)
    #         plt.close(fig)

    # Rewind with spacebar
    elif event.key == " ":
        if current_idx > 0:
            current_idx -= 1
            print(
                f"Rewound to image {current_idx}. Current label: {labels.get(current_idx, 'None')}"
            )
            update_plot()


def update_plot():
    global current_idx, images, labels, fig, ax

    ax.clear()
    ax.imshow(images[current_idx])

    width = images[current_idx].shape[1]
    third = width // 3

    # update progress bar
    progress = current_idx / len(images)
    rect = plt.Rectangle((0, 0), int(progress * width), 10, color="salmon")
    ax.add_patch(rect)

    ax.axis("off")
    fig.canvas.draw()


def check_if_labeled(traj_file, label_type):
    """Check if a trajectory file has been labeled."""
    with h5py.File(traj_file, "r") as hf:
        data = hf["data"]
        if label_type in data:
            if data[label_type].shape[0] > 0:
                return True
            else:
                return False
        else:
            return False


def process_trajectory(traj_file):
    """Load images from a trajectory file and set up labels."""
    global images, labels, current_idx, current_traj

    current_traj = os.path.splitext(os.path.basename(traj_file))[0]
    print(f"Processing trajectory: {current_traj}")
    labels = {}

    # Load images
    images = []
    with h5py.File(traj_file, "r") as hf:
        data = hf["data"]
        assert "camera_1" in data, (
            f"Expected 'camera_1' dataset in the HDF5 file {traj_file}."
        )
        for i in range(data["camera_1"][:].shape[0]):
            front = data["camera_1"][i]

            joint = np.concatenate([front], axis=1)
            joint = crop_transform(joint).permute(1, 2, 0)  # Convert to CxHxW format
            images.append(joint)

    # Initialize index
    current_idx = 0
    while current_idx < len(images) and current_idx in labels:
        current_idx += 1


# def process_trajectory_safe(traj_file):
#     """Load images from a trajectory file and set up labels."""
#     global images, labels, current_idx, current_traj

#     current_traj = os.path.splitext(os.path.basename(traj_file))[0]
#     print(f"Processing trajectory: {current_traj}")
#     labels = {}

#     # Load images
#     images = []
#     with h5py.File(traj_file, "r") as hf:
#         data = hf['data']
#         for i in range(data["camera_0"][:].shape[0]):
#             labels[i] = 0


def postprocess_trajectory(traj_file, labels, label_type):
    """Load images from a trajectory file and set up labels."""

    # write to done_file
    with h5py.File(traj_file, "r+") as hf:
        data_group = hf["data"]

        print(f"Assigning labels to {traj_file}.")
        labels = np.array(list(labels.values()))
        print(f"Labels: {labels}")
        print(labels.shape)
        print(data_group["camera_1"].shape)
        if label_type in data_group:
            del data_group[label_type]
        data_group.create_dataset(label_type, data=np.array(labels))


# Initialize the plot
plt.ion()

if __name__ == "__main__":
    directory = "/data/sunny/sweeper/train/optimal"
    label_type = "separated_label"
    reset_regardless_of_label = False
    start_idx = 0
    # Get all pickle files with "unsafe" in the filename
    hdf5_files = [f for f in os.listdir(directory) if "traj" in f]
    hdf5_files = sorted(hdf5_files)
    print("total files:", len(hdf5_files))
    # Get the full paths

    tot = len(hdf5_files)
    don = 0

    for idx, traj_file in enumerate(hdf5_files):  # in range(10):
        # done_file = os.path.join(labeled_directory, traj_file)
        traj_file = os.path.join(directory, traj_file)
        labeled = check_if_labeled(traj_file, label_type)
        if idx < start_idx:
            don += 1
            continue

        if labeled and not reset_regardless_of_label:
            don += 1
            continue

        if not os.path.exists(traj_file):
            print(f"File {traj_file} not found, skipping.")
            continue

        print(f"Processing {traj_file}...")
        # if "safe" in traj_file and "unsafe" not in traj_file:
        #     process_trajectory_safe(traj_file)
        # else:
        fig, ax = plt.subplots()
        fig.suptitle(f"Trajectory {don + 1}/{tot}")
        plt.subplots_adjust(bottom=0.2)
        fig.text(
            0.5,
            0.05,
            'Press "0" for not divided,\n"1" for divided,\nspace to rewind',
            ha="center",
            fontsize=12,
        )
        # add progress bar
        rect = plt.Rectangle((0, 0), 224, 10, color="lightgray")
        ax.add_patch(rect)


        process_trajectory(traj_file)
        if images:
            update_plot()
            key_press_cid = fig.canvas.mpl_connect("key_press_event", on_key_press)
            print(f"Press '0' as not divided or '1' as divided to label {traj_file}.")
            plt.show(block=True)
            

        postprocess_trajectory(traj_file, labels, label_type=label_type)
        don += 1
        print(f"Done {don}/{tot}")
        print(f"Finished labeling for {traj_file}.")
