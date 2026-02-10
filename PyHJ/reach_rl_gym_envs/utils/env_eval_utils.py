from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import measure
from sklearn.metrics import roc_auc_score

from PyHJ.utils.eval_utils import evaluate_V


def get_metrics(rl_values, gt_values, config):
    assert rl_values.shape == gt_values.shape

    rl_values_subzero = rl_values > config.safety_margin_threshold
    gt_values_subzero = gt_values > 0
    tp = np.sum((rl_values_subzero == 1) & (gt_values_subzero == 1))
    fp = np.sum((rl_values_subzero == 1) & (gt_values_subzero == 0))
    fn = np.sum((rl_values_subzero == 0) & (gt_values_subzero == 1))
    tn = np.sum((rl_values_subzero == 0) & (gt_values_subzero == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    balanced_accuracy = 0.5 * (tpr + tnr)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Advanced metrics
    # IOU
    intersection = np.sum((rl_values_subzero == 1) & (gt_values_subzero == 1))
    union = np.sum((rl_values_subzero == 1) | (gt_values_subzero == 1))
    iou = intersection / union if union > 0 else 0

    # Compute AUROC
    auc = roc_auc_score(gt_values_subzero.flatten(), rl_values.flatten())

    return {
        "FP": fp,
        "TP": tp,
        "FN": fn,
        "TN": tn,
        # "FPR": fpr,
        # "TPR": tpr,
        # "FNR": fnr,
        # "Accuracy": accuracy,
        # "Balanced Accuracy": balanced_accuracy,
        # "Precision": prec,
        # "Recall": rec,
        # "F1": f1,
        # "IOU": iou,
        "AUC": auc,
    }


def get_eval_plot(env, policy, critic, in_distribution=True):
    nx, ny, nt = 51, 51, 51
    # thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2]
    thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]

    fig1, axes1 = plt.subplots(2, len(thetas), figsize=(2 * len(thetas), 6))
    fig2, axes2 = plt.subplots(2, len(thetas), figsize=(2 * len(thetas), 6))
    X, Y = np.meshgrid(
        np.linspace(-1.0, 1.0, nx, endpoint=True),
        np.linspace(-1.0, 1.0, ny, endpoint=True),
    )
    env.select_constraints(in_distribution=in_distribution)
    gt_values = env.solver.solve(  # (nx, ny, nt)
        constraints=env.constraint, constraints_shape=env.constraint_shape
    )

    all_metrics = []

    for i in range(len(thetas)):
        V = np.zeros_like(X)
        for ii in range(nx):
            for jj in range(ny):
                tmp_point = np.array(
                    [
                        X[ii, jj],
                        Y[ii, jj],
                        np.sin(thetas[i]),
                        np.cos(thetas[i]),
                    ]
                )
                temp_obs = {
                    "state": tmp_point,
                    "constraints": np.array(env.constraint),
                }
                V[ii, jj] = evaluate_V(obs=temp_obs, policy=policy, critic=critic)

        nt_index = int(
            np.round((thetas[i] / (2 * np.pi)) * (nt - 1))
        )  # Convert theta to index in the grid

        metrics = get_metrics(rl_values=V, gt_values=gt_values[:, :, nt_index].T)
        all_metrics.append(metrics)

        # Find contours for gt and rl Value functions
        contours_rl = measure.find_contours(np.array(V > 0).astype(float), level=0.5)
        contours_gt = measure.find_contours(
            np.array(gt_values[:, :, nt_index].T > 0).astype(float), level=0.5
        )

        # Show sub-zero level set
        axes1[0, i].imshow(V > 0, extent=(-1.0, 1.0, -1.0, 1.0), origin="lower")
        axes1[1, i].imshow(
            gt_values[:, :, nt_index].T > 0,
            extent=(-1.1, 1.1, -1.1, 1.1),
            origin="lower",
        )
        # Show value functions
        axes2[0, i].imshow(
            V, extent=(-1.0, 1.0, -1.0, 1.0), vmin=-1.0, vmax=1.0, origin="lower"
        )
        axes2[1, i].imshow(
            gt_values[:, :, nt_index].T,
            extent=(-1.1, 1.1, -1.1, 1.1),
            vmin=-1.0,
            vmax=1.0,
            origin="lower",
        )

        # Plot contours for RL Value function
        for contour in contours_rl:
            for axes in [axes1, axes2]:
                extent = 1.0
                [
                    ax.plot(
                        contour[:, 1] * (2 * extent / (nx - 1)) - extent,
                        contour[:, 0] * (2 * extent / (ny - 1)) - extent,
                        color="blue",
                        linewidth=1,
                        label="RL Value Contour",
                    )
                    for ax in axes[:, i]
                ]
        # Plot contours for GT Value function
        for contour in contours_gt:
            for axes in [axes1, axes2]:
                extent = 1.1
                [
                    ax.plot(
                        contour[:, 1] * (2 * extent / (nx - 1)) - extent,
                        contour[:, 0] * (2 * extent / (ny - 1)) - extent,
                        color="Green",
                        linewidth=1,
                        label="GT Value Contour",
                    )
                    for ax in axes[:, i]
                ]

        # Add constraint patches
        x, y, r, u = env.constraint

        # Add the circle to the axes
        [
            ax.add_patch(Circle((x, y), r, color="red", fill=False, label="Fail Set"))
            for ax in axes1[:, i]
        ]
        [
            ax.add_patch(Circle((x, y), r, color="red", fill=False, label="Fail Set"))
            for ax in axes2[:, i]
        ]
        axes1[0, i].set_title(
            "theta = {}".format(np.round(thetas[i], 2)),
            fontsize=12,
        )
        axes2[0, i].set_title(
            "theta = {}".format(np.round(thetas[i], 2)),
            fontsize=12,
        )
        axes1[1, i].set_title(
            "GT,\ntheta = {}".format(np.round(thetas[i], 2)),
            fontsize=12,
        )
        axes2[1, i].set_title(
            "GT,\ntheta = {}".format(np.round(thetas[i], 2)),
            fontsize=12,
        )

    for axes in [axes1, axes2]:
        for ax in axes.flat:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")

    for fig, axes in zip([fig1, fig2], [axes1, axes2]):
        handles, labels = [], []
        for ax in axes.flat:
            h, label = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(label)

        # Remove duplicates while preserving order
        unique = dict(zip(labels, handles))

        # Create a single, global legend
        fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the legend

    aggregated = defaultdict(list)
    for metrics in all_metrics:
        for key, value in metrics.items():
            aggregated[key].append(value)

    # Compute averages
    averaged_metrics = {key: np.mean(values) for key, values in aggregated.items()}

    return (
        fig1,
        fig2,
        averaged_metrics,
    )
