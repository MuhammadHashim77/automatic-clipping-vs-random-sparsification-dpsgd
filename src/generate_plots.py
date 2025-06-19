import ast
import os
import re

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def save_results_to_csv(file_name: str, **kwargs):
    """Save results as a row to a CSV file, appending if the file exists."""
    data = {key: [value] for key, value in kwargs.items()}
    df = pd.DataFrame(data)
    if os.path.isfile(file_name):
        df_existing = pd.read_csv(file_name)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(file_name, index=False)


def extract_tensors_from_string(s: str):
    """Extract tensor values from a string representation."""
    numbers = re.findall(r"tensor\((.*?)\)", s)
    return [torch.tensor(float(number)) for number in numbers]


def get_unique_experiment_keys(df: pd.DataFrame) -> np.ndarray:
    """Return unique experiment keys based on algorithm and noise multiplier (and fold if present)."""
    required_columns = ["algorithm", "noise_multiplier"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    keys = df["algorithm"].astype(str) + " " + df["noise_multiplier"].fillna("").astype(str)
    if "k_fold_iteration" in df.columns and "k_fold" in df.columns:
        keys += " F:" + df["k_fold_iteration"].astype(str) + "/" + df["k_fold"].astype(str)
    return keys.unique()


def get_color_map(unique_keys):
    """Return a color map for unique experiment keys."""
    colors = cm.rainbow(np.linspace(0, 1, len(unique_keys)))
    return dict(zip(unique_keys, colors))


def get_plot_label(row):
    """Return a formatted label for a plot legend from a DataFrame row."""
    label = f"{row['algorithm']} {row['noise_multiplier']}".strip()
    if "k_fold_iteration" in row and "k_fold" in row:
        label += f" F:{row['k_fold_iteration']}/{row['k_fold']}"
    return label


def plot_metric_vs_epoch(df, metric_key, ylabel, title, save_path, color_map=None, logy=False):
    """Generic function to plot a metric vs. epoch for all experiment keys."""
    unique_keys = get_unique_experiment_keys(df)
    if color_map is None:
        color_map = get_color_map(unique_keys)
    plt.figure()
    for _, row in df.iterrows():
        epochs = list(range(1, len(row[metric_key]) + 1))
        label = get_plot_label(row)
        color = color_map.get(label, None)
        plt.plot(epochs, row[metric_key], color=color, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_privacy_vs_epoch(df, privacy_key, save_path, color_map=None):
    """Plot privacy budget (epsilon) vs. epoch for all DP experiments."""
    unique_keys = get_unique_experiment_keys(df)
    if color_map is None:
        color_map = get_color_map(unique_keys)
    plt.figure()
    for _, row in df.iterrows():
        if row["algorithm"] == "Non-DP":
            continue
        epsilons = ast.literal_eval(row[privacy_key])
        epochs = list(range(1, len(epsilons) + 1))
        label = get_plot_label(row)
        color = color_map.get(label, None)
        plt.plot(epochs, epsilons, color=color, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Privacy Budget (ε)")
    plt.title(f"Cumulative Privacy Budget vs Epoch\nModel: {row['model_name']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_all_metrics_from_csv():
    """Load results from CSVs and plot all relevant metrics for each model."""
    # File paths for results
    model_files = {
        "unet": [
            "results/Non-DP/unet.csv",
            "results/Opacus-AC/unet.csv",
            "results/Opacus-RS/unet.csv",
        ],
        "nested_unet": [
            "results/Non-DP/NestedUNet.csv",
            "results/Opacus-AC/NestedUNet.csv",
            "results/Opacus-RS/NestedUNet.csv",
        ],
    }
    save_dir = "results/collected_data"
    os.makedirs(save_dir, exist_ok=True)

    for model_name, file_list in model_files.items():
        df = pd.DataFrame()
        for file_name in file_list:
            df = pd.concat([df, pd.read_csv(file_name)], ignore_index=True)
        # Convert stringified lists to actual lists
        for col in ["iteration_train_times", "training_losses", "validation_losses", "validation_accuracies"]:
            if col in df.columns:
                df[col] = df[col].apply(ast.literal_eval)
        # Plot metrics
        color_map = get_color_map(get_unique_experiment_keys(df))
        plot_metric_vs_epoch(
            df,
            metric_key="iteration_train_times",
            ylabel="Training Time per Epoch (log scale, seconds)",
            title=f"Training Time per Epoch for Each Configuration\nModel: {model_name}",
            save_path=os.path.join(save_dir, f"Iterations_Train_times_vs_Epochs_{model_name}.png"),
            color_map=color_map,
            logy=True,
        )
        plot_metric_vs_epoch(
            df,
            metric_key="training_losses",
            ylabel="Training Loss",
            title=f"Training Loss Over Epochs\nModel: {model_name}",
            save_path=os.path.join(save_dir, f"Training_Loss_vs_Epochs_{model_name}.png"),
            color_map=color_map,
        )
        plot_metric_vs_epoch(
            df,
            metric_key="validation_losses",
            ylabel="Validation Loss",
            title=f"Validation Loss Over Epochs\nModel: {model_name}",
            save_path=os.path.join(save_dir, f"Validation_Loss_vs_Epochs_{model_name}.png"),
            color_map=color_map,
        )
        plot_metric_vs_epoch(
            df,
            metric_key="validation_accuracies",
            ylabel="Validation Accuracy",
            title=f"Validation Accuracy Over Epochs\nModel: {model_name}",
            save_path=os.path.join(save_dir, f"Validation_Accuracy_Vs_Epochs_{model_name}.png"),
            color_map=color_map,
        )
        if "overall_privacy_spent" in df.columns:
            plot_privacy_vs_epoch(
                df,
                privacy_key="overall_privacy_spent",
                save_path=os.path.join(save_dir, f"Epsilon_vs_Epochs_{model_name}.png"),
                color_map=color_map,
            )


if __name__ == "__main__":
    plot_all_metrics_from_csv()