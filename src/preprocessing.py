import argparse
import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from octprocessing import get_unlabelled_bscans, get_valid_img_seg_reimpl


def plot_prediction(image, mask, prediction, file_name):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input Image", "True Segmentation", "Model Output"]
    cmaps = ["gray", "plasma", "viridis"]

    for i, (img, title, cmap) in enumerate(zip([image, mask, prediction], titles, cmaps)):
        axs[i].imshow(img, cmap=cmap)
        axs[i].set_title(title, fontsize=12)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout(pad=2.0)
    plt.savefig(file_name, dpi=200)
    plt.close(fig)


def pad_to_max_num(i: int, mx: int) -> str:
    return str(i).zfill(len(str(mx)))


def slice_to_bscans(x: np.ndarray) -> List[np.ndarray]:
    return [x[:, :, i] for i in range(x.shape[2])]


def slicing(x: torch.Tensor) -> List[torch.Tensor]:
    col_width = 12
    slices = []
    for bscan in slice_to_bscans(x.numpy()):
        for i in range(bscan.shape[1] // col_width):
            slices.append(bscan[:, i * col_width : (i + 1) * col_width])
    return [torch.tensor(s) for s in slices]


def test_slicing(path):
    obj = np.load(path, allow_pickle=True).item()
    x, _ = get_valid_img_seg_reimpl(obj)
    x = torch.Tensor(x)
    slices = slicing(x)

    col_num = x.shape[1] // 12
    for b in range(x.shape[2]):
        for col in range(col_num):
            assert torch.all(x[:, col * 12:(col + 1) * 12, b] == slices[b * col_num + col]), \
                f"Slice {col} in image {b} is off"
    sum_scans = sum([e.shape[1] for e in slices])
    assert x.shape[1] // 12 * 12 * x.shape[2] == sum_scans, \
        f"Sample count mismatch: {sum_scans} vs expected {x.shape[1] // 12 * 12 * x.shape[2]}"


class DataPreprocessor:
    def __init__(self, data_path, dest_path, slicing: Callable, is_mat=False, labelled_dataset=True):
        self.ft_mat = {True: ".mat", False: ".npy"}
        self.is_mat = is_mat
        self.labelled_dataset = labelled_dataset
        self.dest_path = dest_path
        self.slicing = slicing

        assert data_path != dest_path, "Source and destination paths must differ"
        self.img_files = [
            os.path.abspath(e.path)
            for e in os.scandir(data_path)
            if e.is_file() and e.name.endswith(self.ft_mat[is_mat])
        ]
        assert self.img_files, f"No {self.ft_mat[is_mat]} files found in {data_path}"

        self.img_slice_nums = {}
        self.labels = {"images", "masks"} if labelled_dataset else {"images"}


        for f in self.img_files:
            obj = sio.loadmat(f) if is_mat else np.load(f, allow_pickle=True).item()
            if labelled_dataset:
                img, mask = map(torch.Tensor, get_valid_img_seg_reimpl(obj))
                self.img_slice_nums[f] = len(slicing(img))
                assert self.img_slice_nums[f] == len(slicing(mask)), f"Image/mask slice mismatch in {f}"
            else:
                img = torch.Tensor(get_unlabelled_bscans(obj))
                self.img_slice_nums[f] = len(slicing(img))


        for lbl in self.labels:
            for split in ["train", "val", "test"]:
                os.makedirs(os.path.join(dest_path, split, lbl), exist_ok=True)

    def preprocess(self):
        for i, fpath in enumerate(self.img_files):
            filename = os.path.basename(fpath)
            subject_num = int(filename.split(".")[0].split("_")[1])

            if subject_num < 7:
                split = "train"
            elif subject_num in [7, 8]:
                split = "val"
            elif subject_num in [9, 10]:
                split = "test"
            else:
                raise ValueError(f"Unexpected subject number in filename: {filename}")

            obj = sio.loadmat(fpath) if self.is_mat else np.load(fpath, allow_pickle=True).item()
            if self.labelled_dataset:
                img, mask = map(torch.Tensor, get_valid_img_seg_reimpl(obj))
                data = {"images": img, "masks": mask}
            else:
                data = {"images": torch.Tensor(get_unlabelled_bscans(obj))}

            slices = {k: self.slicing(v) for k, v in data.items() if k in self.labels}
            total_slices = self.img_slice_nums[fpath]

            for slice_num in tqdm(range(total_slices), desc="Saving slices"):
                for label in self.labels:
                    sample = slices[label][slice_num]
                    out_dir = os.path.join(self.dest_path, split, label)
                    base = os.path.splitext(filename)[0]
                    fname = f"{base}_{pad_to_max_num(slice_num, total_slices)}.npy"
                    np.save(os.path.join(out_dir, fname), sample.numpy())

    def __init__(self, data_path, dest_path, slicing: Callable, is_mat=False):
        assert data_path != dest_path, "Source and destination paths must differ"

        self.dest_path = dest_path
        self.is_mat = is_mat
        self.slicing = slicing
        self.labels = {"images", "masks"}

        dataset = sio.loadmat(data_path)
        self.images = dataset["AllSubjects"][0][:29]
        self.masks = dataset["ManualFluid1"][0]
        self.img_slice_nums = {}

        for i in range(len(self.images)):
            img, mask = map(torch.Tensor, (self.images[i], self.masks[i]))
            self.img_slice_nums[str(i)] = len(slicing(img))
            assert self.img_slice_nums[str(i)] == len(slicing(mask)), f"Mismatch at index {i}"

        for lbl in self.labels:
            for split in ["train", "val", "test"]:
                os.makedirs(os.path.join(dest_path, split, lbl), exist_ok=True)

    def preprocess(self):
        for i, (img_np, mask_np) in enumerate(zip(self.images, self.masks)):

            split = "train" if i < 19 else "val" if i < 24 else "test"
            img, mask = map(torch.Tensor, (img_np, mask_np))
            slices = {k: self.slicing(v) for k, v in {"images": img, "masks": mask}.items()}
            total_slices = self.img_slice_nums[str(i)]

            for slice_num in tqdm(range(total_slices), desc="Saving slices"):
                for label in self.labels:
                    sample = slices[label][slice_num]
                    out_dir = os.path.join(self.dest_path, split, label)
                    fname = f"{str(i)}_{pad_to_max_num(slice_num, total_slices)}.npy"
                    np.save(os.path.join(out_dir, fname), sample.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing for dataset containing siim-acr-pneumothorax-segmentation")
    parser.add_argument("data_path", type=str, help="Path to input data")
    parser.add_argument("dest_path", type=str, help="Path to save output")
    parser.add_argument("--dataset", choices=["Pneumothorax", None], default=None)
    parser.add_argument("--is_mat", type=bool, default=True)
    parser.add_argument("--full_bscan", type=bool, default=True)
    parser.add_argument(
        "--extract_dataset",
        choices=["labelled_dataset", "unlabelled_dataset"],
        default="labelled_dataset"
    )
    args = parser.parse_args()

    labelled = args.extract_dataset == "labelled_dataset"
    slicing_fn = slice_to_bscans if args.full_bscan else slicing

    processor = DataPreprocessor(args.data_path, args.dest_path, slicing_fn, args.is_mat, labelled)
    processor.preprocess()
