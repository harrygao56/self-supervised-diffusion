import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from guided_diffusion.fastmri_dataloader import (FastBrainMRI, uniformly_cartesian_mask, fmult, ftran, get_weighted_mask)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


def load_data(
    *,
    data_dir,
    batch_size,
    dataset_type="",
    split="",
    class_cond=False,
    random_crop=False,
    random_flip=True,
    acceleration_rate=4,
    acceleration_rate_inter=6,
    acceleration_rate_further=8,
):
    if dataset_type == "supervised":
        print("creating supervised dataset")
        dataset = FastMRIDataset(
            split
        )
    elif dataset_type == "ambient" or dataset_type == "ambient2":
        print("creating ambient dataset")
        dataset = AmbientDataset(
            split,
            acceleration_rate=acceleration_rate,
            acceleration_rate_further=acceleration_rate_further,
        )
    elif dataset_type == "fullrank" or dataset_type == "fullrank2":
        print("creating fullrank dataset")
        dataset = FullRankDataset(
            split,
            acceleration_rate=acceleration_rate,
            acceleration_rate_further=acceleration_rate_further,
        )
    elif dataset_type == "selfindi":
        print("creating selfindi dataset")
        dataset = SelfInDIDataset(
            split,
            acceleration_rate=acceleration_rate,
            acceleration_rate_inter=acceleration_rate_inter,
            acceleration_rate_further=acceleration_rate_further,
        )

    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset size: {len(dataset)}")
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "gz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class FastMRIDataset(FastBrainMRI):
    # Dataset for conditional diffusion
    def __init__(
            self,
            split,
            acceleration_rate=8,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            modality_subset="T2",
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat, modality_subset=modality_subset)

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)
        M = uniformly_cartesian_mask(x.shape, self.acceleration_rate).astype(int)
        M = torch.from_numpy(M)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])

        out_dict = {
            "smps": smps.squeeze(), 
            "M": M.squeeze(),
            "x_hat": x_hat
        }
        return x, out_dict
    

class AmbientDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            acceleration_rate_further=8,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            modality_subset="T2",
        ):
            self.acceleration_rate_further = acceleration_rate_further
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat, modality_subset=modality_subset)

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        M = uniformly_cartesian_mask(x.shape, self.acceleration_rate).astype(int)
        M = torch.from_numpy(M)

        M_ = uniformly_cartesian_mask(x.shape, self.acceleration_rate_further).astype(int)
        M_ = torch.from_numpy(M_)

        y = fmult(x, smps, M)
        x_hat = ftran(y, smps, M)

        y_ = fmult(x, smps, M_)
        x_hat_ = ftran(y_, smps, M_)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])
        x_hat_ = torch.cat([x_hat_.real, x_hat_.imag])

        out_dict = {
            "smps": smps.squeeze(), 
            "M_": M_.squeeze(),
            "M": M.squeeze(),
            "y": y.squeeze(),
            "x_hat_": x_hat_.squeeze(),
            "x_hat": x_hat.squeeze(),
        }
        return x.squeeze(), out_dict
    

class SelfInDIDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            acceleration_rate_inter=6,
            acceleration_rate_further=8,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            modality_subset="T2",
        ):
            self.acceleration_rate_further = acceleration_rate_further
            self.acceleration_rate_inter = acceleration_rate_inter
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat, modality_subset=modality_subset)

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        M = uniformly_cartesian_mask(x.shape, self.acceleration_rate).astype(int)
        M = torch.from_numpy(M)

        M_bar = uniformly_cartesian_mask(x.shape, self.acceleration_rate_inter).astype(int)
        M_bar = torch.from_numpy(M_bar)

        M_ = uniformly_cartesian_mask(x.shape, self.acceleration_rate_further).astype(int)
        M_ = torch.from_numpy(M_)

        y = fmult(x, smps, M)
        x_hat = ftran(y, smps, M)

        y_bar = fmult(x, smps, M_bar)
        x_hat_bar = ftran(y_bar, smps, M_bar)
        
        y_ = fmult(x, smps, M_)
        x_hat_ = ftran(y_, smps, M_)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])
        x_hat_bar = torch.cat([x_hat_bar.real, x_hat_bar.imag])
        x_hat_ = torch.cat([x_hat_.real, x_hat_.imag])

        out_dict = {
            "smps": smps.squeeze(),
            "M_bar": M_bar.squeeze(),
            "M_": M_.squeeze(),
            "M": M.squeeze(),
            "y": y.squeeze(),
            "x_hat": x_hat.squeeze(),
            "x_hat_bar": x_hat_bar.squeeze(),
            "x_hat_": x_hat_.squeeze(),
        }
        return x.squeeze(), out_dict


class FullRankDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            acceleration_rate_further=8,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            modality_subset="T2",
            preload=True,
        ):
            self.acceleration_rate_further = acceleration_rate_further
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat, modality_subset=modality_subset, is_pre_load=preload)

    def __getitem__(self, item):
        x, x_hat_init, smps, _, _, M_init, _, _ = super().__getitem__(item)
        M_init = M_init.int()
        
        ind = torch.where(M_init[0,0] == 1)[0][0].item()

        M = uniformly_cartesian_mask(x.shape, self.acceleration_rate_further, get_specific=ind)
        M = torch.from_numpy(M).int()

        M_ = uniformly_cartesian_mask(x.shape, self.acceleration_rate_further, get_specific=ind + self.acceleration_rate)
        M_ = torch.from_numpy(M_).int()

        W = torch.from_numpy(get_weighted_mask(x.shape, 4).astype(int))

        y = fmult(x_hat_init, smps, M)
        x_hat = ftran(y, smps, M)

        y_ = fmult(x_hat_init, smps, M_)
        x_hat_ = ftran(y_, smps, M_)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])
        x_hat_ = torch.cat([x_hat_.real, x_hat_.imag])

        out_dict = {
            "smps": smps.squeeze(), 
            "M_": M_.squeeze(),
            "M": M.squeeze(),
            "y": y.squeeze(),
            "x_hat_": x_hat_.squeeze(),
            "x_hat": x_hat.squeeze(),
            "W": W.squeeze(),
        }
        return x.squeeze(), out_dict

class SamplingDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            modality_subset="T2",
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat, modality_subset=modality_subset)

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        M = uniformly_cartesian_mask(x.shape, 4).astype(int)
        M = torch.from_numpy(M)

        M_ = uniformly_cartesian_mask(x.shape, 8).astype(int)
        M_ = torch.from_numpy(M_)

        y = fmult(x, smps, M)
        x_hat = ftran(y, smps, M)

        y_ = fmult(x, smps, M_)
        x_hat_ = ftran(y_, smps, M_)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])
        x_hat_ = torch.cat([x_hat_.real, x_hat_.imag])

        out_dict = {
            "smps": smps.squeeze(), 
            "M_": M_.squeeze(),
            "M": M.squeeze(),
            "y": y.squeeze(),
            "x_hat_": x_hat_.squeeze(),
            "x_hat": x_hat.squeeze(),
        }
        return x.squeeze(), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    dataset = FullRankDataset(
        "tst_large"
    )
    for i in range(10):
        dataset.__getitem__(i)


if __name__ == "__main__":
    main()