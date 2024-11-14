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
):
    if dataset_type == "supervised":
        print("creating supervised dataset")
        dataset = FastMRIDataset(
            split
        )
    elif dataset_type == "ambient" or dataset_type == "ambient2":
        print("creating ambient dataset")
        dataset = AmbientDataset(
            split
        )
    elif dataset_type == "fullrank" or dataset_type == "fullrank2":
        print("creating fullrank dataset")
        dataset = FullRankDataset(
            split
        )
    elif dataset_type == "selfindi":
        print("creating selfindi dataset")
        dataset = SelfInDIDataset(
            split
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
            acceleration_rate=4,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat)

    def __getitem__(self, item):
        x, x_hat, smps, _, _, _, _, _ = super().__getitem__(item)
        M = uniformly_cartesian_mask(x.shape, 8).astype(int)
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
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat)

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
    

class SelfInDIDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat)

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        M = uniformly_cartesian_mask(x.shape, 4).astype(int)
        M = torch.from_numpy(M)

        M_ = uniformly_cartesian_mask(x.shape, 6).astype(int)
        M_ = torch.from_numpy(M_)

        M__ = uniformly_cartesian_mask(x.shape, 8).astype(int)
        M__ = torch.from_numpy(M__)

        y = fmult(x, smps, M)
        x_hat = ftran(y, smps, M)

        y_ = fmult(x, smps, M_)
        x_hat_ = ftran(y_, smps, M_)
        
        y__ = fmult(x, smps, M__)
        x_hat__ = ftran(y__, smps, M__)

        x = torch.cat([x.real, x.imag])
        x_hat = torch.cat([x_hat.real, x_hat.imag])
        x_hat_ = torch.cat([x_hat_.real, x_hat_.imag])
        x_hat__ = torch.cat([x_hat__.real, x_hat__.imag])

        out_dict = {
            "smps": smps.squeeze(),
            "M__": M__.squeeze(),
            "M_": M_.squeeze(),
            "M": M.squeeze(),
            "y": y.squeeze(),
            "x_hat": x_hat.squeeze(),
            "x_hat_": x_hat_.squeeze(),
            "x_hat__": x_hat__.squeeze(),
        }
        return x.squeeze(), out_dict


class FullRankDataset(FastBrainMRI):
    def __init__(
            self,
            split,
            acceleration_rate=4,
            noise_sigma=0.01,
            num_coil_subset=20,
            is_return_y_smps_hat=True,
            preload=True,
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat)

            # If preload is True, we preload the masks to make sure we apply the same masks each time
            self.preload = preload
            if preload:
                self.preloaded_masks = []
                mask_shape = super().__getitem__(0)[0].shape
                for i in range(len(self)):
                    self.preloaded_masks.append(uniformly_cartesian_mask(mask_shape, 8, randomly_return=True, get_two=True))

    def __getitem__(self, item):
        x, _, smps, _, _, _, _, _ = super().__getitem__(item)

        # Generate a cartesian mask for the groundtruth
        if self.preload:
            M, M_ = self.preloaded_masks[item]
        else:
            M, M_ = uniformly_cartesian_mask(x.shape, 8, randomly_return=True, get_two=True)
        
        M = torch.from_numpy(M.astype(int))
        M_ = torch.from_numpy(M_.astype(int))
        W = torch.from_numpy(get_weighted_mask(x.shape, 4).astype(int))

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
        ):
            super().__init__(split, acceleration_rate=acceleration_rate, noise_sigma=noise_sigma, num_coil_subset=num_coil_subset, is_return_y_smps_hat=is_return_y_smps_hat)

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
    dataset = AmbientDataset(
            "tst_large"
        )
    
    x, args = next(dataset)
    path = "/project/cigserver3/export1/g.harry/self-supervised-diffusion/test"
    for _ in range(5):
        plt.imshow(abs(x), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/x{i}")

        AtAx = abs(args["AtAx"][0,:,:] + 1j * args["AtAx"][1,:,:])
        plt.imshow(AtAx, cmap='gray')
        plt.show()
        plt.savefig(f"{path}/AtAx{i}")

        AtAhat_x = fmult(AtAx, smps, args["A_hat"])
        AtAhat_x = ftran(AtAhat_x, smps, args["A_hat"])
        plt.imshow(abs(AtAhat_x), cmap='gray')
        plt.show()
        plt.savefig(f"{path}/AtAhat_x{i}")


    # print(args['x'].shape)
    # print(args['Ax'].shape)
    # print(args['mask'].shape)
    # print(im.shape)

    # plt.imshow(im[0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat")
    # plt.imshow(args["Ax"][0,:,:], cmap='gray')
    # plt.show()
    # plt.savefig("x-hat-masked")
    # plt.imshow(args["x"], cmap='gray')
    # plt.show()
    # plt.savefig("x")


if __name__ == "__main__":
    main()