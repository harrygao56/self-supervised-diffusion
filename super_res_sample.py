"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    ambient_sr_create_model_and_diffusion,
)


def main(cust_args=None, lowres_sample_array=None, mask_arr=None, smps_arr=None):
    if cust_args is not None:
        args = cust_args
    else:
        args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    for k,v in vars(cust_args).items():
        logger.log(f'{k}: {v}')

    if args.ambient:
        model, diffusion = ambient_sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
        )
    else:
        model, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
        )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    if lowres_sample_array is None:
        data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    else:
        data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond, lowres_sample_array=lowres_sample_array, ambient=args.ambient, mask_arr=mask_arr, smps_arr=smps_arr)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        if args.ambient:
            sample, _ = diffusion.p_sample_loop(
                model,
                (args.batch_size, 2, args.original_height, args.original_width),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        else:
            sample, _ = diffusion.p_sample_loop(
                model,
                (args.batch_size, 1, args.large_size, args.large_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")
    return arr


def load_data_for_worker(base_samples, batch_size, class_cond, lowres_sample_array=None, ambient=False, mask_arr=None, smps_arr=None):
    # Modified this function so we have the option to provide an array directly
    # Makes it so we don't need to save a sample array to memory before sampling
    if lowres_sample_array is None:
        with bf.BlobFile(base_samples, "rb") as f:
            obj = np.load(f)
            image_arr = obj["arr_0"]
            if class_cond:
                label_arr = obj["arr_1"]
    else:
        image_arr=lowres_sample_array
    
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    mask_buffer = []
    smps_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if mask_arr is not None:
                mask_buffer.append(mask_arr[i])
                smps_buffer.append(smps_arr[i])
            if len(buffer) == batch_size:
                if ambient:
                    batch = th.from_numpy(np.concatenate([np.expand_dims(arr, 0) for arr in buffer], dtype=image_arr.dtype))
                else:
                    batch = th.from_numpy(np.stack(buffer))
                # batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                if ambient:
                    res = dict(AtAx=batch)
                else:
                    res = dict(low_res=batch)
                if mask_arr is not None:
                    res["A_hat"] = th.from_numpy(np.stack(mask_buffer)).float()
                    res["smps"] = th.from_numpy(np.stack(smps_buffer))
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer, mask_buffer, smps_buffer = [], [], [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
