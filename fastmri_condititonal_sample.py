import torch.nn.functional as F
import os
import argparse
from guided_diffusion.image_datasets import FastMRIDataset, AmbientDatasetComplex, FullRankDataset, InDIDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    dn_create_model_and_diffusion,
    fullrank_dn_create_model_and_diffusion,
    ambient_dn_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    indi_create_model
)
import torch.distributed as dist
import inspect


def sample_defaults():
    return dict(
        image_size=384,
        original_height=768,
        original_width=396,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        diffusion_steps=1000,
        noise_schedule="cosine",
        batch_size=1,
        attention_resolutions="16,8",
        timestep_respacing="1000",
        num_samples=1,
        clip_denoised="",
    )


def create_lowres_samples(num_samples, height, width, type, indi, cust_inds=None):
    if indi == True:
        dataset = InDIDataset("test")
    elif type == "supervised":
        dataset = FastMRIDataset("test")
    elif type == "ambient":
        dataset = AmbientDatasetComplex("test")
    elif type == "fullrank":
        dataset = FullRankDataset("test")
    else:
        print("dataset not implemented")
        return

    if indi == True:
        sample_array = np.zeros((num_samples, height, width, 2), dtype=np.float32)
    else:
        sample_array = np.zeros((num_samples, height, width, 1), dtype=np.complex64)

    gt_array = np.zeros((num_samples, height, width))
    mask_arr = np.zeros((num_samples, height, width))
    smps_arr = np.zeros((num_samples, 20, height, width), dtype=np.complex64)

    if cust_inds:
        inds = [int(element) for element in cust_inds]
    else:
        inds = random.sample(range(len(dataset)), num_samples)

    curr_it = 0
    for i in inds:
        print(i)
        x, dict_out = dataset[i]
        
        sample_array[curr_it] = np.transpose(dict_out["AtAx"].cpu().detach().numpy(), [1, 2, 0])
        if indi == True:
            gt_array[curr_it] = abs(x[0,:,:] + 1j*x[1,:,:])
        else:
            gt_array[curr_it] = abs(x)

        if type != "supervised":
            mask_arr[curr_it] = dict_out["A"].cpu().detach().numpy()
            smps_arr[curr_it] = dict_out["smps"].cpu().detach().numpy()
        
        curr_it += 1

    if type == "supervised":
        mask_arr, smps_arr = None, None
    
    return sample_array, gt_array, mask_arr, smps_arr

def main():
    parser = create_argparser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--indi", required=True, type=bool)
    parser.add_argument("--indisteps", required=False, type=int)
    parser.add_argument('--indi_noise', required=False, type=float)
    parser.add_argument('--cust', nargs='*', required=False)

    defaults = sample_defaults()

    # Change argument defaults that already exist in parser
    parser.set_defaults(image_size=defaults['image_size'])
    parser.set_defaults(num_channels=defaults['num_channels'])
    parser.set_defaults(num_res_blocks=defaults['num_res_blocks'])
    parser.set_defaults(num_heads=defaults['num_heads'])
    parser.set_defaults(diffusion_steps=defaults['diffusion_steps'])
    parser.set_defaults(noise_schedule=defaults['noise_schedule'])
    parser.set_defaults(batch_size=defaults['batch_size'])
    parser.set_defaults(attention_resolutions=defaults['attention_resolutions'])
    parser.set_defaults(timestep_respacing=defaults['timestep_respacing'])
    parser.set_defaults(num_samples=defaults['num_samples'])
    parser.set_defaults(clip_denoised=defaults['clip_denoised'])

    # Adding defaults to parser
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    lowres_samples, gt_array, masks, smps = create_lowres_samples(args.num_samples, args.original_height, args.original_width, args.type, args.indi, cust_inds=args.cust)
    
    # Take model path and retrieves the directory to get the save path
    save_path = args.model_path[:(len(args.model_path) - args.model_path[::-1].index("/"))] + "samples"

    os.environ["OPENAI_LOGDIR"] = save_path

    print("sampling...")
    arr  = create_samples(args=args, lowres_sample_array=lowres_samples, mask_arr=masks, smps_arr=smps)
    arr = arr[:,:,:,0] + 1j * arr[:,:,:,1]
    if args.indi == True:
        lowres_samples = np.expand_dims(lowres_samples[:,:,:,0] + 1j * lowres_samples[:,:,:,1], 3)

    # Save results
    with open(f"{save_path}/metrics.txt", "w") as f:
        f.write("Sample metrics:\n")
        
        sums = {
            "error": 0,
            "lowres_error": 0,
            "psnr": 0,
            "lowres_psnr": 0,
            "ssim": 0,
            "lowres_ssim": 0
        }

        for i in range(len(lowres_samples)):
            plt.imshow(abs(lowres_samples[i,:,:,0]), cmap='gray')
            plt.show()
            plt.savefig(f"{save_path}/lowres{i}")

            plt.imshow(gt_array[i,:,:], cmap='gray')
            plt.show()
            plt.savefig(f"{save_path}/gt{i}")

            plt.imshow(abs(arr[i,:,:]), cmap='gray')
            plt.show()
            plt.savefig(f"{save_path}/sample{i}")

            lowres_range = abs(lowres_samples[i,:,:,0]).max() - abs(lowres_samples[i,:,:,0]).min()
            sample_range = abs(arr[i,:,:]).max() - abs(arr[i,:,:]).min()

            error = np.linalg.norm(abs(arr[i,:,:]) - abs(gt_array[i,:,:])) / np.linalg.norm(abs(arr[i,:,:]))
            lowres_err = np.linalg.norm(abs(lowres_samples[i,:,:,0]) - abs(gt_array[i,:,:])) / np.linalg.norm(abs(gt_array[i,:,:]))

            psnr = peak_signal_noise_ratio(abs(gt_array[i,:,:]), abs(arr[i,:,:]), data_range=sample_range)
            lowres_psnr = peak_signal_noise_ratio(abs(gt_array[i,:,:]), abs(lowres_samples[i,:,:,0]), data_range=lowres_range)

            ssim = structural_similarity(abs(gt_array[i,:,:]), abs(arr[i,:,:]), data_range=sample_range)
            lowres_ssim = structural_similarity(abs(gt_array[i,:,:]), abs(lowres_samples[i,:,:,0]), data_range=lowres_range)

            f.write(f"\nSample {i}\n")
            f.write(f"Lowres Error: {lowres_err}, Sample Error: {error}\n")
            f.write(f"Lowres PSNR: {lowres_psnr}, Sample PSNR: {psnr}\n")
            f.write(f"Lowres SSIM: {lowres_ssim}, Sample SSIM: {ssim}\n")

            sums["error"] += error
            sums["lowres_error"] += lowres_err
            sums["psnr"] += psnr
            sums["lowres_psnr"] += lowres_psnr
            sums["ssim"] += ssim
            sums["lowres_ssim"] += lowres_ssim

        for k, v in sums.items():
            f.write(f"Average {k}: {v / len(lowres_samples)}\n")


def create_samples(args, lowres_sample_array=None, mask_arr=None, smps_arr=None):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    for k,v in vars(args).items():
        logger.log(f'{k}: {v}')

    if args.indi == True:
        model = indi_create_model(
            **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
        )
    elif args.type == "ambient":
        model, diffusion = ambient_dn_create_model_and_diffusion(
            **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
        )
    elif args.type == "fullrank":
        model, diffusion = fullrank_dn_create_model_and_diffusion(
            **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
        )
    elif args.type == "supervised":
        model, diffusion = dn_create_model_and_diffusion(
            **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
        )
    else:
        print("Model not declared")
        return
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.batch_size, args.class_cond, lowres_sample_array=lowres_sample_array, mask_arr=mask_arr, smps_arr=smps_arr)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        
        if args.indi == True:
            AtAx = model_kwargs["AtAx"]
            n = torch.randn(AtAx.shape).to(dist_util.dev())
            sample = AtAx + args.indi_noise * n
            timesteps = np.arange(args.indisteps, 0, -1)
            # timesteps = np.linspace(1.0, 0, args.indisteps, endpoint=False)
            with torch.no_grad():
                for t in timesteps:
                    print(t / args.indisteps)
                    t_tensor = torch.tensor([t / int(args.indisteps)], dtype=torch.float32).repeat(args.batch_size, 1, 1, 1).to(dist_util.dev())
                    # t_tensor = torch.tensor([t], dtype=torch.float32).repeat(args.batch_size, 1, 1, 1).to(dist_util.dev())
                    t_model = t_tensor.view(args.batch_size)
                    model_output = model(sample, t_model)
                    d = 1 / args.indisteps
                    sample = (d / t_tensor) * model_output + (1 - d / t_tensor) * sample
        else:
            sample, _ = diffusion.p_sample_loop(
                model,
                (args.batch_size, 2, args.original_height, args.original_width),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
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


def load_data_for_worker(batch_size, class_cond, lowres_sample_array=None, mask_arr=None, smps_arr=None):
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    mask_buffer = []
    smps_buffer = []
    while True:
        for i in range(rank, len(lowres_sample_array), num_ranks):
            buffer.append(lowres_sample_array[i])
            if class_cond:
                label_buffer.append(lowres_sample_array[i])
            if mask_arr is not None:
                mask_buffer.append(mask_arr[i])
                smps_buffer.append(smps_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.concatenate([np.expand_dims(arr, 0) for arr in buffer], dtype=lowres_sample_array.dtype))
                batch = batch.permute(0, 3, 1, 2)
                res = dict(AtAx=batch)
                if mask_arr is not None:
                    res["A_hat"] = torch.from_numpy(np.stack(mask_buffer)).float()
                    res["smps"] = torch.from_numpy(np.stack(smps_buffer))
                if class_cond:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer, mask_buffer, smps_buffer = [], [], [], []


def create_argparser():
    defaults = model_and_diffusion_defaults()

    arg_names = inspect.getfullargspec(dn_create_model_and_diffusion)[0]
    for k in defaults.copy().keys():
        if k not in arg_names:
            del defaults[k]

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()