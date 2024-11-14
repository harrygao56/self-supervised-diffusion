import torch.nn.functional as F
import os
import argparse
from guided_diffusion.image_datasets import FastMRIDataset, AmbientDataset, SamplingDataset, SelfInDIDataset
from guided_diffusion.fastmri_dataloader import (fmult, ftran)
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
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import lpips


def sample_defaults():
    return dict(
        image_size=320,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        diffusion_steps=1000,
        noise_schedule="cosine",
        batch_size=4,
        attention_resolutions="16,8",
        timestep_respacing="1000",
        num_samples=float('inf'),
        clip_denoised="",
    )

def main():
    parser = create_argparser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--indisteps", required=False, type=int)
    parser.add_argument('--indinoise', required=False, type=float)
    parser.add_argument('--cust', nargs='*', required=False, type=int)
    parser.add_argument('--log_name', required=False)
    parser.add_argument('--indi', action='store_true')

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
    parser.set_defaults(indi=False)
    parser.set_defaults(log_name="sample")

    # Adding defaults to parser
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    save_path = args.model_path[:(len(args.model_path) - args.model_path[::-1].index("/"))] + args.log_name
    os.environ["OPENAI_LOGDIR"] = save_path

    if args.type == "selfindi":
        dataset = SelfInDIDataset(
            "tst_small"
        )
    else:
        dataset = SamplingDataset(
            "tst_small"
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, generator=torch.Generator().manual_seed(42)
    )

    if args.cust:
        subset = Subset(dataloader.dataset, args.cust)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print(f"dataset length: {len(dataset)}, dataloder length: {len(dataloader)}")

    create_samples(args, dataloader, save_path)

def load_model(args):
    logger.log("creating model...")
    for k,v in vars(args).items():
        logger.log(f'{k}: {v}')

    if args.type == "fullrank":
        if args.indi == True:
            print("creating indi fullrank model")
            model = indi_create_model(
                **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
            )
        else:
            print("creating fullrank model")
            model, diffusion = fullrank_dn_create_model_and_diffusion(
                **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
            )
    elif args.type == "ambient":
        if args.indi == True:
            print("creating indi ambient model")
            model = indi_create_model(
                **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
            )
        else:
            print("creating ambient model")
            model, diffusion = fullrank_dn_create_model_and_diffusion(
                **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
            )
    elif args.type == "supervised":
        if args.indi == True:
            print("creating indi supervised model")
            model = indi_create_model(
                **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
            )
        else:
            print("creating supervised model")
            model, diffusion = dn_create_model_and_diffusion(
                **args_to_dict(args, inspect.getfullargspec(dn_create_model_and_diffusion)[0])
            )
    elif args.type == "selfindi":
        print("creating self indi model")
        model = indi_create_model(
            **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
        )
    else:
        print("type not implemented")
        return
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.indi:
        diffusion = None

    return model, diffusion

def create_samples(args, dataloader, save_path):
    dist_util.setup_dist()
    logger.configure()

    model, diffusion = load_model(args)
    lpips_fun = lpips.LPIPS(net='alex')

    count = 0
    sums = {
        "sample_error": 0,
        "lowres_error": 0,
        "sample_psnr": 0,
        "lowres_psnr": 0,
        "sample_ssim": 0,
        "lowres_ssim": 0,
        "sample_lpips": 0,
        "lowres_lpips": 0
    }

    for x, model_kwargs in dataloader:
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        sample = sample_loop(model, diffusion, x, model_kwargs, args).cpu().numpy()
        x = x.cpu().numpy()
        # sample[x == 0.0] = 0.0

        sample = abs(sample[:,0,:,:] + 1j * sample[:,1,:,:])
        if args.type == "selfindi":
            lowres = abs(model_kwargs["x_hat__"][:,0,:,:] + 1j * model_kwargs["x_hat__"][:,1,:,:]).cpu().numpy()
        else:
            lowres = abs(model_kwargs["x_hat_"][:,0,:,:] + 1j * model_kwargs["x_hat_"][:,1,:,:]).cpu().numpy()

        gt = abs(x[:,0,:,:] + 1j * x[:,1,:,:])

        # np.save(f"{save_path}/samples-batch-{count}", sample)
        # np.save(f"{save_path}/gt-batch-{count}", gt)
        # np.save(f"{save_path}/lowres-batch-{count}", lowres)

        for i in range(args.batch_size):
            curr_it = count + i
            # plt.imshow(lowres[i,:,:], cmap='gray')
            # plt.show()
            # plt.savefig(f"{save_path}/lowres{curr_it}")

            # plt.imshow(gt[i,:,:], cmap='gray')
            # plt.show()
            # plt.savefig(f"{save_path}/gt{curr_it}")

            # plt.imshow(sample[i,:,:], cmap='gray')
            # plt.show()
            # plt.savefig(f"{save_path}/sample{curr_it}")

            sample_psnr, sample_ssim, sample_error, sample_lpips = compute_metrics(sample[i], gt[i], lpips_fun)
            lowres_psnr, lowres_ssim, lowres_error, lowres_lpips = compute_metrics(lowres[i], gt[i], lpips_fun)
            with open(f"{save_path}/metrics.txt", "a") as f:
                f.write(f"\nSample {curr_it}\nLowres Error: {lowres_error}, Sample Error: {sample_error}\nLowres PSNR: {lowres_psnr}, Sample PSNR: {sample_psnr}\nLowres SSIM: {lowres_ssim}, Sample SSIM: {sample_ssim}\nLowres LPIPS: {lowres_lpips}, Sample LPIPS: {sample_lpips}\n__________________\n")

            sums["sample_error"] += sample_error
            sums["lowres_error"] += lowres_error
            sums["sample_psnr"] += sample_psnr
            sums["lowres_psnr"] += lowres_psnr
            sums["sample_ssim"] += sample_ssim
            sums["lowres_ssim"] += lowres_ssim
            sums["sample_lpips"] += sample_lpips
            sums["lowres_lpips"] += lowres_lpips

        count += args.batch_size
        print(f"generated {count} samples")
        if count >= args.num_samples:
            break
    
    with open(f"{save_path}/metrics.txt", "a") as f:
        for k, v in sums.items():
            f.write(f"Average {k}: {v / count}\n")

def compute_metrics(lowres, gt, lpips_fun):
    error = np.linalg.norm(lowres - gt) / np.linalg.norm(gt)
    psnr = peak_signal_noise_ratio(gt, lowres, data_range=1)
    ssim = structural_similarity(gt, lowres, data_range=1)
    lpips_ = lpips_fun.forward(torch.from_numpy(lowres), torch.from_numpy(gt)).squeeze().item()
    return psnr, ssim, error, lpips_

def sample_loop(model, diffusion, x, model_kwargs, args):
    if args.indi == True:
        n = torch.randn(x.shape).to(dist_util.dev())
        if args.type == "selfindi":
            sample = model_kwargs["x_hat__"] + args.indinoise * n
        elif args.type == "supervised":
            sample = model_kwargs["x_hat"] + args.indinoise * n
        else:
            sample = model_kwargs["x_hat_"] + args.indinoise * n
        timesteps = np.arange(args.indisteps, 0, -1)
        with torch.no_grad():
            timesteps = tqdm(timesteps)
            for t in timesteps:
                model_input = sample
                if args.type == "ambient" or args.type == "fullrank":
                    model_input = model_input[:,0,:,:] + 1j*model_input[:,1,:,:]
                    model_input = fmult(model_input, model_kwargs["smps"], model_kwargs["M_"])
                    model_input = ftran(model_input, model_kwargs["smps"], model_kwargs["M_"]).unsqueeze(1)
                    model_input = torch.cat([model_input.real, model_input.imag], axis=1)
                t_tensor = torch.tensor([t / int(args.indisteps)], dtype=torch.float32).repeat(args.batch_size, 1, 1, 1).to(dist_util.dev())
                t_model = t_tensor.view(args.batch_size)
                model_output = model(model_input, t_model, **model_kwargs)
                if t != 1.0 and args.type == "selfindi":
                    model_output = model_output[:,0,:,:] + 1j*model_output[:,1,:,:]
                    model_output = fmult(model_output, model_kwargs["smps"], model_kwargs["M_"])
                    model_output = ftran(model_output, model_kwargs["smps"], model_kwargs["M_"]).unsqueeze(1)
                    model_output = torch.cat([model_output.real, model_output.imag], axis=1)
                d = 1 / args.indisteps
                sample = (d / t_tensor) * model_output + (1 - d / t_tensor) * sample
    else:
        model_kwargs["type"] = args.type
        if args.type == "supervised":
            model_kwargs["cond"] = model_kwargs["x_hat"]
        else:
            model_kwargs["cond"] = model_kwargs["x_hat_"]
        sample, _ = diffusion.p_sample_loop(
            model,
            (args.batch_size, 2, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
    
    return sample.contiguous()

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