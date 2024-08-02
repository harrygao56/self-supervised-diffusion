"""
Train an ambient super-resolution model.
"""

import argparse

import torch.nn.functional as F
import os
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults, 
    dn_create_model_and_diffusion,
    fullrank_dn_create_model_and_diffusion,
    ambient_dn_create_model_and_diffusion,
    indi_create_model,
    self_indi_create_model,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop, TrainLoopInDI
import inspect
import matplotlib.pyplot as plt

def train_defaults():
    return dict(
        image_size=320,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        diffusion_steps=1000,
        noise_schedule="cosine",
        lr=1e-4,
        batch_size=4,
        attention_resolutions="16,8",
        save_interval=10000,
    )

def start_train(args):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and dataloader...")
    for k,v in vars(args).items():
        logger.log(f'{k}: {v}')
    
    if args.type == "fullrank":
        if args.indi == True:
            print("creating indi fullrank model")
            model = self_indi_create_model(
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
            model = self_indi_create_model(
                **args_to_dict(args, inspect.getfullargspec(indi_create_model)[0])
            )
        else:
            print("creating ambient model")
            model, diffusion = ambient_dn_create_model_and_diffusion(
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
    else:
        print("Train type not implemented")
        return

    model.to(dist_util.dev())
    
    if args.indi == False:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_dn_data(
        args.data_dir,
        args.batch_size,
        dataset_type=args.type,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    if args.indi == True:
        TrainLoopInDI(
            model=model,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            diffusion_steps=args.diffusion_steps,
            type=args.type,
            noise=args.indinoise,
            pt=args.pt,
        ).run_loop()
    else:
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
        ).run_loop()


def load_dn_data(data_dir, batch_size, dataset_type, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        dataset_type=dataset_type,
        split="tra_large",
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    denoise_defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    
    defaults = model_and_diffusion_defaults()

    arg_names = inspect.getfullargspec(dn_create_model_and_diffusion)[0]
    for k in defaults.copy().keys():
        if k not in arg_names:
            del defaults[k]

    defaults.update(denoise_defaults)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    parser = create_argparser()
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument('--indi', action='store_true')
    parser.add_argument('--run_override', action='store_true')
    parser.add_argument("--indinoise", required=False, type=float)
    parser.add_argument("--pt", required=False)

    defaults = train_defaults()
    
    parser.set_defaults(image_size=defaults['image_size'])
    parser.set_defaults(num_channels=defaults['num_channels'])
    parser.set_defaults(num_res_blocks=defaults['num_res_blocks'])
    parser.set_defaults(num_heads=defaults['num_heads'])
    parser.set_defaults(diffusion_steps=defaults['diffusion_steps'])
    parser.set_defaults(noise_schedule=defaults['noise_schedule'])
    parser.set_defaults(batch_size=defaults['batch_size'])
    parser.set_defaults(attention_resolutions=defaults['attention_resolutions'])
    parser.set_defaults(save_interval=defaults['save_interval'])
    parser.set_defaults(lr=defaults['lr'])
    parser.set_defaults(indi=False)
    parser.set_defaults(run_override=False)
    parser.set_defaults(pt="uniform")

    # Adding defaults to parser
    add_dict_to_argparser(parser, defaults)

    args = parser.parse_args()

    if os.path.isdir(args.log_path) and args.run_override == False:
        raise Exception(f"{args.log_path} already exists")
    os.environ["OPENAI_LOGDIR"] = args.log_path

    start_train(args)

if __name__ == "__main__":
    main()
