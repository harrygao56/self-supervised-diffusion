import torch.nn.functional as F
from guided_diffusion.script_util import add_dict_to_argparser
import os
import argparse
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    dn_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def train_defaults():
    return dict(
        image_size=384,
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


def start_train(cust_args):
    args = cust_args

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and dataloader...")
    for k,v in vars(args).items():
        logger.log(f'{k}: {v}')
    
    model, diffusion = dn_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = dn_load_data(
        args.data_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
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


def dn_load_data(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        dataset_type="supervised",
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
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
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    parser = create_argparser()
    parser.add_argument("--run", required=False)
    parser.add_argument("--run_override", required=False)
    parser.add_argument("--log_path", required=False)

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

    # Adding defaults to parser
    add_dict_to_argparser(parser, defaults)

    args = parser.parse_args()

    if args.run is None and args.log_path is None:
        parser.error("Please specify either --run or --log_path")

    if args.run is not None:
        run_path = f"/project/cigserver3/export1/g.harry/guided-diffusion/logs/fastmri-run{args.run}-conditional"
        if os.path.isdir(run_path) and args.run_override is None:
            raise Exception(f"{run_path} already exists")
        os.environ["OPENAI_LOGDIR"] = run_path
    else:
        os.environ["OPENAI_LOGDIR"] = args.log_path

    start_train(args)


if __name__ == "__main__":
    main()