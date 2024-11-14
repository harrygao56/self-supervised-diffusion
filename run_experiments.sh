#!/bin/bash

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 10 --log_name sample-10-0.01
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 16 --log_name sample-16-0.01

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 10 --log_name sample-10-0
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 16 --log_name sample-16-0

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 10 --log_name sample-10-0.01
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 16 --log_name sample-16-0.01

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 10 --log_name sample-10-0
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 16 --log_name sample-16-0

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 10 --log_name sample-10-0
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 16 --log_name sample-16-0

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 10 --log_name sample-10-0.01
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0.01-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 16 --log_name sample-16-0.01

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 10 --log_name sample-10-0
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0 --indisteps 16 --log_name sample-16-0

python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 10 --log_name sample-10-0.01
python fastmri_condititonal_sample.py --model_path /project/cigserver3/export1/g.harry/self-supervised-diffusion/new_experiments/selfindi-0-bias_t1/ema_0.9999_190000.pt --type selfindi --num_samples 30 --indi --indinoise 0.01 --indisteps 16 --log_name sample-16-0.01