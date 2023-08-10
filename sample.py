from ldm.models.diffusion.dpm_solver import (
    get_mid_latent_code,
    model_wrapper,
    NoiseScheduleVP,
    DPM_Solver
)
import pandas as pd
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
import random
import nltk
import time
import argparse, os

import torch

from omegaconf import OmegaConf
from PIL import Image

from einops import  repeat

from contextlib import nullcontext

from pytorch_lightning import seed_everything


from ldm.util import instantiate_from_config

from torch import autocast

from collections import defaultdict

import numpy as np
from ldm.modules.attention import get_global_heat_map, clear_heat_maps, get_rank, edit_rank, clear_rank, \
    clear_lamb, edit_lamb, next_heat_map, add_maps,dont_add_rank,add_rank,dont_add_maps,get_global_heat_map_pic


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--prompt_edit",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to be edit"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--origin_image",
        type=str,
        help="the path of image to be edit",
        default=None,
        required=True
    )

    # opt = parser.parse_args()
    opt = parser.parse_args(args=[
        "--ddim_eta", "0.0",
        "--n_samples", "1",
        "--scale", "10.0",
        "--ddim_steps", "20",
        "--seed", "42",
        "--ckpt", "./models/ldm/stable-diffusion-v1/model.ckpt",#ckpt path
        "--prompt", "",
        "--origin_image", ""
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    return model, opt


model, opt = main()


def load_img(path, opt):
    image = Image.open(path).convert("RGB")
    image = image.resize((opt.W, opt.H), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def latent_to_image(model, latents):
    x_samples = model.decode_first_stage(latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    x_samples = 255. * x_samples
    x_samples = x_samples.astype(np.uint8)

    return x_samples


def repeat_tensor(x, n, dim=0):
    dims = len(x.shape) * [1]
    dims[dim] = n
    return x.repeat(dims)


def get_map(model, src, dst, init_latent, n: int, ddim_steps,
             clamp_rate: float = 3):
    """
    the map value will be clamped to map.mean() * clamp_rate, then values will be scaled into 0~1, then term into binary(split at 0.5). so if a map value is large than map.mean() * clamp_rate * 0.5 will be encode to 1, less will be encode to 0.
    so the larger clamp rate is, less pixes will be encode to 1, the small clamp rate is, the more pixes will be encode to 1.
    """

    device = model.device
    repeated = repeat_tensor(init_latent, n)
    dst = repeat_tensor(dst, n)
    noise = torch.randn(init_latent.shape, device=device)
    scheduler = DDIMScheduler(num_train_timesteps=model.num_timesteps,
                              trained_betas=model.betas.cpu().numpy())
    scheduler.set_timesteps(ddim_steps, device=device)
    noised = scheduler.add_noise(repeated, noise,
                                 scheduler.timesteps[ddim_steps // 2]
                                 )
    t = scheduler.timesteps[ddim_steps // 2]
    t_ = torch.unsqueeze(t, dim=0).to(device)
    clear_heat_maps()
    add_maps()
    pre_dst = model.apply_model(noised, t_, dst)
    next_heat_map()
    dont_add_maps()



# modify sample method of dpm_solver, most code are copied from https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py
# here we just record sample lantents and apply mask in sample process
def sample_edit(self, x, steps=20, t_start=None, t_end=None, order=3, skip_type='time_uniform',
                method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpm_solver',
                atol=0.0078, rtol=0.05, record_list=None, mask=None,record_denoise=False,denoise_list=None
                ):
    t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
    t_T = self.noise_schedule.T if t_start is None else t_start
    device = x.device
    if record_list is not None:
        assert len(record_list) == steps
    if method == 'adaptive':
        with torch.no_grad():
            x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol,
                                         solver_type=solver_type)
    elif method == 'multistep':
        assert steps >= order
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                     solver_type=solver_type) # TODO 也许这里可视化
                next_heat_map()
                if mask is not None and record_list is not None:
                    x = record_list[init_order - 1].to(device) * (1. - mask) + x * mask

                if record_denoise:
                    denoise_list.append(x)
                model_prev_list.append(self.model_fn(x, vec_t))
                t_prev_list.append(vec_t)

            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                vec_t = timesteps[step].expand(x.shape[0])
                if lower_order_final and steps < 15:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, step_order,
                                                     solver_type=solver_type)
                next_heat_map()
                if mask is not None and record_list is not None:
                    x = record_list[step - 1].to(device) * (1. - mask) + x * mask
                if record_denoise:
                    denoise_list.append(x)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                if step==int((steps+1)/2):

                    dont_add_rank()
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, vec_t)
    elif method in ['singlestep', 'singlestep_fixed']:
        if method == 'singlestep':
            timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order,
                                                                                          skip_type=skip_type, t_T=t_T,
                                                                                          t_0=t_0, device=device)
        elif method == 'singlestep_fixed':
            K = steps // order
            orders = [order, ] * K
            timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
        for i, order in enumerate(orders):
            t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
            timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=t_T_inner.item(), t_0=t_0_inner.item(),
                                                  N=order, device=device)
            lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
            vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(x.shape[0])
            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
            x = self.singlestep_dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2)
    if denoise_to_zero:
        x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
    return x


def sample(self, x, steps=20, t_start=None, t_end=None, order=3, skip_type='time_uniform',
           method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpm_solver',
           atol=0.0078, rtol=0.05, record_process=False, record_list=None
           ):
    t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
    t_T = self.noise_schedule.T if t_start is None else t_start
    device = x.device
    if method == 'adaptive':
        with torch.no_grad():
            x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol,
                                         solver_type=solver_type)
    elif method == 'multistep':
        assert steps >= order
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                     solver_type=solver_type)
                if record_process:
                    record_list.append(x.cpu())
                model_prev_list.append(self.model_fn(x, vec_t))
                t_prev_list.append(vec_t)

            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                vec_t = timesteps[step].expand(x.shape[0])
                if lower_order_final and steps < 15:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, step_order,
                                                     solver_type=solver_type)
                if record_process:
                    record_list.append(x.cpu())
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, vec_t)
    elif method in ['singlestep', 'singlestep_fixed']:
        if method == 'singlestep':
            timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order,
                                                                                          skip_type=skip_type, t_T=t_T,
                                                                                          t_0=t_0, device=device)
        elif method == 'singlestep_fixed':
            K = steps // order
            orders = [order, ] * K
            timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
        for i, order in enumerate(orders):
            t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
            timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=t_T_inner.item(), t_0=t_0_inner.item(),
                                                  N=order, device=device)
            lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
            vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(x.shape[0])
            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
            x = self.singlestep_dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2)
    if denoise_to_zero:
        x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
    return x


def diffedit(model, init_image,
             src_prompt: str = "",
             dst_prompt: str = "",
             encode_ratio: float = 0.6,
             ddim_steps: int = 20,
             seed=42,
             scale: float = 7.5,
             precision="autocast"):
    """
    :param init_image: image to be edit
    :param src_prompt: prompt describe origin image(i.e. A bowl of fruits)
    :param dst_prompt: prompt describe desired image(i.e. A bowl of pears)
    :param encode_ratio: how deep to encode origin image, must between 0-1
    :param ddim_steps: total ddim steps, actual encode steps = ddim_steps * encode ratio
    :param seed: random seed
    :param scale: classifier free guidance scale
    :param precision: ema precision
    """
    # If seed is None, randomly select seed from 0 to 2^32-1

    if seed is None:
        seed = random.randrange(2 ** 32 - 1)
    seed_everything(seed)
    device = model.device

    model.cond_stage_model = model.cond_stage_model.to(device)
    precision_scope = autocast if precision == "autocast" else nullcontext
    assert os.path.isfile(init_image)
    init_image = load_img(init_image, opt).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)

    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():

                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning([""])

                dst = model.get_learned_conditioning([dst_prompt])
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

                add_rank()  # Confirm attention map injection.
                clear_rank() # Clear reserved attention maps.

                # get map
                get_map(model, None, dst, init_latent, 1, 2)

                heat_maps = get_global_heat_map()

                # Computed gamma γ
                b = [float(heat_maps[i].sum() / (64 * 64)) for i in range(1, len(dst_prompt.split(' ')) + 1)]
                b_exp = np.exp(b)
                b_sum = np.sum(b_exp, axis=0, keepdims=True)
                b_weight = b_exp / b_sum



                ns = NoiseScheduleVP('discrete', betas=model.betas)
                model_fn = model_wrapper(
                    lambda x, t, c: model.apply_model(x, t, c),
                    ns,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=uc,
                    unconditional_condition=uc,
                    guidance_scale=scale
                )


                noiser = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
                noiser.sample = sample.__get__(noiser, type(noiser))

                noised_sample = noiser.sample(
                    init_latent,
                    t_start=1. / model.num_timesteps,
                    t_end=encode_ratio,
                    method='multistep',
                    order=2,
                    steps=ddim_steps,
                    record_process=False,

                )

                rank = defaultdict(list)
                source_aware="A photo of a "
                edit_aware=""
                for ij in (np.argsort(-1*b_weight)):
                    word=dst_prompt.split(' ')[ij]
                    val = nltk.pos_tag([word])[0][1]  # Determine whether it is a noun
                    if source_aware=="A photo of a " and(val == 'NN' or val == 'NNS' or val == 'NNPS' or val == 'NNP'):
                        source_aware+=word
                    else:
                        edit_aware+=word
                        rank[ij + 1] = [b_weight[ij], heat_maps[ij + 1]]

                edit_rank(rank)
                print("dst_prompt,b_weight:",dst_prompt,b_weight)
                print("edit_aware", edit_aware) # The part to be modified or strengthened
                print("source_aware", source_aware)  # The part that is more relevant to the image

                # source_aware condition
                src_con = model.get_learned_conditioning(1 * [source_aware])


                # perform step wise edit
                model_fn_dst = model_wrapper(
                    lambda x, t, c: model.apply_model(x, t, c),
                    ns,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=dst,
                    src_condition=src_con,
                    unconditional_condition=uc,
                    guidance_scale=scale
                )
                solver = DPM_Solver(model_fn_dst, ns, predict_x0=True, thresholding=False)
                clear_heat_maps()


                solver.sample_edit = sample_edit.__get__(solver, type(solver))

                recover = solver.sample_edit(
                    noised_sample,
                    t_start=encode_ratio,
                    t_end=1. / model.num_timesteps,
                    method='multistep',
                    order=2,
                    steps=ddim_steps,
                    record_denoise=False,
                )

                images = latent_to_image(model, recover)
                return images




import clip
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)


lamb=0.7
edit_lamb(lamb) #lambda


res = diffedit(model,"./data/cat.png",dst_prompt="A cute cat playing with a long pearl necklace.")
Image.fromarray(res[0]).save(f"./output/{lamb}.png")





