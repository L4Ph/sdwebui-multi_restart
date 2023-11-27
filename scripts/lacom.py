import json
import torch
import tqdm
import k_diffusion.sampling
from pathlib import Path
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers

NAME = "LACOM Restart"
ALIAS = "LACOMRestart"
ROOT_DIR = Path().absolute()  # Webui root path


def load_config():
    config_path = ROOT_DIR / "config.json"

    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file: {e}")


config = load_config()


@torch.no_grad()
def lcm_sampler(
    model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = (
        k_diffusion.sampling.default_noise_sampler(x)
        if noise_sampler is None
        else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])
    for i in tqdm.auto.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x


def lacom_restart_sampler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=config["s_noise"],
    restart_list=None,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0
    from k_diffusion.sampling import to_d, get_sigmas_karras

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": step_id,
                    "sigma": new_sigma,
                    "sigma_hat": old_sigma,
                    "denoised": denoised,
                }
            )
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    restart_steps = int(config["restart_steps"])
    if restart_list is None:
        if steps >= restart_steps * 2:
            restart_times = 1
            if steps >= restart_steps * 3:
                restart_times = steps // restart_steps - 1
            sigmas = get_sigmas_karras(
                steps - restart_steps * restart_times,
                sigmas[-2].item(),
                sigmas[0].item(),
                device=sigmas.device,
            )
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {
        int(torch.argmin(abs(sigmas - key), dim=0)): value
        for key, value in restart_list.items()
    }

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(
                    restart_steps,
                    sigmas[min_idx].item(),
                    sigmas[max_idx].item(),
                    device=sigmas.device,
                )[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(
                        [
                            (old_sigma, new_sigma)
                            for (old_sigma, new_sigma) in zip(
                                sigma_restart[:-1], sigma_restart[1:]
                            )
                        ]
                    )

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = (
                x
                + k_diffusion.sampling.torch.randn_like(x)
                * s_noise
                * (old_sigma**2 - last_sigma**2) ** 0.5
            )
        # result_lcm_sampler の結果を使用
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


if not NAME in [x.name for x in sd_samplers.all_samplers]:
    lacom_restart_samplers = [(NAME, lacom_restart_sampler, [ALIAS], {})]
    samplers_data_multi_restart = [
        sd_samplers_common.SamplerData(
            label,
            lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(
                funcname, model
            ),
            aliases,
            options,
        )
        for label, funcname, aliases, options in lacom_restart_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += samplers_data_multi_restart
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}

lcm_result = lcm_sampler(callback=lacom_restart_sampler)