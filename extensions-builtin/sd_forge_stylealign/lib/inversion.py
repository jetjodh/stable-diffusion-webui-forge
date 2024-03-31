from __future__ import annotations
from typing import Callable
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.sd_samplers import create_sampler
from modules.shared import opts
import torch
from tqdm import tqdm
import numpy as np


T = torch.Tensor
TN = T | None
InversionCallback = Callable[[StableDiffusionXLPipeline, int, T, dict[str, T]], dict[str, T]]


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    if prompt == '':
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds


def _encode_text_sdxl(model: StableDiffusionProcessingTxt2Img, prompt: str) -> tuple[dict[str, T], T]:
    device = model._execution_device
    prompt_embeds, pooled_prompt_embeds, = _get_text_embeddings(prompt, model.cond_stage_model.tokenizer, model.text_encoder, device)
    prompt_embeds_2, pooled_prompt_embeds2, = _get_text_embeddings( prompt, model.tokenizer_2, model.text_encoder_2, device)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float16,
                                           text_encoder_projection_dim).to(device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": add_time_ids}
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(model: StableDiffusionProcessingTxt2Img, prompt: str) -> tuple[dict[str, T], T]:
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(model, "")
    prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds, ))
    added_cond_kwargs = {"text_embeds": torch.cat((added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])),
                         "time_ids": torch.cat((added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])),}
    return added_cond_kwargs, prompt_embeds


def _encode_image(model: StableDiffusionProcessingTxt2Img, image: np.ndarray) -> T:
    model.first_stage_model.to(dtype=torch.float32)
    # Ensure the image is in RGB format
    if image.shape[2] == 4:  # Check if the image has 4 channels
        image = image[:, :, :3]  # Keep only the first 3 channels
    image = torch.from_numpy(image).float() / 255.
    image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    model.first_stage_model.to(dtype=torch.float16)
    return latent


def _next_step(model: StableDiffusionProcessingTxt2Img, model_output: T, timestep: int, sample: T) -> T:
    sigmas = model.model_sampling.sigmas
    sigma, sigma_next = sigmas[timestep], sigmas[timestep + 1]
    
    next_original_sample = (sample - sigma ** 2 * model_output) / (1 - sigma ** 2) ** 0.5
    next_sample_direction = sigma_next * model_output
    next_sample = (1 - sigma_next ** 2) ** 0.5 * next_original_sample + next_sample_direction
    
    return next_sample


def _get_noise_pred(model: StableDiffusionProcessingTxt2Img, latent: T, t: T, context: T, guidance_scale: float, added_cond_kwargs: dict[str, T]):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.diffusion_model(latents_input, t, encoder_hidden_states=context, **added_cond_kwargs)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    return noise_pred


def _ddim_loop(model: StableDiffusionProcessingTxt2Img, z0, prompt, guidance_scale) -> T:
    all_latent = [z0]
    added_cond_kwargs, text_embedding = _encode_text_sdxl_with_negative(model, prompt)
    latent = z0.clone().detach().half()

    sigmas = model.sampler.sigmas

    for i in tqdm(range(model.steps)):
        t = sigmas[-(i+1)]
        noise_pred = _get_noise_pred(model, latent, t, text_embedding, guidance_scale, added_cond_kwargs)
        latent = _next_step(model, noise_pred, t, latent)
        all_latent.append(latent)

    return torch.cat(all_latent).flip(0)


def make_inversion_callback(zts, offset: int = 0) -> [T, InversionCallback]:

    def callback_on_step_end(pipeline: StableDiffusionXLPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    return  zts[offset], callback_on_step_end


@torch.no_grad()
def ddim_inversion(model: StableDiffusionProcessingTxt2Img, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:
    z0 = _encode_image(model, x0)
    model.sampler = create_sampler(opts.sampler_name, model.sd_model)
    model.sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0, verbose=False)
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs
