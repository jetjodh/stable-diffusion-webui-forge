import numpy as np
import torch
import gradio as gr

from modules import scripts
import ldm_patched.ldm.modules.attention as attention
from lib.inversion import ddim_inversion

def adain(content_features, style_features):
    # Calculate mean and std of content features
    content_mean, content_std = torch.mean(content_features, dim=-1, keepdim=True), torch.std(content_features, dim=-1, keepdim=True)
    # Calculate mean and std of style features
    style_mean, style_std = torch.mean(style_features, dim=-1, keepdim=True), torch.std(style_features, dim=-1, keepdim=True)
    # Normalize content features and apply style mean and std
    normalized_features = style_std * (content_features - content_mean) / (content_std + 1e-5) + style_mean
    return normalized_features

def sdp(q, k, v, transformer_options):
    return attention.optimized_attention(q, k, v, heads=transformer_options["n_heads"], mask=None)


class StyleAlignForForge(scripts.Script):
    sorting_priority = 17

    def title(self):
        return "StyleAlign Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            shared_attention = gr.Checkbox(label='Share attention in batch', value=False)
            src_prompt = gr.Textbox(label="Source Prompt", placeholder="Enter a prompt")

        return [shared_attention, src_prompt]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.
        shared_attention = script_args[0]
        src_prompt = script_args[1]

        if p.init_images:
            input_image = p.init_images[0]
            latent_height, latent_width = p.init_latent.shape[2:]
            image = np.array(input_image.resize((latent_width, latent_height)))
        
            z = ddim_inversion(p.sd_model, image, src_prompt, 20, 2)


        if not shared_attention:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        def join(x):
            b, f, c = x.shape
            return x.reshape(1, b * f, c)

        def aligned_attention(q, k, v, transformer_options, z=None):
            b, f, c = q.shape
            if z:
                style_features = z[0:1]
            else:
                style_features = q[0:1]
            q = adain(q, style_features)
            k = adain(k, style_features)
            v = adain(v, style_features)
            o = sdp(join(q), join(k), join(v), transformer_options)
            b2, f2, c2 = o.shape
            o = o.reshape(b, b2 * f2 // b, c2)
            return o

        def attn1_proc(q, k, v, transformer_options, z=None):
            cond_indices = transformer_options['cond_indices']
            uncond_indices = transformer_options['uncond_indices']
            cond_or_uncond = transformer_options['cond_or_uncond']
            results = []

            for cx in cond_or_uncond:
                if cx == 0:
                    indices = cond_indices
                else:
                    indices = uncond_indices

                if len(indices) > 0:
                    bq, bk, bv = q[indices], k[indices], v[indices]
                    bo = aligned_attention(bq, bk, bv, transformer_options, z)
                    results.append(bo)

            results = torch.cat(results, dim=0)
            return results
        
        def attn1_proc_wrapper(q, k, v, transformer_options):
            return attn1_proc(q, k, v, transformer_options, z)

        unet.set_model_replace_all(attn1_proc_wrapper, 'attn1')

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            stylealign_enabled=shared_attention,
        ))

        return
