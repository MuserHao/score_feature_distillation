import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import inspect
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from typing import Union, Optional, Any, Dict, List

class MyUNet2DConditionModel(UNet2DConditionModel):
    def name(self):
        return "Myunet"
    
    
    def forward_t(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # torch.set_grad_enabled(True)
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        self.sample = sample
        self.encoder_hidden_states = encoder_hidden_states
        self.added_cond_kwargs = added_cond_kwargs
        self.cross_attention_kwargs = cross_attention_kwargs
        self.down_intrablock_additional_residuals = down_intrablock_additional_residuals
        self.upsample_size=upsample_size
        
        def forward_t_gradient(tt):
            upsample_size = self.upsample_size
            down_intrablock_additional_residuals = self.down_intrablock_additional_residuals
            cross_attention_kwargs = self.cross_attention_kwargs
            added_cond_kwargs = self.added_cond_kwargs
            encoder_hidden_states = self.encoder_hidden_states
            sample = self.sample
            t_emb = self.time_proj(tt)
            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)
            
            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
            if class_emb is not None:
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            aug_emb = self.get_aug_embed(
                emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
            if self.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )

            # 2. pre-process
            sample = self.conv_in(sample)

            # 2.5 GLIGEN position net
            if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

            # 3. down
            # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
            # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
            if cross_attention_kwargs is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                lora_scale = cross_attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0


            is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None
            # maintain backward compatibility for legacy usage, where
            #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
            #       but can only use one or the other
            if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                
                down_intrablock_additional_residuals = down_block_additional_residuals
                is_adapter = True

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            self.res = sample
            return sample
        # sample = forward_t_gradient(timesteps)
        from torch.autograd import Variable 
        timesteps=Variable(timesteps.to(sample.dtype), requires_grad=True) # Enable gradient computation
        from functorch import jacfwd, jacrev, vmap
        gradients = jacfwd(forward_t_gradient)(timesteps)
        # print(self.res.shape, gradients.shape)
        # self.res += 10* gradients[:, :, :, :, 0]
        
        return self.res, gradients[:, :, :, :, 0]
    
    def forward_z(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # torch.set_grad_enabled(True)
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        self.timesteps = timesteps
        self.sample = sample
        self.encoder_hidden_states = encoder_hidden_states
        self.added_cond_kwargs = added_cond_kwargs
        self.cross_attention_kwargs = cross_attention_kwargs
        self.down_intrablock_additional_residuals = down_intrablock_additional_residuals
        self.upsample_size=upsample_size
        
        def forward_z_gradient(sample):
            upsample_size = self.upsample_size
            down_intrablock_additional_residuals = self.down_intrablock_additional_residuals
            cross_attention_kwargs = self.cross_attention_kwargs
            added_cond_kwargs = self.added_cond_kwargs
            encoder_hidden_states = self.encoder_hidden_states
            # sample = self.sample
            tt = self.timesteps
            t_emb = self.time_proj(tt)
            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)
            
            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
            if class_emb is not None:
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            aug_emb = self.get_aug_embed(
                emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
            if self.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )

            # 2. pre-process
            sample = self.conv_in(sample)

            # 2.5 GLIGEN position net
            if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

            # 3. down
            # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
            # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
            if cross_attention_kwargs is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                lora_scale = cross_attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0


            is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None
            # maintain backward compatibility for legacy usage, where
            #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
            #       but can only use one or the other
            if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                
                down_intrablock_additional_residuals = down_block_additional_residuals
                is_adapter = True

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            self.res = sample
            sample = torch.mm(sample.reshape(1, -1), sample.reshape(-1, 1))
            return sample
        # sample = forward_t_gradient(timesteps)
        from torch.autograd import Variable 
        sample=Variable(sample.to(sample.dtype), requires_grad=True) # Enable gradient computation
        from functorch import jacfwd, jacrev, vmap
        gradients = jacrev(forward_z_gradient)(sample)
        # print(gradients.shape, self.res.shape)
        # print(f'xxxxxnorm', torch.norm(gradients))
        # print(f'yyyyynorm', torch.norm(self.res))
        # print(self.res.shape, gradients.shape)
        self.res = gradients[0, 0, :, :, :, :] - self.res
        return self.res
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class OnePipe(StableDiffusionPipeline):
    
    def name(self):
        return "onepipe"
    
    def set_extras(self, timeline=-1, extrap1=False, extrap2=False, pos1=False, pos2=False, t_w1=1, z_w1=0.005, t_w2=1, z_w2=0.005):
        # 0 - timeline, perform pos (+) or neg (-) gradients of t and z
        
        self.timeline = timeline
        self.extrap1 = extrap1
        self.extrap2 = extrap2
        self.pos1 = pos1
        self.pos2 = pos2
        self.t_w1 = t_w1
        self.z_w1 = z_w1
        self.t_w2 = t_w2
        self.z_w2 = z_w2
        
    def reset_extras(self):
        self.timeline = -1
        self.extrap1 = False
        self.extrap2 = False
        self.pos1 = False
        self.pos2 = False
        self.t_w1 = 1
        self.z_w1 = 0.005
        self.t_w2 = 1
        self.z_w2 = 0.005
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=None,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        # print(f')()))))', self.scheduler.__class__.__name__)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred, t_gradient = self.unet.forward_t(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=None,
                    return_dict=False,
                )
                z_gradient = self.unet.forward_z(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=None,
                    return_dict=False,
                )
                if self.timeline != -1:
                    if self.extrap1:
                        if t < self.timeline:
                            if self.pos1:
                                noise_pred = noise_pred + self.t_w1* t_gradient + self.z_w1 * z_gradient
                            else:
                                noise_pred = noise_pred - self.t_w1 * t_gradient - self.z_w1 * z_gradient
                    if self.extrap2:
                        if t >= self.timeline:
                            if self.pos2:
                                noise_pred = noise_pred + self.t_w2 * t_gradient + self.z_w2 * z_gradient
                            else:
                                noise_pred = noise_pred - self.t_w2 * t_gradient - self.z_w2 * z_gradient
                            
                # print(torch.norm(noise_pred), torch.norm(t_gradient), torch.norm(z_gradient))
                # print(noise_pred.mean(), noise_pred.std(), t_gradient.mean(), t_gradient.std(), z_gradient.mean(), z_gradient.std())
                # beta_t = self.scheduler.betas[t]
                # # print(beta_t)
                # cc = noise_pred[1:2, :, :, :]
                # noise_pred = noise_pred - 1* t_gradient - 0.005 * z_gradient
                # noise_pred = noise_pred + 1* t_gradient + 0.005 * z_gradient
                # xx = (0.01 / 500**2) ** 0.5
                # if t < 200:
                    # noise_pred = noise_pred - 1* t_gradient - 0.005 * z_gradient
                    # noise_pred = noise_pred + 1* t_gradient + 0.005 * z_gradient
                # else:
                    # noise_pred = noise_pred + 1* t_gradient + 0.005 * z_gradient
                    # noise_pred = noise_pred - 1* t_gradient - 0.005 * z_gradient
                
                # noise_pred[1, :, :, :] = cc
                # noise_pred += 10* t_gradient
                # noise_pred += - 0.01*z_gradient

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
 

        assert not output_type == "latent"
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        return image

class SDGrader():
    def __init__(self, pretrained_model_name_or_path, device=torch.device('cuda')):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        unet = MyUNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.one_pipe = OnePipe.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
        self.one_pipe = self.one_pipe.to(self.device)
        
    def run(self, prompt, guidance_scale=7.5, generator=None, num_inference_steps=100):
        self.one_pipe.extrap = False
        image = self.one_pipe(prompt, guidance_scale=guidance_scale, generator=generator, num_inference_steps=num_inference_steps)[0]
        print(image.size)    
        image.save('temp48.jpg')
        
    def run_hpsv2(self, guidance_scale=7.5, height=512, width=512, generator=None, num_inference_steps=50, save_folder='./'):
        import hpsv2
        import os
        if guidance_scale > 0:
            save_folder = os.path.join(save_folder, 'guid' + str(guidance_scale), f'h{height}w{width}')
        else:
            save_folder = os.path.join(save_folder, 'woguid', f'h{height}w{width}')
        # os.makedirs(save_folder, exist_ok=True)
        all_prompts = hpsv2.benchmark_prompts('all') 
        prompt_keys = ['anime', 'concept-art', 'paintings', 'photo']
        all_prompts = all_prompts['photo'][:50]
        #         self.timeline = timeline
        # timeline extrap1 extrap2 pos1 pos2 t_w1 z_w1 t_w2 z_w2
        # kwargs1 = {'timeline':-1}
        # kwargs2 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        # kwargs3 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        # kwargs4 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        # kwargs5 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        
        # kwargs6 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        # kwargs7 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs8 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        # kwargs9 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
        # kwargs10 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
        # kwargs11 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        
        # kwargs35 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0, 't_w2':1, 'z_w2':0}
        # kwargs36 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0, 't_w2':10, 'z_w2':0}

        # kwargs = [kwargs1, kwargs2, kwargs3, kwargs4, kwargs5, kwargs6, kwargs7, kwargs8, kwargs9, kwargs10, kwargs11, kwargs35, kwargs36]
        
        # kwargs12 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        # kwargs13 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        # kwargs14 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        # kwargs15 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        
        # kwargs16 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
        # kwargs17 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs18 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
        # kwargs19 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
        # kwargs20 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
        # kwargs21 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}

        # kwargs = [kwargs12, kwargs13, kwargs14, kwargs15, kwargs16, kwargs17, kwargs18, kwargs19, kwargs20, kwargs21]
        
        # kwargs22 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':-0.005}
        # kwargs23 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs24 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':-0.002}
        # kwargs25 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.002, 't_w2':10, 'z_w2':0.002}
        
        # kwargs26 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':-0.005}
        # kwargs27 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs28 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':-0.002}
        # kwargs29 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.002, 't_w2':10, 'z_w2':0.002}
        
        # kwargs30 = {'timeline':300, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs31 = {'timeline':400, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs32 = {'timeline':500, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}        
        # kwargs33 = {'timeline':200, 'extrap1':True, 'extrap2':False, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs34 = {'timeline':200, 'extrap1':False, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
        # kwargs = [kwargs22, kwargs23, kwargs24, kwargs25, kwargs26, kwargs27, kwargs28, kwargs29, kwargs30, kwargs31, kwargs32, kwargs33, kwargs34]
        
        kwargs37 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.005, 't_w2':1, 'z_w2':-0.002}
        
        kwargs38 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':-0.005, 't_w2':1, 'z_w2':-0.002}
        kwargs39 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':-0.005, 't_w2':10, 'z_w2':-0.002}
        kwargs40 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':-0.002}
        
        kwargs41 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.005, 't_w2':1, 'z_w2':-0.005}
        kwargs42 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.002, 't_w2':1, 'z_w2':-0.002}
        
        kwargs43 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':-0.002}
        kwargs44 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.005, 't_w2':1, 'z_w2':-0.002}
        kwargs45 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':-0.005, 't_w2':10, 'z_w2':-0.002}
        kwargs46 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':-0.005, 't_w2':1, 'z_w2':-0.002}
        kwargs47 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':-0.005}
        kwargs48 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.002, 't_w2':10, 'z_w2':-0.002}

        kwargs = [kwargs37, kwargs38, kwargs39, kwargs40, kwargs41, kwargs42, kwargs43, kwargs44, kwargs45, kwargs46, kwargs47, kwargs48]
        
        for kwarg in kwargs:
            self.one_pipe.reset_extras()
            self.one_pipe.set_extras(**kwarg)
            cus_save_folder = save_folder
            for k,v in kwarg.items():
                cus_save_folder +=  '_' + k+'_'+str(v)
            os.makedirs(cus_save_folder, exist_ok=True)
            pbar = tqdm(total=len(all_prompts), desc='Steps')
            for idx, prompt in enumerate(all_prompts):
                image = self.one_pipe(prompt.strip(), guidance_scale=guidance_scale, height=height, width=width, generator=generator, num_inference_steps=num_inference_steps)[0]
                image.save(os.path.join(cus_save_folder, f'{idx}.jpg'))
                pbar.update(1)
            # for style, prompts in all_prompts.items():
            #     for idx, prompt in enumerate(prompts):
            #         print(style, idx, prompt)
        

if __name__ == '__main__':
    generator = torch.Generator("cuda").manual_seed(1024)   
    
    # prompt = 'A dog is running.'
    prompt = 'fat rabbits with oranges, photograph iphone, trending'
    model_id = "stabilityai/stable-diffusion-2"
    model = SDGrader(pretrained_model_name_or_path=model_id)
    # model.run(prompt, guidance_scale=7.5, generator=generator, num_inference_steps=50)
    model.run_hpsv2(generator=generator)