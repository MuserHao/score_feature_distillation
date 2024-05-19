from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
import gc
import os
from PIL import Image
from torchvision.transforms import PILToTensor
import gc

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

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

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > np.max(up_ft_indices):
                break

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
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        output = {}
        output['up_ft'] = up_ft
        return output
    
    def forward_super(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
            torch.set_grad_enabled(True)
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                # logger.info("Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

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

            t_emb1 = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb1.to(dtype=self.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            # 2. pre-process
            sample = self.conv_in(sample)

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # 4. mid
            if self.mid_block is not None:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

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
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )
                # print(i, up_ft_indices)
                if i in up_ft_indices:
                    from torch.autograd import Variable
                    # self.up_sample = Variable(sample, requires_grad=True)
                    sample.requires_grad_()
                    sample.retain_grad()
                    self.up_sample = sample
                    # self.up_sample.requires_grad_()
                    # self.up_sample = sample
                    
            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            
            return sample, t_emb1, emb
        
        
    def forward_grad(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        point = None
        ):
            self.img_size = image_size
            self.point = point
        
            torch.set_grad_enabled(True)
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                # logger.info("Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

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
            timesteps = timesteps.to(dtype=self.dtype)
            # for param in self.parameters():
            #     param.requires_grad = False
            
            from torch.autograd import Variable 
            timesteps=Variable(timesteps, requires_grad=True) # Enable gradient computation
            self.up_emb = timesteps
            self.sample = sample
            self.upsample_size = upsample_size
            self.index = 0

            def forward_loss(tt):
                sample = self.sample
                upsample_size = self.upsample_size
                t_emb1 = self.time_proj(tt)
                # timesteps does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb1.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)
                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                    emb = emb + class_emb

                # 2. pre-process
                sample = self.conv_in(sample)

                # 3. down
                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                    down_block_res_samples += res_samples

                # 4. mid
                if self.mid_block is not None:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]
                    # else:
                    #     upsample_size = self.upsample_size

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                        )
                    if i in up_ft_indices:
                        self.up_sample = sample
                if self.img_size != -1:
                    self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                if self.point != None:
                    self.up_sample = self.up_sample[0, :, point[1], point[0]]
                    self.feature_sample = self.up_sample.detach().clone()
                    self.up_sample = self.up_sample[self.index:self.index+self.interval]
                return self.up_sample
            
            import time
            # time1 = time.time()
            # from functorch import jacfwd, jacrev
            # gradients = jacfwd(forward_loss)(timesteps)
            # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
            # print(gradients.shape)
            time1 = time.time()
            gradients_res = torch.zeros((1280)).cuda()
            interval = 16
            self.interval = interval
            # out = forward_loss(timesteps)
            for i in range(0, 1280, interval):
                self.index = i
                
                # out = forward_loss(timesteps)
                # outt = out[i]
                # outt.backward(retain_graph=True)
                # gradients_res[i] = timesteps.grad.detach().clone()
                # time1 = time.time()
                gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                xx = gradients.detach().clone()
                gradients_res[i:i+interval] = xx[:, 0]
                self.zero_grad()
                if timesteps.grad is not None:
                    timesteps.grad.zero_()
                gc.collect()
                torch.cuda.empty_cache()
                # print(time.time() - time1)
            
            return self.feature_sample, gradients_res

    def forward_twograds(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        point = None
        ):
            self.img_size = image_size
            self.point = point
        
            torch.set_grad_enabled(True)
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                # logger.info("Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

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
            timesteps = timesteps.to(dtype=self.dtype)
            # for param in self.parameters():
            #     param.requires_grad = False
            
            from torch.autograd import Variable 
            timesteps=Variable(timesteps, requires_grad=True) # Enable gradient computation
            self.up_emb = timesteps
            self.sample = sample
            self.upsample_size = upsample_size
            self.index = 0

            def forward_loss(tt):
                sample = self.sample
                upsample_size = self.upsample_size
                t_emb1 = self.time_proj(tt)
                # timesteps does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb1.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)
                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                    emb = emb + class_emb

                # 2. pre-process
                sample = self.conv_in(sample)

                # 3. down
                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                    down_block_res_samples += res_samples

                # 4. mid
                if self.mid_block is not None:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]
                    # else:
                    #     upsample_size = self.upsample_size

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                        )
                    if i in up_ft_indices:
                        self.up_sample = sample
                if self.img_size != -1:
                    self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                    batch_size, channels, height, width = self.up_sample.shape
                    random_projection_matrix = torch.randn(1280, 128).to(self.up_sample.device)
                    data_reshaped = self.up_sample.permute(0, 2, 3, 1).reshape(-1, 1280)
                    projected_data_reshaped = torch.matmul(data_reshaped, random_projection_matrix)
                    self.up_sample = projected_data_reshaped.reshape(batch_size, height, width, 128).permute(0, 3, 1, 2)

                if self.point != None:
                    self.up_sample = self.up_sample[0, :, point[1], point[0]]
                    self.feature_sample = self.up_sample.detach().clone()
                    self.up_sample = self.up_sample[self.index:self.index+self.interval]
                return self.up_sample
            
            import time
            # time1 = time.time()
            # from functorch import jacfwd, jacrev
            # gradients = jacfwd(forward_loss)(timesteps)
            # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
            # print(gradients.shape)
            time1 = time.time()
            gradients_res = torch.zeros((128)).cuda()
            interval = 16
            self.interval = interval
            # out = forward_loss(timesteps)
            for i in range(0, 128, interval):
                self.index = i
                
                # out = forward_loss(timesteps)
                # outt = out[i]
                # outt.backward(retain_graph=True)
                # gradients_res[i] = timesteps.grad.detach().clone()
                # time1 = time.time()
                gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                xx = gradients.detach().clone()
                gradients_res[i:i+interval] = xx[:, 0]
                self.zero_grad()
                if timesteps.grad is not None:
                    timesteps.grad.zero_()
                gc.collect()
                torch.cuda.empty_cache()
                # print(time.time() - time1)
            
            return self.feature_sample, gradients_res

    

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):

        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output

    @torch.no_grad()
    def super_call(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet.forward_super(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output, latents, noise, latents_noisy, t
    
    @torch.enable_grad()
    def super_grad(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size=-1,
        point=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_grad(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               point=point)
        return gradient
    
    @torch.enable_grad()
    def super_twograds(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size=-1,
        point=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_twograds(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               point=point)
        return gradient

class SDFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', null_prompt=''):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self,
                img_tensor,
                prompt='',
                t=261,
                up_ft_index=1,
                ensemble_size=8):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds = self.pipe._encode_prompt(
                prompt=prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        return unet_ft


class SDFeaturizer4Eval(SDFeaturizer):
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', null_prompt='', cat_list=[]):
        super().__init__(sd_id, null_prompt)
        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                prompt = f"a photo of a {cat}"
                prompt_embeds = self.pipe._encode_prompt(
                    prompt=prompt,
                    device='cuda',
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False) # [1, 77, dim]
                cat2prompt_embeds[cat] = prompt_embeds
            self.cat2prompt_embeds = cat2prompt_embeds

        self.pipe.tokenizer = None
        self.pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()


    @torch.no_grad()
    def forward(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8):
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        return unet_ft
    
    @torch.no_grad()
    def forward_super(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8):
        torch.set_grad_enabled(True)
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        unet_ft_all = self.pipe.super_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)

        return unet_ft_all
    
    @torch.enable_grad()
    def forward_grad(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                point=None):
        torch.set_grad_enabled(True)
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        grad = self.pipe.super_grad(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            point=point)

        return grad
    
    @torch.enable_grad()
    def forward_twograds(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                point=None):
        torch.set_grad_enabled(True)
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        grad = self.pipe.super_twograds(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            point=point)

        return grad