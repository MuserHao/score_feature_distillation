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
    
    def forward_grad1(self,
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
        
        
    def forward_grad2(self,
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

    def forward_grad3(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        points = None
        ):
            self.img_size = points[0]
            self.points = points
        
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

                if self.points != None:
                    npy_points = np.array(self.points[1:])
                    up_sample = self.up_sample[0, :, npy_points[:, 1], npy_points[:, 0]]
                    self.feature_sample = up_sample.detach().clone()
                    self.up_sample = up_sample[self.index:self.index+self.interval]
                return self.up_sample
            
            
            from functorch import jacfwd, jacrev, vmap
            import time
            time1 = time.time()
            gradients_res = torch.zeros((1280, len(self.points[1:]))).cuda()
            interval = 32
            self.interval = interval
            for i in range(0, 1280, interval):
                # time1 = time.time()
                self.index = i
                # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                # gradients = vmap(jacfwd(forward_loss))(self.up_emb)
                gradients = jacfwd(forward_loss)(timesteps)
                xx = gradients.detach().clone()
                gradients_res[i:i+interval] = xx[:, :, 0]
                self.zero_grad()
                if self.up_emb.grad is not None:
                    self.up_emb.grad.zero_()
                del gradients
                del xx
                gc.collect()
                torch.cuda.empty_cache()
                # print(time.time() - time1)
            
            return self.feature_sample, gradients_res
        
    
    def forward_grad4(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        points = None
        ):
            self.img_size = image_size
        
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
                self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                self.feature_sample = self.up_sample[0, :, :, :].detach().clone()
                self.size1 = self.feature_sample.shape[-2]
                self.size2 = self.feature_sample.shape[-1]
                self.up_sample = self.up_sample[0, self.index:self.index+self.interval, :, :]
                return self.up_sample
            
            from functorch import jacfwd, jacrev, vmap
            import time
            time1 = time.time()
            gradients_res = torch.zeros((1280, self.img_size[0], self.img_size[1])).cuda()
            # gradients_res = torch.zeros((1280, 48, 48)).cuda()
            interval = 1280
            self.interval = interval
            for i in range(0, 1280, interval):
                # time1 = time.time()
                self.index = i
                # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                # gradients = vmap(jacfwd(forward_loss))(self.up_emb)
                gradients = jacfwd(forward_loss)(timesteps)
                xx = gradients.detach().clone()
                gradients_res[i:i+interval] = xx[:, :, :, 0]
                self.zero_grad()
                if self.up_emb.grad is not None:
                    self.up_emb.grad.zero_()
                del gradients
                del xx
                gc.collect()
                torch.cuda.empty_cache()
                # print(time.time() - time1)
            
            return self.feature_sample, gradients_res
        
    def forward_grad5_backup(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        points = None
        ):
            self.img_size = image_size
        
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
            sample=Variable(sample, requires_grad=True) # Enable gradient computation
            # timesteps = Variable(timesteps, requires_grad=True) # Enable gradient computation
            self.timesteps = timesteps
            self.sample = sample
            self.upsample_size = upsample_size
            self.index = 0

            def forward_sample(sample):
                sample = self.sample
                upsample_size = self.upsample_size
                timesteps_ = self.timesteps
                t_emb1 = self.time_proj(timesteps_)
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
                # self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                self.feature_sample = self.up_sample[0, :, :, :].detach().clone()
                self.size1 = self.feature_sample.shape[-2]
                self.size2 = self.feature_sample.shape[-1]
                self.up_sample = self.up_sample[0, self.index:self.index+self.interval, :, :]
                return self.up_sample
            
            from functorch import jacfwd, jacrev, vmap
            import time
            time1 = time.time()
            gradients_res = torch.zeros((1280, 48, 48, 1, 4, 96, 96))
            # gradients_res = torch.zeros((1280, 48, 48)).cuda()
            interval = 64
            self.interval = interval
            for i in range(0, 1280, interval):
                # time1 = time.time()
                self.index = i
                # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                # gradients = vmap(jacfwd(forward_loss))(self.up_emb)
                print(sample.shape)
                gradients = jacfwd(forward_sample)(sample).detach().cpu()
                print(gradients.shape)
                # yy = gradients.cpu().detach()
                # xx = gradients.cpu()
                gradients_res[i:i+interval] = gradients
                self.zero_grad()
                if self.sample.grad is not None:
                    self.sample.grad.zero_()
                del gradients
                # del xx
                gc.collect()
                torch.cuda.empty_cache()
                # print(time.time() - time1)
            
            return self.feature_sample, gradients_res

    def forward_grad5(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        points = None
        ):
            # up_ft_indices = [2]
            self.img_size = image_size
        
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
            copy_sample = sample.clone()
            copy_sample = copy_sample.cpu()
            pp_timesteps = timesteps
            sample=Variable(sample, requires_grad=True) # Enable gradient computation
            timesteps = Variable(timesteps, requires_grad=True) # Enable gradient computation
            self.timesteps = timesteps
            self.sample = sample
            self.upsample_size = upsample_size
            self.index = 0

            def forward_sample(timesteps, sample):
                # sample = self.sample
                # print(sample.shape, self.sample.shape)
                # sample = torch.cat([self.sample[0:1, 1:, :, :], sample], dim=1)
                # timesteps = self.timesteps
                upsample_size = self.upsample_size
                timesteps_ = timesteps
                t_emb1 = self.time_proj(timesteps_)
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
                # self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                self.feature_sample = self.up_sample[0, :, :, :].detach().cpu().clone()
                self.size1 = self.feature_sample.shape[-2]
                self.size2 = self.feature_sample.shape[-1]
                self.up_sample = self.up_sample[0]
                if self.flag is True:
                    self.up_sample = self.up_sample.mean(dim=0)
                    self.up_sample = self.up_sample[self.index2:self.index2+self.interval2, self.index3:self.index3+self.interval3]
                return self.up_sample
            
            from functorch import jacfwd, jacrev, vmap
            # import time
            # time1 = time.time()
            
            self.flag = True
            self.interval1 = 1280
            self.interval2 = 1
            self.interval3 = 2
            interval = 8
            gradients_z = torch.zeros((48//interval*self.interval2, 48//interval*self.interval3))
            import time
            time1 = time.time()
            from tqdm.auto import tqdm
            xxx = 48//interval*48//interval
            pbar = tqdm(total=xxx)
            for i in range(0, 1280, self.interval1):
                for j in range(0, 48, interval):
                    for k in range(0, 48, interval):
                        pbar.update(1)
                        # time1 = time.time()
                        self.index1 = i
                        self.index2 = j
                        self.index3 = k
                        # gradients = torch.autograd.functional.jacobian(forward_loss, timesteps)
                        # gradients = vmap(jacfwd(forward_loss))(self.up_emb)
                        # time1 = time.time()
                        gradients = jacrev(forward_sample, 1)(timesteps, sample)
                        
                        xx = torch.mm(gradients.reshape(self.interval2*self.interval3, -1), sample.reshape(-1, 1)).reshape(self.interval2, self.interval3)
                        s_g = xx.detach().cpu().clone()
                        
                        # gradients_t[i:i+interval] = t_g[:, :, :, 0]
                        gradients_z[self.index2//interval:self.index2//interval+self.interval2, self.index3//interval:self.index3//interval+self.interval3] = s_g
                        self.zero_grad()
                        if sample.grad is not None:
                            sample.grad.zero_()
                        if timesteps.grad is not None:
                            timesteps.grad.zero_()
                        del gradients
                        # del t_g
                        del s_g
                        del xx
                        gc.collect()
                        torch.cuda.empty_cache()
            self.zero_grad()
            if sample.grad is not None:
                sample.grad.zero_()
            if timesteps.grad is not None:
                timesteps.grad.zero_()
            gc.collect()
            torch.cuda.empty_cache()
            time2 = time.time()
            print(time2-time1)
            
            gradients_t = torch.zeros((1280, 48, 48))
            # self.interval = 1280
            self.flag = False
            gradients = jacfwd(forward_sample)(timesteps, sample)
            print(gradients.shape)
            gradients_t = gradients.detach().cpu().clone()
            del gradients
            self.zero_grad()
            if sample.grad is not None:
                sample.grad.zero_()
            if timesteps.grad is not None:
                timesteps.grad.zero_()
            # print(time2 - time1)
            # gradients_z = gradients_z.reshape(1280*48*48, -1)
            # copy_sample = copy_sample.reshape(-1, 1)
            # gradients_z_sam = torch.mm(gradients_z, copy_sample)
            # gradients_z_sam = gradients_z_sam.reshape(1280, 48, 48)
            # print(time.time() - time1)
            return self.feature_sample, gradients_t.cpu(), gradients_z.cpu()

    def forward_grad6(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_size: Optional[int] = -1,
        points = None,
        repeat_direc = None
        ):
            self.img_size = image_size
            self.repeat_direc = repeat_direc
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
            self.sample = sample
            feature_map_size = 1280*48*48
            input_sample_size = 1*4*96*96
            if feature_map_size % input_sample_size != 0:
                raise ValueError('feature map size should be integer dividble to input sample size')
            feat_times = feature_map_size // input_sample_size
            if self.repeat_direc == 'bs':
                sample = sample.repeat(feat_times, 1, 1, 1)
            elif self.repeat_direc == 'ch':
                s1, s2, s3, s4 = sample.shape
                sample = sample.repeat(1, feat_times, 1, 1)
                sample = sample.reshape(feat_times, s2, s3, s4)
            else:
                raise NotImplementedError
            
            from torch.autograd import Variable 
            copy_sample = sample.clone()
            copy_sample = copy_sample.cpu()
            pp_timesteps = timesteps
            sample=Variable(sample, requires_grad=True) # Enable gradient computation
            timesteps = Variable(timesteps, requires_grad=True) # Enable gradient computation
            self.timesteps = timesteps
            # self.sample = sample
            self.upsample_size = upsample_size
            self.index = 0

            def forward_sample(timesteps, input_sample):
                # sample = self.sample
                # print(sample.shape, self.sample.shape)
                # sample = torch.cat([self.sample[0:1, 1:, :, :], sample], dim=1)
                # timesteps = self.timesteps
                sample = input_sample.mean(dim=0, keepdim=True)
                upsample_size = self.upsample_size
                timesteps_ = timesteps
                t_emb1 = self.time_proj(timesteps_)
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
                # self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                # self.feature_sample = self.up_sample[0, :, :, :].detach().cpu().clone()
                # self.size1 = self.feature_sample.shape[-2]
                # self.size2 = self.feature_sample.shape[-1]
                feature_ = self.up_sample.reshape(1, -1)
                input_sample_ = input_sample.reshape(-1, 1)
                
                return torch.mm(feature_, input_sample_)
            
            from functorch import jacfwd, jacrev, vmap
            
            import time
            time1 = time.time()
            from tqdm.auto import tqdm
            gradients = jacrev(forward_sample, 1)(timesteps, sample).detach()
            # gradients_z1 = gradients.reshape(-1, 1) - sample.reshape(-1, 1)
            # gradients_z1 = gradients_z1.reshape(1280, 48, 48)
            gradients_z = gradients.cpu().clone()
            # print(f'gradients_z.shape: {gradients_z.shape}')
            del gradients
            del gradients_z1
            del self.up_sample
            self.zero_grad()
            if sample.grad is not None:
                sample.grad.zero_()
            if timesteps.grad is not None:
                timesteps.grad.zero_()
            gc.collect()
            torch.cuda.empty_cache()
            # continue
   
            def forward_time(timesteps):
                # sample = self.sample
                # print(sample.shape, self.sample.shape)
                # sample = torch.cat([self.sample[0:1, 1:, :, :], sample], dim=1)
                # timesteps = self.timesteps
                sample = self.sample
                upsample_size = self.upsample_size
                timesteps_ = timesteps
                t_emb1 = self.time_proj(timesteps_)
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
                # self.up_sample = nn.Upsample(size=self.img_size, mode='bilinear')(self.up_sample)
                self.feature_sample = self.up_sample[0, :, :, :].detach().cpu().clone()
                self.size1 = self.feature_sample.shape[-2]
                self.size2 = self.feature_sample.shape[-1]
                self.up_sample = self.up_sample[0]
                
                return self.up_sample
                        
            # import time
            # time1 = time.time()
            gradients = jacfwd(forward_time)(timesteps).detach()
            gradients_t = gradients.cpu().clone()
            del gradients
            
            # print(f'gradients_t.shape {gradients_t.shape}')
 
            
            return self.feature_sample, gradients_t[...,0].cpu(), gradients_z[0, 0].cpu()
    

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
    def grad1_call(
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
        unet_output = self.unet.forward_grad1(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output, latents, noise, latents_noisy, t
    
    @torch.enable_grad()
    def grad2_call(
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
        gradient = self.unet.forward_grad2(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               point=point)
        return gradient
    
    @torch.enable_grad()
    def grad3_call(
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
        points=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_grad3(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               points=points)
        return gradient
    
    @torch.enable_grad()
    def grad4_call(
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
        points=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_grad4(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               points=points)
        return gradient
    
    @torch.enable_grad()
    def grad5_call(
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
        points=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_grad5(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               points=points)
        return gradient
    
    @torch.enable_grad()
    def grad6_call(
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
        points=None,
        repeat_direc=None
    ):

        torch.set_grad_enabled(True)
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        gradient = self.unet.forward_grad6(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs,
                               image_size=image_size,
                               points=points,
                               repeat_direc=repeat_direc)
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
    def forward_grad1(self,
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
        unet_ft_all = self.pipe.grad1_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)

        return unet_ft_all
    
    @torch.enable_grad()
    def forward_grad2(self,
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
        grad = self.pipe.grad2_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            point=point)

        return grad
    
    @torch.enable_grad()
    def forward_grad3(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                points=None):
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
        grad = self.pipe.grad3_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            points=points)

        return grad
    
    
    @torch.enable_grad()
    def forward_grad4(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                points=None):
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
        grad = self.pipe.grad4_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            points=points)

        return grad
    
    @torch.enable_grad()
    def forward_grad5(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                points=None):
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
        grad = self.pipe.grad5_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            points=points)

        return grad
    
    @torch.enable_grad()
    def forward_grad6(self,
                img,
                category=None,
                img_size=[768, 768],
                t=261,
                up_ft_index=1,
                ensemble_size=8,
                image_size=-1,
                points=None,
                repeat_direc = None):
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
        grad = self.pipe.grad6_call(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            image_size=image_size,
            points=points,
            repeat_direc=repeat_direc)

        return grad