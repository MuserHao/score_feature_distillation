import torch
from diffusers import DDIMScheduler
import gc
from torchvision.transforms import PILToTensor
from .dift_sd import MyUNet2DConditionModel, OneStepSDPipeline

class SEQFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', null_prompt='', device='cuda'):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.device = device
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self,
                img_tensor,
                prompt='',
                t=[261],
                up_ft_index=1,
                ensemble_size=8):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t_list: a list of time steps to use, each should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a list of torch tensors, each in the shape of [1, c, h, w]
        '''
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda(self.device) # ensem, c, h, w
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds = self.pipe._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        
        unet_ft_list = []
        for t_val in t:
            unet_ft_all = self.pipe(
                img_tensor=img_tensor,
                t=t_val,
                up_ft_indices=[up_ft_index],
                prompt_embeds=prompt_embeds)
            unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
            unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
            unet_ft_list.append(unet_ft)
        
        return torch.cat(unet_ft_list, dim=1)


class SEQFeaturizer4Eval(SEQFeaturizer):
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', null_prompt='', cat_list=[], device='cuda'):
        super().__init__(sd_id, null_prompt)
        self.device = device
        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                prompt = f"a photo of a {cat}"
                prompt_embeds = self.pipe._encode_prompt(
                    prompt=prompt,
                    device=self.device,
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
                t=[261],
                up_ft_index=1,
                ensemble_size=8):
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda(self.device) # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        
        unet_ft_list = []
        for t_val in t:
            unet_ft_all = self.pipe(
                img_tensor=img_tensor,
                t=t_val,
                up_ft_indices=[up_ft_index],
                prompt_embeds=prompt_embeds)
            unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
            unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
            unet_ft_list.append(unet_ft)
        
        return torch.cat(unet_ft_list, dim=1)
