# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open('t_z_g_g.jpg')

# inputs = processor(text=["A dog is running."], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# print(logits_per_image)
# probs = logits_per_image.softmax(dim=1)  # w
# # print(probs)


# # fg = 28.152
# # tgg = 28.758
# # zgg = 29.2668
# # tzgwg = 31.0606
# # tzgg = 31.1687
import numpy as np
import torch
def calculate_clip_score(images, prompts):
    # import pdb;pdb.set_trace()
    # images_int = (np.asarray(images[0]) * 255).astype("uint8")
    images_int = (np.asarray(images) * 255).astype("uint8")
    images_int = images_int[None]
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

image = Image.open('temp48.jpg')
prompt = 'fat rabbits with oranges, photograph iphone, trending'
sd_clip_score = calculate_clip_score(image, prompt)
print(f"CLIP score: {sd_clip_score}")

# tzgg 29.5073
# tzgwg 29.3698
# tgg 26.7722
# zgg 26.573
# fg 26.0413

# fg: 27.0594
# 200_1_0.005: 25.8098 
# 200_1_0.002: 26.644
# -200_1_0.005: 28.6801
# 200_1_0.01: 26.1989
# 200_1_0.02  25.7691

# < 200 - t - z  else + t + z 28
# < 200 + t - z else + t + z 27.5822
# < 200 - t - z else - t + z 

# without classifier free guidance
# standard:  22.577
# < 200 + t + z  else - t - z    1 0.005  25.8919
# < 200 + t + z  else - t - z    20 0.03  25.393
# < 200 - t - z  else + t + z    1 0.005  18.2443
# < 200 + t + z  else - t - z    10 0.01  24.1707

# with classifier free guidance
# standard:  27.0594
# only cond < 200 + t + z  else - t - z  1 0.005 26.0941
# only cond < 200 - t - z  else + t + z  1 0.005 24.8429
# only uncond < 200 - t - z  else + t + z  1 0.005 24.6725
# only uncond < 200 + t + z  else - t - z  1 0.005 24.8762
# uncond + cond < 200 + t + z  else - t - z  1 0.005 25.9565
# uncond + cond < 200 - t - z  else + t + z  1 0.005 28.4049
# uncond + cond < 200 - t - z  1 0.005  26.2671 
# uncond + cond < 200 + t + z  1 0.005  26.5049