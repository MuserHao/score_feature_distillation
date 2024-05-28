# import os

# dir1 = 'plots'
# dir2 = 'plots_person_diff2'

# subdirs = os.listdir(dir1)
# subdirs = [ dirr for dirr in subdirs if os.path.isdir(os.path.join(dir1, dirr))]
# subdirs = [dirr for dirr in subdirs if dirr != 'overall']
# # file_label = 'Per_Image'
# # folder = 'temp_images'
# file_label = 'Per_Point'
# folder = 'temp_points'


# for subdir in subdirs:
#     dir_path = os.path.join(dir1, subdir)
#     res = []
#     xxs = []
#     for file in os.listdir(dir_path):
#         if file_label in file:
#             with open(os.path.join(dir_path, file), 'r') as f:
#                 data = f.readlines()
#                 for dd in data:
#                     xx = dd.strip().split('\t')
#                     xx = [float(x) for x in xx]
#                     xxs.append(xx)
#     xxs.sort(key = lambda x: x[1])
#     res.append([subdir] + xxs[-1])
#     yys = []
#     for file in os.listdir(os.path.join(dir2, subdir)):
#         if file_label in file:
#              with open(os.path.join(dir2, subdir, file), 'r') as f:
#                 data = f.readlines()
#                 tts = []
#                 for dd in data:
#                     xx = dd.strip().split('\t')
#                     xx = [float(x) for x in xx]
#                     tts.append(xx)
#                 tts.sort(key = lambda x: x[1])
#                 yys.append([file.split('0.1_0.1_')[-1]] + tts[-1])
#     yys.sort(key = lambda x: x[2])
#     yys = yys[::-1]
    
#     with open(f'{folder}/results_{subdir}.txt', 'w') as f:
#         strings = subdir + '\n'
#         f.write(strings)
#         strings = str(xxs[-1][0]) + '\t' + str(xxs[-1][1]) + '\n'
#         f.write(strings)
#         for yy in yys:
#             strings = str(yy[1]) + '\t' + str(yy[2]) + '\t' + yy[0] + '\n'
#             f.write(strings)
            
            
import torch
import math

def inverse_timestep_embedding(
    embeddings: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    Compute the inverse of the get_timestep_embedding function.

    :param embeddings: an [N x dim] Tensor of positional embeddings.
    :param embedding_dim: the dimension of the input embeddings.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: a 1-D Tensor of N timesteps.
    """
    assert len(embeddings.shape) == 2 and embeddings.shape[1] == embedding_dim, "Embeddings should be a 2d-array of shape [N x dim]"

    half_dim = embedding_dim // 2

    if flip_sin_to_cos:
        embeddings = torch.cat([embeddings[:, half_dim:], embeddings[:, :half_dim]], dim=-1)

    # Separate sine and cosine components
    sine_emb = embeddings[:, :half_dim]
    cosine_emb = embeddings[:, half_dim:]

    # Recover the scaled timesteps using arctan2 to handle sine and cosine properly
    scaled_timesteps = torch.atan2(sine_emb, cosine_emb) / scale

    # Compute the exponent
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=embeddings.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    # Compute the frequency
    frequencies = torch.exp(exponent)

    # Recover the original timesteps
    timesteps = scaled_timesteps / frequencies

    # Assuming timesteps are aligned and averaged across dimensions
    timesteps = timesteps.mean(dim=-1)
    return round(timesteps.item())
    # return timesteps

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

# xx = 0
# from tqdm.auto import tqdm
# for i in tqdm(range(1000)):
#     t = torch.Tensor([i])    
#     t_emb = get_timestep_embedding(t, embedding_dim=320, flip_sin_to_cos=True)
#     t_r = inverse_timestep_embedding(t_emb, embedding_dim=320,
#         flip_sin_to_cos=True)
#     print(t.item(), t_r)
#     res = t.item() == t_r
#     if res == True:
#         xx += 1
# print(xx/1000)
# print(t_r)
import numpy as np
import os
import json
from torch.nn import functional as F

npy_dir = '/home/shilei/projects/score_feature_distillation/plots_person_grad2/person_npys'

folder = 'SPair-71k/PairAnnotation/test' 
files = os.listdir(folder)
files = [file for file in files if 'person.json' in file]
xx = 0
yy = 0
zz = 0
names = []
# for file in files:
#     json_path = os.path.join(folder, file)
#     with open(json_path) as temp_f:
#         data = json.load(temp_f)
#         temp_f.close()
    
#     for idx in range(len(data['src_kps'])):
#         src_point = data['src_kps'][idx]
#         trg_point = data['trg_kps'][idx]
        
#         src_feature_path = 'src_' + data['src_imname'] + '_' + str(src_point[1]) + '_' + str(src_point[0]) + '_feature.npy'
#         src_gradient_path = 'src_' + data['src_imname'] + '_' + str(src_point[1]) + '_' + str(src_point[0]) + '_gradient.npy'
        
#         trg_feature_path = 'trg_' + data['trg_imname'] + '_' + str(trg_point[1]) + '_' + str(trg_point[0]) + '_feature.npy'
#         trg_gradient_path = 'trg_' + data['trg_imname'] + '_' + str(trg_point[1]) + '_' + str(trg_point[0]) + '_gradient.npy'
#         names.append(data['src_imname'])
#         names.append(data['trg_imname'])
        
#         zz += 2

# print(len(names))
# print(len(set(names)))
# print(zz)
# print(xx, yy)
# print(yy/xx)


import torch

x = torch.zeros((1, 3, 44, 44))
ll = [[1, 2], [3, 4], [4, 5], [6, 7], [7, 8]]
print(np.array(ll).shape)
ll = torch.from_numpy(np.array(ll))
y = x[0, :, ll[:, 0], ll[:, 1]]
print(y.shape)