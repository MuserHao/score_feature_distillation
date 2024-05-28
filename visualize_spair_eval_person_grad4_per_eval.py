import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_spair import SPairEvaluator
import os
import json
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch import nn

def resize(feat_in, size, limit_size=48):
    if len(feat_in.shape) == 3:
        feat_in = feat_in[None, ...]
    if feat_in.shape[-1] == limit_size:
        feature_out = nn.Upsample(size=size, mode='bilinear')(feat_in)
    else:
        feature_out = feat_in
    if len(feature_out.shape) == 4:
        feature_out = feature_out[0]
    return feature_out

def main(args):
    # Initialize the SpairEvaluator
    args.save_path = args.save_path
    args.t = 0
    evaluator = SPairEvaluator(args)
    # cat_list = evaluator.get_cat_list()
    # cat_list.append('overall')
    cat_list = ['person']
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # Define the range of 't' values to evaluate
    feature_folder = './plot_res/plots_person_grad4_48x48'
    args.plot_path = feature_folder
    t_values = range(args.time_diff, args.t_range[1], args.time_diff)
    # t_values = range(args.time_diff, 320, args.time_diff)

    evaluator.set_threshold(args.threshold)
    evaluator.set_timediff(args.time_diff)
    evaluator.set_plot_path(args.plot_path)
    t_values = range(0, 990, args.time_diff)

    res1 = {}
    res2 = {}
    # ws = [[0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    # ws = [[-1, 0, 0], [0, -1, -1], [0, -1, 0], [0, 0, -1]]
    # ws = [[0.5, 0.25, 0.25], [-0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, 0.5], [0.25, 0.25, -0.5]]
    # ws = [[10, 1, 1], [1, 10, 1], [1, 1, 10], [-10, 1, 1],[1, -10, 1], [1, 1, -10]]
    # ws = [[0, 0, 0], [0, -10, -10], [0, -10, 0], [0, 0, -10], [-10, 0, 0], [-10, -10, 0], [-10, 0, -10]]
    # ws = [[0, 0, 0], [0, -100, -100], [0, -100, 0], [0, 0, -100], [-100, 0, 0], [-100, -100, 0], [-100, 0, -100]]
    # ws = [[0, 0, 0, 0], [0, 0, 10, 10]]
    # 1280x48x48 -> 4x64x64    4x64x64x1 = 1280x48x48
    
    # ws = [[0, 0, 0, 0], [0, -10, 10, 10], [-10, 0, 10, 10], [-10, -10, 0, 10], [-10, -10, 10, 0]]
    # ws = [[0, 0, 0, 0], [10, 10, -10, -10], [-10, -10, 10, 10]]
    # ws = [[0, 0, 0, 0], [-10, 0, 0, 0], [0, -10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]
    # ws = [[0, 0, 0, 0], [-20, 0, 0, 0], [0, -20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20]]
    ws = [[0, 0, 0, 0], [-50, 0, 0, 0], [0, -50, 0, 0], [0, 0, 50, 0], [0, 0, 0, 50]]
    # ws = [[0, 0, 0, 0]]
    per_point_pck_ress = []
    per_image_pck_ress = []
    for w1, w2, w3, w4 in ws:
        values1 = []
        values2 = []
        per_point_pck_res = []
        per_image_pck_res = []
        for t in t_values:
            evaluator.set_t(t)
            total_pck = []
            all_correct = 0
            all_total = 0
            per_point_pck_dict = {}
            per_image_pck_dict = {}
            for cat in ['person']:
                cat_pck = []
                cat_correct = 0
                cat_total = 0
                for json_path in tqdm(evaluator.cat2json[cat]):
                    with open(os.path.join(evaluator.dataset_path, evaluator.test_path, json_path)) as temp_f:
                        data = json.load(temp_f)
                        temp_f.close()
                        
                    src_imname = data['src_imname']
                    trg_imname = data['trg_imname']
                    
                    src_img_size = data['src_imsize'][:2][::-1]
                    trg_img_size = data['trg_imsize'][:2][::-1]  

                    src_fet = torch.load(os.path.join(args.plot_path, str(t), src_imname +'_feature.pth')).cuda()
                    # src_fet = resize(src_fet, src_img_size)
                    src_gra = torch.load(os.path.join(args.plot_path, str(t), src_imname +'_gradient.pth')).cuda()
                    # print(f'src gradient: {torch.norm(src_fet)}, {torch.norm(src_gra)}')
                    # src_gra = resize(src_gra, src_img_size)
                    src_gra_t_10 = torch.load(os.path.join(args.plot_path, str(max(t-10, min(t_values))), src_imname +'_gradient.pth')).cuda()
                    # src_gra_t_10 = resize(src_gra_t_10, src_img_size)
                    src_gra_t__10 = torch.load(os.path.join(args.plot_path, str(min(t+10, max(t_values))), src_imname +'_gradient.pth')).cuda()
                    # src_gra_t__10 = resize(src_gra_t__10, src_img_size)
                    src_gra_t__20 = torch.load(os.path.join(args.plot_path, str(min(t+20, max(t_values))), src_imname +'_gradient.pth')).cuda()
                    # src_gra_t__20 = resize(src_gra_t__20, src_img_size)
                    
                    with open(os.path.join(os.path.join(args.plot_path, str(t), src_imname +'_bbox.json')), 'r') as f:
                        src_boxes = json.load(f)
                    # assert src_fet.shape[1] == src_gra.shape[1]
                    # assert src_fet.shape[1] == len(src_boxes)-1
                    # assert src_img_size == src_boxes[0]
                    
                    trg_fet = torch.load(os.path.join(args.plot_path, str(t), trg_imname +'_feature.pth')).cuda()
                    # trg_fet = resize(trg_fet, trg_img_size)
                    trg_gra = torch.load(os.path.join(args.plot_path, str(t), trg_imname +'_gradient.pth')).cuda()
                    # trg_gra = resize(trg_gra, trg_img_size)
                    trg_gra_t_10 = torch.load(os.path.join(args.plot_path, str(max(t-10, min(t_values))), trg_imname +'_gradient.pth')).cuda()
                    # trg_gra_t_10 = resize(trg_gra_t_10, trg_img_size)
                    trg_gra_t__10 = torch.load(os.path.join(args.plot_path, str(min(t+10, max(t_values))), trg_imname +'_gradient.pth')).cuda()
                    # trg_gra_t__10 = resize(trg_gra_t__10, trg_img_size)
                    trg_gra_t__20 = torch.load(os.path.join(args.plot_path, str(min(t+20, max(t_values))), trg_imname +'_gradient.pth')).cuda()
                    # trg_gra_t__20 = resize(trg_gra_t__20, trg_img_size)
                    
                    with open(os.path.join(os.path.join(args.plot_path, str(t), trg_imname +'_bbox.json')), 'r') as f:
                        trg_boxes = json.load(f)
                    # assert trg_fet.shape[1] == trg_gra.shape[1]
                    # assert trg_fet.shape[1] == len(trg_boxes)-1
                    # assert trg_img_size == trg_boxes[0]
                    
                    h = trg_img_size[0]
                    w = trg_img_size[1]
                    
                    trg_bndbox = data['trg_bndbox']
                    threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
                    
                    total = 0
                    correct = 0
                    resized = False
                    for idx in range(len(data['src_kps'])):
                        total += 1
                        cat_total += 1
                        all_total += 1
                        src_point = data['src_kps'][idx]
                        trg_point = data['trg_kps'][idx]
                        
                        src_fet = src_fet - w1*src_gra - w2*src_gra_t_10 + w3*src_gra_t__10 + w4*src_gra_t__20
                        resized_src_fet = resize(src_fet, src_img_size)
                            
                        trg_fet = trg_fet - w1*trg_gra - w2*trg_gra_t_10 + w3*trg_gra_t__10 + w4*trg_gra_t__20
                        resized_trg_fet = resize(trg_fet, trg_img_size)

                        num_channel = src_fet.shape[0]
                        src_vec = resized_src_fet[:, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                        trg_vec = resized_trg_fet.view(num_channel, -1).transpose(0, 1) # HW, C
                        src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                        trg_vec = F.normalize(trg_vec) # HW, c
                        cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                        dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                        if (dist / threshold) <= args.threshold:
                            correct += 1
                            cat_correct += 1
                            all_correct += 1
                    cat_pck.append(correct / total)
                total_pck.extend(cat_pck)

                per_point_pck_dict[cat] = cat_correct / cat_total * 100 # average pck
                per_image_pck_dict[cat] = np.mean(cat_pck) * 100 # average pck per image
                
            per_image_pck_res.append(per_image_pck_dict['person'])
            per_point_pck_res.append(per_point_pck_dict['person'])
        
        per_image_pck_ress.append(per_image_pck_res)
        per_point_pck_ress.append(per_point_pck_res)
    plt.figure()
    for res, ww in zip(per_image_pck_ress, ws):
        ww1, ww2, ww3, ww4 = ww
        wsstr = str(ww1) + '_' + str(ww2) + '_' + str(ww3) + '_' + str(ww4)
        plt.plot(t_values, res, label=f'{wsstr}')
    plt.xlabel('t')
    plt.ylabel('PCK')
    plt.title(f'PCK Metric vs. t - Category: Person')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, f'Person_Per_Image_evaluation_curve_t10t20_6.png'))
    
    plt.figure()
    # plt.plot(t_values, per_point_pck_res, label='Per Point PCK@0.1')
    for res, ww in zip(per_point_pck_ress, ws):
        ww1, ww2, ww3, ww4 = ww
        wsstr = str(ww1) + '_' + str(ww2) + '_' + str(ww3) + '_' + str(ww4)
        plt.plot(t_values, res, label=f'{wsstr}')
    plt.xlabel('t')
    plt.ylabel('PCK')
    plt.title(f'PCK Metric vs. t - Category: Person')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, f'Person_Per_Point_evaluation_curve_t10t20_6.png'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='./plot_res/plots_person_grad4_eval', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t_range', nargs='+', type=int, default=[0, 999], help='range of t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--time_diff', type=int, default=10)
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--plot_path', type=str, default='./plot_res/plots_person_grad4', help='path to save plots')
    args = parser.parse_args()
    main(args)

"""
The feature used for computing performance is the gradient of the feature map which is obtained by loss
"""