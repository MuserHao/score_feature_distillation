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
    # print(cat_list)
    # exit(0)
    # cat_list.append('overall')
    cat_list = [args.cls_name]
    # cls_names1 = ['bicycle', 'pottedplant', 'cow', 'cat', 'bottle', 'dog', 'bus', 'motorbike', 'person'] 
    # cls_names2 = ['train', 'tvmonitor', 'boat', 'car', 'bird', 'sheep', 'horse', 'aeroplane', 'chair']
    
    # args.plot_path = f'./plot_res/plots_{args.cls_name}_grad4_eval'
    args.save_path = f'./plot_res/plots_{args.cls_name}_grad4_eval'
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # Define the range of 't' values to evaluate
    feature_folder = f'/media/data2/shilei/score_feature_distillation/plots_{args.cls_name}_grad4'
    t_values = range(args.t_range[0], args.t_range[1], args.time_diff)
    # t_values = range(args.time_diff, 320, args.time_diff)

    evaluator.set_threshold(args.threshold)
    evaluator.set_timediff(args.time_diff)
    evaluator.set_plot_path(args.plot_path)
    t_values = range(0, 990, args.time_diff)
    # t_values = range(0, 500, args.time_diff)

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
    # ws1 = [[0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0], [0, 0, 0, 10, 0, 0], [0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 0, 10]]
    # ws2 = [[0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0], [10, 10, 0, 0, 0, 0], [10, 10, 10, 0, 0, 0], [10, 10, 10, 10, 0, 0], [10, 10, 10, 10, 10, 0], [10, 10, 10, 10, 10, 10]]
    # ws1 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0]]
    # ws2 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 10, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0], [1, 10, 0, 0, 10, 0, 0, 10, 0, 0, 0, 0], [1, 10, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0]]
    # ws3 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
    # wss = [ws1, ws2, ws3]
    ws0 = [[-1], [0]]
    ws1 = [[0], [6], [11], [16], [21]]
    ws2 = [[0], [26], [31], [36], [41]]
    ws3 = [[0], [46], [51], [56], [61]]
    ws4 = [[0], [6], [11], [16], [21], [26]]
    ws4 = [[0], [11], [16], [21]]
    # wss = [ws0, ws1, ws2, ws3]
    wss = [ws4]
    # wss = [ws0, ws1, ws2, ws3]
    for ws_idxx, ws in enumerate(wss):
        ws_idxx += 3
        per_point_pck_ress = []
        per_image_pck_ress = []
        for w_idx in ws:
            if isinstance(w_idx, list):
                w_idx = w_idx[0]
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
                for cat in [args.cls_name]:
                    cat_pck = []
                    cat_correct = 0
                    cat_total = 0
                    
                    dicts = {}
                    
                    for json_path in tqdm(evaluator.cat2json[cat]):
                        with open(os.path.join(evaluator.dataset_path, evaluator.test_path, json_path)) as temp_f:
                            data = json.load(temp_f)
                            temp_f.close()
                            
                        src_imname = data['src_imname']
                        trg_imname = data['trg_imname']
                        
                        src_img_size = data['src_imsize'][:2][::-1]
                        trg_img_size = data['trg_imsize'][:2][::-1]  
                        
                        if src_imname not in dicts:
                            src_grads = []
                            for xx in t_values:
                                src_gra = torch.load(os.path.join(feature_folder, str(xx), src_imname +'_gradient.pth')).cuda()
                                src_grads.append(src_gra)
                            dicts[src_imname] = src_grads
                        src_grads = dicts[src_imname]
                        
                        if trg_imname not in dicts:
                            trg_grads = []
                            for xx in t_values:
                                trg_gra = torch.load(os.path.join(feature_folder, str(xx), trg_imname +'_gradient.pth')).cuda()
                                trg_grads.append(trg_gra)
                            dicts[trg_imname] = trg_grads
                        trg_grads = dicts[trg_imname]
                        
                        src_fet = torch.load(os.path.join(feature_folder, str(t), src_imname +'_feature.pth')).cuda()
                        trg_fet = torch.load(os.path.join(feature_folder, str(t), trg_imname +'_feature.pth')).cuda()
                        
                        if w_idx == -1:
                            if args.neg_vec:
                                vv = 10*sum(src_grads[:t//10])
                            else:
                                vv = 10*sum(src_grads[t//10:])
                            if args.neg_direc:
                                src_fet -= vv
                            else:
                                src_fet += vv
                        elif w_idx != 0:
                            if args.neg_vec:
                                vv = 10*sum(src_grads[max(t//10-w_idx, 0): t//10])
                            else:
                                vv = 10*sum(src_grads[t//10:t//10+w_idx])
                            
                            if args.neg_direc:
                                src_fet -= vv    
                            else:
                                src_fet += vv

                        if w_idx == -1:
                            if args.neg_vec:
                                vv = 10*sum(trg_grads[:t//10])
                            else:
                                vv = 10*sum(trg_grads[t//10:])
                            if args.neg_direc:
                                trg_fet -= vv
                            else:
                                trg_fet += vv
                        elif w_idx != 0:
                            if args.neg_vec:
                                yy = 10*sum(trg_grads[max(t//10-w_idx, 0):t//10])
                            else:
                                yy = 10*sum(trg_grads[t//10:t//10+w_idx])
                            
                            if args.neg_direc:
                                trg_fet -= yy
                            else:
                                trg_fet += yy
                        
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
                            
                            resized_src_fet = resize(src_fet, src_img_size)
                                
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
                    
                per_image_pck_res.append(per_image_pck_dict[args.cls_name])
                per_point_pck_res.append(per_point_pck_dict[args.cls_name])
            
            per_image_pck_ress.append(per_image_pck_res)
            per_point_pck_ress.append(per_point_pck_res)
        plt.figure()
        for res, ww in zip(per_image_pck_ress, ws):
            if isinstance(ww, list):
                ww = ww[0]
            wsstr = str(ww)
            plt.plot(t_values, res, label=f'{wsstr}')
        plt.xlabel('t')
        plt.ylabel('PCK')
        plt.title(f'PCK Metric vs. t - Category: {args.cls_name}')
        plt.legend()
        if args.neg_direc and args.neg_vec:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Image_evaluation_curve_restall_{ws_idxx}_negrec_negvec.png'))    
        elif args.neg_vec:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Image_evaluation_curve_restall_{ws_idxx}_negvec.png'))    
        elif args.neg_direc:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Image_evaluation_curve_restall_{ws_idxx}_negrec.png'))    
        else:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Image_evaluation_curve_restall_{ws_idxx}.png'))
        
        plt.figure()
        # plt.plot(t_values, per_point_pck_res, label='Per Point PCK@0.1')
        for res, ww in zip(per_point_pck_ress, ws):
            if isinstance(ww, list):
                ww = ww[0]
            wsstr = str(ww)
            plt.plot(t_values, res, label=f'{wsstr}')
        plt.xlabel('t')
        plt.ylabel('PCK')
        plt.title(f'PCK Metric vs. t - Category: {args.cls_name}')
        plt.legend()
        if args.neg_direc and args.neg_vec:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Point_evaluation_curve_restall_{ws_idxx}_negrec_negvec.png'))    
        elif args.neg_vec:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Point_evaluation_curve_restall_{ws_idxx}_negvec.png'))    
        elif args.neg_direc:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Point_evaluation_curve_restall_{ws_idxx}_negrec.png'))    
        else:
            plt.savefig(os.path.join(args.save_path, f'{args.cls_name}_Per_Point_evaluation_curve_restall_{ws_idxx}.png'))
    
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
    parser.add_argument('--cls_name', type=str, default='person')
    parser.add_argument('--neg_direc', action='store_true')
    parser.add_argument('--neg_vec', action='store_true')
    args = parser.parse_args()
    main(args)

"""
The feature used for computing performance is the gradient of the feature map which is obtained by loss
"""