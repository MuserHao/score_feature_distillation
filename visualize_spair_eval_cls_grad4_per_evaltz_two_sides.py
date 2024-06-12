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

def reshape_f(matrix):
    if len(matrix.shape) == 4 and matrix.shape[0] == 1:
        b, c, h, w = matrix.shape
        matrix = matrix.reshape(c, h, w)
    if len(matrix.shape) == 5 and matrix.shape[0] == 1 and matrix.shape[-1] == 1:
        b, c, h, w, x = matrix.shape
        matrix = matrix.reshape(c, h, w)
    return matrix

def main(args):
    # Initialize the SpairEvaluator
    args.save_path = args.save_path
    args.t = 0
    evaluator = SPairEvaluator(args)
    # cat_list = evaluator.get_cat_list()
    # cat_list.append('overall')
    cat_list = [args.cls_name]
    
    cls_names1 = ['bicycle', 'pottedplant', 'cow', 'cat'] 
    cls_names2 = ['bottle', 'dog', 'bus', 'motorbike', 'person'] 
    cls_names3 = ['train', 'tvmonitor', 'boat', 'car', 'bird', 'sheep', 'horse', 'aeroplane', 'chair']
    cls_names4 = ['person']
    cls_temps = ['dog']
    # cls_names4 = []
    if args.cls_names == "cls_names1":
        run_names = cls_names1
    elif args.cls_names == "cls_names2":
        run_names = cls_names2
    elif args.cls_names == "cls_names3":
        run_names = cls_names3
    elif args.cls_names == "cls_names4":
        run_names = cls_names4
    elif args.cls_names == 'cls_temps':
        run_names = cls_temps
    else:
        raise NotImplementedError
    
    tz_acctimes1 = [[0, -1], [-1, 0],[0, 0]]
    tz_leftweights1 = [[-10, -0.005], [-10, 0.005]]
    tz_rightweights1 = [[-10, -0.005], [-10, 0.005]]
    
    tz_acctimes2 = [[5, 0], [8, 0], [11, 0], [5, 5], [8, 8], [11, 11]]
    tz_leftweights2 = [[-10, 0.005], [-10, 0.005], [10, 0.005], [-10, -0.005], [-10, -0.005], [10, -0.005]]
    tz_rightweights2 = [[-10, 0.005], [10, 0.005], [-10, 0.005], [-10, -0.005], [10, -0.005], [-10, -0.005]]
    
    param_groups = [[tz_acctimes2, tz_leftweights2, tz_rightweights2]]
    
    t_values = range(args.t_range[0], args.t_range[1], args.time_diff)
    
    for cls_name in run_names:
        # args.cls_name = cls_name
        args.save_path = f'./plot_res_tz/plots_{cls_name}_grad4_evaltz_twosides'
        os.makedirs(args.plot_path, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)

        # Define the range of 't' values to evaluate
        feature_folder = f'/media/data2/shilei/score_feature_distillation/plots_{cls_name}_grad4tz'
        
        # t_values = range(args.time_diff, 320, args.time_diff)
        evaluator.set_threshold(args.threshold)
        evaluator.set_timediff(args.time_diff)
        evaluator.set_plot_path(args.plot_path)
        pic_num = 0
        for tz_acctimes, tz_leftweights, tz_rightweights in param_groups:
            tz_groups = [tz_acctimes]
            for tz_groupid, tz_group in enumerate(tz_groups):
                pic_num += 1
                tz_groupid += 10
                per_point_pck_ress = []
                per_image_pck_ress = []
                for t_acctime, z_acctime in tz_group:
                    if t_acctime == -1 and z_acctime == -1:
                        leftweights_loop = [tz_leftweights[0]]
                        rightweights_loop = [tz_rightweights[0]]
                    else:
                        leftweights_loop = tz_leftweights
                        rightweights_loop = tz_rightweights

                    for i, (t_leftweight, z_leftweight) in enumerate(leftweights_loop):
                        t_rightweight, z_rightweight = rightweights_loop[i]
                        per_point_pck_res = []
                        per_image_pck_res = []
                        dicts_image_pck = {}
                        dicts_point_pck = {}
                        total = len(t_values) * len(evaluator.cat2json[cls_name])
                        pbar = tqdm(total=total, desc='steps')
                        for t in t_values:
                            evaluator.set_t(t)
                            total_pck = []
                            all_correct = 0
                            all_total = 0
                            per_point_pck_dict = {}
                            per_image_pck_dict = {}
                            cat = cls_name
                            cat_pck = []
                            cat_correct = 0
                            cat_total = 0
                            
                            dicts_t = {}
                            dicts_z = {}
                            
                            for json_path in evaluator.cat2json[cat]:
                                with open(os.path.join(evaluator.dataset_path, evaluator.test_path, json_path)) as temp_f:
                                    data = json.load(temp_f)
                                    temp_f.close()

                                pbar.update(1)
                                
                                src_imname = data['src_imname']
                                trg_imname = data['trg_imname']
                                
                                src_img_size = data['src_imsize'][:2][::-1]
                                trg_img_size = data['trg_imsize'][:2][::-1]  
                                
                                if src_imname not in dicts_t:
                                    src_grads_t = []
                                    for xx in t_values:
                                        src_gra = torch.load(os.path.join(feature_folder, str(xx), src_imname +'_t_gradient.pth')).cuda()
                                        src_grads_t.append(src_gra)
                                    dicts_t[src_imname] = src_grads_t
                                src_grad_t = dicts_t[src_imname]
                                
                                if trg_imname not in dicts_t:
                                    trg_grads_t = []
                                    for xx in t_values:
                                        trg_gra = torch.load(os.path.join(feature_folder, str(xx), trg_imname +'_t_gradient.pth')).cuda()
                                        trg_grads_t.append(trg_gra)
                                    dicts_t[trg_imname] = trg_grads_t
                                trg_grad_t = dicts_t[trg_imname]
                                
                                if src_imname not in dicts_z:
                                    src_grads_z = []
                                    for xx in t_values:
                                        src_gra = torch.load(os.path.join(feature_folder, str(xx), src_imname +'_z_gradient.pth')).cuda()
                                        src_grads_z.append(src_gra)
                                    dicts_z[src_imname] = src_grads_z
                                src_grad_z = dicts_z[src_imname]
                                
                                if trg_imname not in dicts_z:
                                    trg_grads_z = []
                                    for xx in t_values:
                                        trg_gra = torch.load(os.path.join(feature_folder, str(xx), trg_imname +'_z_gradient.pth')).cuda()
                                        trg_grads_z.append(trg_gra)
                                    dicts_z[trg_imname] = trg_grads_z
                                trg_grad_z = dicts_z[trg_imname]
                                
                                src_fet = torch.load(os.path.join(feature_folder, str(t), src_imname +'_feature.pth')).cuda()
                                src_fet = reshape_f(src_fet)
                                trg_fet = torch.load(os.path.join(feature_folder, str(t), trg_imname +'_feature.pth')).cuda()
                                trg_fet = reshape_f(trg_fet)
                                # print('f norm', torch.norm(src_fet))
                                if z_acctime != -1:
                                    if z_acctime == 0:
                                        src_graz = torch.load(os.path.join(feature_folder, str(t), src_imname +'_z_gradient.pth')).cuda()
                                        trg_graz = torch.load(os.path.join(feature_folder, str(t), trg_imname +'_z_gradient.pth')).cuda()
                                        src_graz = src_graz.reshape(src_fet.shape) - src_fet
                                        trg_graz = trg_graz.reshape(trg_fet.shape) - trg_fet
                                        # print(src_graz.shape, trg_graz.shape, src_fet.shape)
                                        src_fet += z_leftweight * src_graz
                                        trg_fet += z_leftweight * trg_graz
                                        # print(src_graz.shape, trg_graz.shape, src_fet.shape, z_leftweight)
                                    if z_acctime != 0:
                                        tmp1 = src_grad_z[max(t//10-z_acctime, 0):max(t//10-1, 0)]
                                        tmp2 = src_grad_z[t//10:min(t//10+z_acctime, len(src_grads_z))]
                                        src_leftgraz = sum(tmp1)
                                        src_rightgraz = sum(tmp2)
                                        if len(tmp1) == 0:
                                            src_leftgraz = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            src_leftgraz = src_leftgraz.reshape(src_fet.shape) - len(tmp1) * src_fet
                                        if len(tmp2) == 0:
                                            src_rightgraz = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            src_rightgraz = src_rightgraz.reshape(src_fet.shape) - len(tmp2) * src_fet
                                        src_fet += z_leftweight * src_leftgraz + z_rightweight * src_rightgraz
                                        
                                        tmp3 = trg_grad_z[max(t//10-z_acctime, 0):max(t//10-1, 0)]
                                        tmp4 = trg_grad_z[t//10:min(t//10+z_acctime, len(trg_grads_z))]
                                        trg_leftgraz = sum(tmp3)
                                        trg_rightgraz = sum(tmp4)
                                        if len(tmp3) == 0:
                                            trg_leftgraz = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            trg_leftgraz = trg_leftgraz.reshape(src_fet.shape) - len(tmp3) * src_fet
                                        if len(tmp4) == 0:
                                            trg_rightgraz = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            trg_rightgraz = trg_rightgraz.reshape(src_fet.shape) - len(tmp4) * src_fet
                                        trg_fet += z_leftweight * trg_leftgraz + z_rightweight * trg_rightgraz

                                # print(t_acctime, src_fet.shape)
                                if t_acctime != -1:
                                    if t_acctime == 0:
                                        src_grat = torch.load(os.path.join(feature_folder, str(t), src_imname +'_t_gradient.pth')).cuda()
                                        trg_grat = torch.load(os.path.join(feature_folder, str(t), trg_imname +'_t_gradient.pth')).cuda()
                                        src_grat = reshape_f(src_grat)
                                        trg_grat = reshape_f(trg_grat)

                                        src_fet += t_leftweight * src_grat
                                        trg_fet += t_leftweight * trg_grat
                                    if t_acctime != 0:
                                        tmp1 = src_grad_t[max(t//10-t_acctime, 0):max(t//10-1, 0)]
                                        tmp2 = src_grad_t[t//10:min(t//10+t_acctime, len(src_grads_t))]
                                        src_leftgrat = sum(tmp1)
                                        src_rightgrat = sum(tmp2)
                                        if len(tmp1) == 0:
                                            src_leftgrat = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            src_leftgrat = src_leftgrat.reshape(src_fet.shape)
                                        if len(tmp2) == 0:
                                            src_rightgrat = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            src_rightgrat = src_rightgrat.reshape(src_fet.shape)
     
                                        src_fet += t_leftweight * src_leftgrat + t_rightweight * src_rightgrat
                                        
                                        tmp3 = trg_grad_t[max(t//10-t_acctime, 0):max(t//10-1, 0)]
                                        tmp4 = trg_grad_t[t//10:min(t//10+t_acctime, len(trg_grads_t))]
                                        trg_leftgrat = sum(tmp3)
                                        trg_rightgrat = sum(tmp4)
                                        
                                        if len(tmp3) == 0:
                                            trg_leftgrat = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            trg_leftgrat = trg_leftgrat.reshape(src_fet.shape)
                                        if len(tmp4) == 0:
                                            trg_rightgrat = torch.zeros_like(src_fet).to(src_fet.device)
                                        else:
                                            trg_rightgrat = trg_rightgrat.reshape(src_fet.shape)
                                        
                                        trg_fet += t_leftweight * trg_leftgrat + t_rightweight * trg_rightgrat
                                
                                h = trg_img_size[0]
                                w = trg_img_size[1]
                                
                                trg_bndbox = data['trg_bndbox']
                                threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
                                
                                total = 0
                                correct = 0
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
                                
                            per_image_pck_res.append(per_image_pck_dict[cat])
                            per_point_pck_res.append(per_point_pck_dict[cat])
                            dicts_image_pck[t] = per_image_pck_dict[cat]
                            dicts_point_pck[t] = per_point_pck_dict[cat]
                        
                        fol = os.path.join(args.save_path, 'jsons')
                        os.makedirs(fol, exist_ok=True)
                        with open(os.path.join(args.save_path, 'jsons', f'{cat}_Per_Image_evaluation_curve_{t_acctime}_{z_acctime}_{t_leftweight}_{z_leftweight}_{t_rightweight}_{z_rightweight}.json'), 'w') as f:
                            json.dump(dicts_image_pck, f)
                        
                        with open(os.path.join(args.save_path, 'jsons', f'{cat}_Per_Point_evaluation_curve_{t_acctime}_{z_acctime}_{t_leftweight}_{z_leftweight}_{t_rightweight}_{z_rightweight}.json'), 'w') as f:
                            json.dump(dicts_point_pck, f)
                            
                        per_image_pck_ress.append(per_image_pck_res)
                        per_point_pck_ress.append(per_point_pck_res)
                        
                plt.figure()
                group_idx = 0
                for t_acctime1, z_acctime1 in tz_group:
                    if t_acctime1 == -1 and z_acctime1 == -1:
                        leftweights_loop1 = [tz_leftweights[0]]
                        rightweights_loop1 = [tz_rightweights[0]]
                    else:
                        leftweights_loop1 = tz_leftweights
                        rightweights_loop1 = tz_rightweights

                    for i, (t_leftweight1, z_leftweight1) in enumerate(leftweights_loop1):
                        t_rightweight1, z_rightweight1 = rightweights_loop1[i]
                        wsstr = str(t_acctime1) + '_' + str(z_acctime1) + '_' + str(t_leftweight1) + '_' + str(z_leftweight1)+ '_' + str(t_rightweight1) + '_' + str(z_rightweight1)
                        plt.plot(t_values, per_image_pck_ress[group_idx], label=f'{wsstr}')
                        group_idx += 1
                    
                plt.xlabel('t')
                plt.ylabel('PCK')
                plt.title(f'PCK Metric vs. t - Category: {cls_name}')
                plt.legend()
                plt.savefig(os.path.join(args.save_path, f'{cls_name}_Per_Image_evaluation_curve_{pic_num}.png'))
                
                plt.figure()
                group_idx = 0
                for t_acctime1, z_acctime1 in tz_group:
                    if t_acctime1 == -1 and z_acctime1 == -1:
                        leftweights_loop1 = [tz_leftweights[0]]
                        rightweights_loop1 = [tz_rightweights[0]]
                    else:
                        leftweights_loop1 = tz_leftweights
                        rightweights_loop1 = tz_rightweights

                    for i, (t_leftweight1, z_leftweight1) in enumerate(leftweights_loop1):
                        t_rightweight1, z_rightweight1 = rightweights_loop1[i]
                        
                        wsstr = str(t_acctime1) + '_' + str(z_acctime1) + '_' + str(t_leftweight1) + '_' + str(z_leftweight1)+ '_' + str(t_rightweight1) + '_' + str(z_rightweight1)
                        plt.plot(t_values, per_point_pck_ress[group_idx], label=f'{wsstr}')
                        group_idx += 1
                    
                plt.xlabel('t')
                plt.ylabel('PCK')
                plt.title(f'PCK Metric vs. t - Category: {cls_name}')
                plt.legend()
                plt.savefig(os.path.join(args.save_path, f'{cls_name}_Per_Point_evaluation_curve_{pic_num}.png'))
        
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
    parser.add_argument('--cls_names', type=str, default='cls_temps')
    parser.add_argument('--neg_direc', action='store_true')
    args = parser.parse_args()
    main(args)

"""
The evaluation only consider accumulation of gradients in the direction of t+
"""