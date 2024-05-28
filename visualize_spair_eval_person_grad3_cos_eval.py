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

    t_values = range(args.time_diff, args.t_range[1], args.time_diff)
    # t_values = range(args.time_diff, 320, args.time_diff)

    # Initialize dict of lists to store evaluation results
    per_image_pck = {}
    per_point_pck = {}
    for cat in cat_list:
        per_image_pck[cat] = []
        per_point_pck[cat] = []

    evaluator.set_threshold(args.threshold)
    evaluator.set_timediff(args.time_diff)
    evaluator.set_plot_path(args.plot_path)
    t_values = range(20, 400, args.time_diff)

    res1 = {}
    res2 = {}
    # ws = [[0, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    # ws = [[-1, 0, 0], [0, -1, -1], [0, -1, 0], [0, 0, -1]]
    # ws = [[0.5, 0.25, 0.25], [-0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, 0.5], [0.25, 0.25, -0.5]]
    ws = [[10, 1, 1], [1, 10, 1], [1, 1, 10], [-10, 1, 1],[1, -10, 1], [1, 1, -10]]
    for w1, w2, w3 in ws:
        values1 = []
        values2 = []
        for t in t_values:
            evaluator.set_t(t)
            
            good = 0
            all = 0
            neg_good = 0
            neg_all = 0
            for cat in ['person']:
                
                for json_path in evaluator.cat2json[cat]:
                    with open(os.path.join(evaluator.dataset_path, evaluator.test_path, json_path)) as temp_f:
                        data = json.load(temp_f)
                        temp_f.close()
                        
                    src_imname = data['src_imname']
                    trg_imname = data['trg_imname']
                    
                    src_img_size = data['src_imsize'][:2][::-1]
                    trg_img_size = data['trg_imsize'][:2][::-1]  

                    src_fet = torch.load(os.path.join(args.plot_path, str(t), src_imname +'_feature.pth'))
                    src_gra = torch.load(os.path.join(args.plot_path, str(t), src_imname +'_gradient.pth'))
                    src_gra_t_10 = torch.load(os.path.join(args.plot_path, str(t-10), src_imname +'_gradient.pth'))
                    src_gra_t__10 = torch.load(os.path.join(args.plot_path, str(t+10), src_imname +'_gradient.pth'))
                    
                    with open(os.path.join(os.path.join(args.plot_path, str(t), src_imname +'_bbox.json')), 'r') as f:
                        src_boxes = json.load(f)
                    assert src_fet.shape[1] == src_gra.shape[1]
                    assert src_fet.shape[1] == len(src_boxes)-1
                    assert src_img_size == src_boxes[0]
                    
                    trg_fet = torch.load(os.path.join(args.plot_path, str(t), trg_imname +'_feature.pth'))
                    trg_gra = torch.load(os.path.join(args.plot_path, str(t), trg_imname +'_gradient.pth'))
                    trg_gra_t_10 = torch.load(os.path.join(args.plot_path, str(t-10), trg_imname +'_gradient.pth'))
                    trg_gra_t__10 = torch.load(os.path.join(args.plot_path, str(t+10), trg_imname +'_gradient.pth'))
                    with open(os.path.join(os.path.join(args.plot_path, str(t), trg_imname +'_bbox.json')), 'r') as f:
                        trg_boxes = json.load(f)
                    assert trg_fet.shape[1] == trg_gra.shape[1]
                    assert trg_fet.shape[1] == len(trg_boxes)-1
                    assert trg_img_size == trg_boxes[0]
                    
                    for idx in range(len(data['src_kps'])):
                        src_point = data['src_kps'][idx]
                        trg_point = data['trg_kps'][idx]
                        
                        src_idx = src_boxes.index(src_point)
                        trg_idx = trg_boxes.index(trg_point)
                        for i in range(len(trg_boxes)):
                            neg_trg_idx = random.randint(0, len(trg_boxes)-2)
                            if neg_trg_idx != trg_idx:
                                break
                        
                        src_feat_idx = src_fet[:, src_idx].view(1, 1280)
                        src_feat_norm = F.normalize(src_feat_idx, 1)
                        trg_feat_idx = trg_fet[:, trg_idx].view(1280, 1)
                        trg_feat_norm = F.normalize(trg_feat_idx, 0)
                        neg_trg_feat_idx = trg_fet[:, neg_trg_idx].view(1280, 1)
                        neg_trg_feat_norm = F.normalize(neg_trg_feat_idx, 0)
                        
                        src_grad_idx = src_feat_idx - w1*src_gra[:, src_idx].view(1, 1280) - w2*src_gra_t_10[:, src_idx].view(1, 1280) + w3*src_gra_t__10[:, src_idx].view(1, 1280)
                        src_grad_norm = F.normalize(src_grad_idx, 1)
                        trg_grad_idx = trg_feat_idx - w1*trg_gra[:, trg_idx].view(1280, 1) - w2*trg_gra_t_10[:, trg_idx].view(1280, 1) + w3*trg_gra_t__10[:, trg_idx].view(1280, 1)
                        trg_grad_norm = F.normalize(trg_grad_idx, 0)
                        neg_trg_grad_idx = neg_trg_feat_idx - w1*trg_gra[:, neg_trg_idx].view(1280, 1) - w2*trg_gra_t_10[:, neg_trg_idx].view(1280, 1) + w3*trg_gra_t__10[:, neg_trg_idx].view(1280, 1)
                        neg_trg_grad_norm = F.normalize(neg_trg_grad_idx, 0)
                        
                        cos_feat = torch.mm(src_feat_norm, trg_feat_norm).cpu().numpy()[0, 0]
                        cos_grad = torch.mm(src_grad_norm, trg_grad_norm).cpu().numpy()[0, 0]
                        if cos_grad > cos_feat:
                            good += 1
                        all += 1
                        
                        neg_cos_feat = torch.mm(src_feat_norm, neg_trg_feat_norm).cpu().numpy()[0, 0]
                        neg_cos_grad = torch.mm(src_grad_norm, neg_trg_grad_norm).cpu().numpy()[0, 0]
                        if neg_cos_grad < neg_cos_feat:
                            neg_good += 1
                        neg_all += 1
            values1.append(good / all)
            values2.append(neg_good / neg_all)
            print(t, good / all, neg_good / neg_all)
        key = str(w1) + '_' + str(w2) + '_' + str(w3)
        res1[key] = values1
        res2[key] = values2
    
        plt.figure()
        for k, v in res1.items():
            plt.plot(list(t_values), v, label=k)
        plt.xlabel('t')
        plt.ylabel('cos sim inc')
        plt.title(f'cos sim inc vs. t - Category: Person')
        plt.legend()
        plt.savefig(os.path.join(args.save_path, f'person_evaluation_curve_cos_sim_inc.png'))
        
        plt.figure()
        for k, v in res2.items():
            plt.plot(list(t_values), v, label=k)
        plt.xlabel('t')
        plt.ylabel('neg cos sim inc')
        plt.title(f'neg cos sim inc vs. t - Category: Person')
        plt.legend()
        plt.savefig(os.path.join(args.save_path, f'person_evaluation_curve_neg_cos_sim_inc.png'))
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='./plot_res/plots_person_grad3_eval', help='path to save features')
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
    parser.add_argument('--plot_path', type=str, default='./plot_res/plots_person_grad3', help='path to save plots')
    args = parser.parse_args()
    main(args)

"""
The feature used for computing performance is the gradient of the feature map which is obtained by loss
"""