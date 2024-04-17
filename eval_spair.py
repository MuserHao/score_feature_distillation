import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.models.dift_sd import SDFeaturizer4Eval
from src.models.dift_adm import ADMFeaturizer4Eval
from src.models.dift_high_order import SEQFeaturizer4Eval
import os
import json
from PIL import Image
import torch.nn as nn

class SPairEvaluator:
    def __init__(self, args):
        self.args = args
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                print('%s: %s' % (str(arg), str(value)))

        self.dataset_path = args.dataset_path
        self.test_path = 'PairAnnotation/test'
        self.json_list = os.listdir(os.path.join(self.dataset_path, self.test_path))
        if len(args.all_cats) > 0:
            self.all_cats = args.cat_list
        self.all_cats = os.listdir(os.path.join(self.dataset_path, 'JPEGImages'))
        self.save_path = args.save_path
        self.img_size = args.img_size
        self.up_ft_index = args.up_ft_index
        self.ensemble_size = args.ensemble_size
        self.t = args.t[0]
        if args.dift_model == 'sd':
            self.dift = SDFeaturizer4Eval(cat_list=self.all_cats)
        elif args.dift_model == 'adm':
            self.dift = ADMFeaturizer4Eval()
        elif args.dift_model == 'sdh':
            self.dift = SEQFeaturizer4Eval()
            self.t = args.t

        self.cat2json = {}
        self.cat2img = {}
        # prepare json & image lists
        self.get_data_lists()

    def set_t(self, t):
        self.t = t

    def get_cat_list(self):
        return self.all_cats.copy()

    def get_data_lists(self):
        for cat in self.all_cats:
            self.cat2json[cat] = []
            self.cat2img[cat] = []
            for json_path in self.json_list:
                if cat in json_path:
                    self.cat2json[cat].append(json_path)
                    with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                        data = json.load(temp_f)
                        temp_f.close()
                    src_imname = data['src_imname']
                    trg_imname = data['trg_imname']
                    if src_imname not in self.cat2img[cat]:
                        self.cat2img[cat].append(src_imname)
                    if trg_imname not in self.cat2img[cat]:
                        self.cat2img[cat].append(trg_imname)

    def infer_and_save_features(self):
        print("saving all test images' features...")
        os.makedirs(self.save_path, exist_ok=True)
        for cat in tqdm(self.all_cats):
            output_dict = {}
            for image_path in self.cat2img[cat]:
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                output_dict[image_path] = self.dift.forward(img,
                                                    category=cat,
                                                    img_size=self.img_size,
                                                    t=self.t,
                                                    up_ft_index=self.up_ft_index,
                                                    ensemble_size=self.ensemble_size)
            torch.save(output_dict, os.path.join(self.save_path, f'{cat}.pth'))
            
    def evaluate(self, vocal=False):
        torch.cuda.set_device(0)
        self.infer_and_save_features()
        
        total_pck = []
        all_correct = 0
        all_total = 0
        per_point_pck_dict = {}
        per_image_pck_dict = {}

        for cat in tqdm(self.all_cats):
            output_dict = torch.load(os.path.join(self.args.save_path, f'{cat}.pth'))

            cat_pck = []
            cat_correct = 0
            cat_total = 0

            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()

                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]

                src_ft = output_dict[data['src_imname']]
                trg_ft = output_dict[data['trg_imname']]

                src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
                trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
                h = trg_ft.shape[-2]
                w = trg_ft.shape[-1]

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

                    num_channel = src_ft.size(1)
                    src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                    trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                    src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                    trg_vec = F.normalize(trg_vec) # HW, c
                    cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                    max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                    dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                    if (dist / threshold) <= 0.1:
                        correct += 1
                        cat_correct += 1
                        all_correct += 1

                cat_pck.append(correct / total)
            total_pck.extend(cat_pck)

            per_point_pck_dict[cat] = cat_correct / cat_total * 100 # average pck
            per_image_pck_dict[cat] = np.mean(cat_pck) * 100 # average pck per image
            if vocal:
                print(f'{cat} per image PCK@0.1: {per_image_pck_dict[cat]:.2f}')
                print(f'{cat} per point PCK@0.1: {per_point_pck_dict[cat]:.2f}')

        per_point_pck_dict['overall'] = all_correct / all_total * 100 # overall pck
        per_image_pck_dict['overall'] = np.mean(total_pck) * 100 # average pck per image
        if vocal:
            print(f'All per image PCK@0.1: {per_image_pck_dict["overall"]:.2f}')
            print(f'All per point PCK@0.1: {per_point_pck_dict["overall"]:.2f}')
        return per_point_pck_dict, per_image_pck_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', nargs='+', default=[261], type=int, help='t list for diffusion')
    parser.add_argument('--cat_list', nargs='+', default=[], type=str, help='Category list to process')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    evaluator = SPairEvaluator(args)
    evaluator.evaluate()
