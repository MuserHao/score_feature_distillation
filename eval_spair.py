import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.models.dift_sd import SDFeaturizer4Eval
from src.models.dift_adm import ADMFeaturizer4Eval
import os
import json
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable 
from collections import defaultdict

class SPairEvaluator:
    def __init__(self, args):
        self.args = args
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                print('%s: %s' % (str(arg), str(value)))

        self.dataset_path = args.dataset_path
        self.test_path = 'PairAnnotation/test'
        self.json_list = os.listdir(os.path.join(self.dataset_path, self.test_path)) # 12234
        self.all_cats = os.listdir(os.path.join(self.dataset_path, 'JPEGImages'))
        self.save_path = args.save_path
        self.img_size = args.img_size
        self.t = args.t
        self.up_ft_index = args.up_ft_index
        self.ensemble_size = args.ensemble_size

        if args.dift_model == 'sd':
            self.dift = SDFeaturizer4Eval(cat_list=self.all_cats)
        elif args.dift_model == 'adm':
            self.dift = ADMFeaturizer4Eval()

        self.cat2json = {}
        self.cat2img = {}
        # prepare json & image lists
        self.get_data_lists()

    def set_t(self, t):
        self.t = t

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_timediff(self, time_diff):
        self.time_diff = time_diff
        
    def set_coef(self, extrap_coef):
        self.extrap_coef = extrap_coef
        
    def set_plot_path(self, plot_path):
        self.plot_path = plot_path
        
    def set_repeat_direc(self, repeat_direc):
        self.repeat_direc = repeat_direc
        
    def get_cat_list(self):
        return self.all_cats.copy()
    
    def set_random_project(self, proj_dim):
        probabilities = [1/6, 2/3, 1/6]
        values = torch.tensor([1, 0, -1], dtype=torch.float32) * torch.sqrt(torch.tensor(3.0))
        random_matrix = torch.multinomial(torch.tensor(probabilities), 1280 * proj_dim, replacement=True)
        random_matrix = values[random_matrix].view(1280, proj_dim)
        return random_matrix

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
    
    def infer_and_save_features_persons(self):
        print("saving all test images' features...")
        os.makedirs(self.save_path, exist_ok=True)
        for cat in tqdm(['person']):
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
            
    def infer_and_save_features_persons_time_diff(self, t):
        print("saving all test images' features...")
        os.makedirs(self.save_path, exist_ok=True)
        self.all_cats = ['person']
        for cat in tqdm(self.all_cats):
            save_file_path = os.path.join(self.save_path, f'{cat}_{t}.pth')
            if os.path.exists(save_file_path):
                break
            output_dict = {}
            for image_path in self.cat2img[cat]:
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                output_dict[image_path] = self.dift.forward(img,
                                                    category=cat,
                                                    img_size=self.img_size,
                                                    t=t,
                                                    up_ft_index=self.up_ft_index,
                                                    ensemble_size=self.ensemble_size)
            torch.save(output_dict, save_file_path)
            
    def gaussian_random_proj(self, feature_in, random_matrix):
        batch_size, channels, height, width = feature_in.shape
        random_projection_matrix = random_matrix.to(feature_in.device)
        data_reshaped = feature_in.permute(0, 2, 3, 1).reshape(-1, random_matrix.shape[0])
        projected_data_reshaped = torch.matmul(data_reshaped, random_projection_matrix)
        feature_in = projected_data_reshaped.reshape(batch_size, height, width, random_matrix.shape[1]).permute(0, 3, 1, 2)
        return feature_in
    
    def evaluate_person(self, vocal=False):
        torch.cuda.set_device(0)
        self.infer_and_save_features_persons()
        
        total_pck = []
        all_correct = 0
        all_total = 0
        per_point_pck_dict = {}
        per_image_pck_dict = {}

        for cat in tqdm(['person']):
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
                    if (dist / threshold) <= self.threshold:
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
    
    def evaluate_person_random_proj(self, proj_dim):
        torch.cuda.set_device(0)
        self.infer_and_save_features_persons()
        random_matrix = self.set_random_project(proj_dim)
        total_pck = []
        all_correct = 0
        all_total = 0
        per_point_pck_dict = {}
        per_image_pck_dict = {}

        for cat in tqdm(['person']):
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
                src_ft = self.gaussian_random_proj(src_ft, random_matrix)
                trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
                trg_ft = self.gaussian_random_proj(trg_ft, random_matrix)
                
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
                    if (dist / threshold) <= self.threshold:
                        correct += 1
                        cat_correct += 1
                        all_correct += 1

                cat_pck.append(correct / total)
            total_pck.extend(cat_pck)

            per_point_pck_dict[cat] = cat_correct / cat_total * 100 # average pck
            per_image_pck_dict[cat] = np.mean(cat_pck) * 100 # average pck per image
            # if vocal:
            #     print(f'{cat} per image PCK@0.1: {per_image_pck_dict[cat]:.2f}')
            #     print(f'{cat} per point PCK@0.1: {per_point_pck_dict[cat]:.2f}')

        per_point_pck_dict['overall'] = all_correct / all_total * 100 # overall pck
        per_image_pck_dict['overall'] = np.mean(total_pck) * 100 # average pck per image
        # if vocal:
        #     print(f'All per image PCK@0.1: {per_image_pck_dict["overall"]:.2f}')
        #     print(f'All per point PCK@0.1: {per_point_pck_dict["overall"]:.2f}')
        return per_point_pck_dict, per_image_pck_dict
    
    def evaluate_person_grad1(self, vocal=False):
        """
        compute gradient of loss to feature map
        """
        torch.cuda.set_device(0)
        
        total_pck = []
        all_correct = 0
        all_total = 0
        per_point_pck_dict = {}
        per_image_pck_dict = {}

        torch.set_grad_enabled(True)
        for cat in tqdm(self.all_cats):
            output_dict = {}
            for image_path in self.cat2img[cat]:
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                feature = self.dift.forward_grad1(img,
                                                    category=cat,
                                                    img_size=self.img_size,
                                                    t=self.t,
                                                    up_ft_index=self.up_ft_index,
                                                    ensemble_size=1)
                net_output, latents, noise, latents_noisy, tensor_t = feature # 1x4x96x96
                sample, t_emb1, emb = net_output
                loss = F.mse_loss(sample.float(), noise.float(), reduction="none")
                loss = loss.mean()
                loss.backward(retain_graph=True)
                feature = self.dift.pipe.unet.up_sample.grad
                output_dict[image_path] = feature # 1280x48x48

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
                    if (dist / threshold) <= self.threshold:
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
    
    def evaluate_person_grad2(self, vocal=False):
        torch.cuda.set_device(0)
        torch.set_grad_enabled(True)

        res = defaultdict(list)
        self.all_cats = ['person']
        xxx = 0
        for cat in tqdm(self.all_cats):
            feature_folder = os.path.join(self.plot_path, cat + '_npys', str(self.t))
            os.makedirs(feature_folder, exist_ok=True)
            flag2 = False
            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()

                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]
                
                src_img_name = os.path.join(self.dataset_path, 'JPEGImages', cat, data['src_imname'])
                src_img = Image.open(src_img_name)
                trg_img_name = os.path.join(self.dataset_path, 'JPEGImages', cat, data['trg_imname'])
                trg_img = Image.open(trg_img_name)
                flag1 = False
                for idx in range(len(data['src_kps'])):
                    src_point = data['src_kps'][idx]
                    
                    features = self.dift.forward_grad2(src_img,
                                                    category=cat,
                                                    img_size=self.img_size,
                                                    t=self.t,
                                                    up_ft_index=self.up_ft_index,
                                                    ensemble_size=1, 
                                                    image_size = src_img_size,
                                                    point = src_point)
                    feature, gradient = features
                    save_src_feature = feature.cpu().numpy()
                    save_src_gradient = gradient.cpu().numpy()
                    np.save(os.path.join(feature_folder, 'src_' + data['src_imname'] + '_' + str(src_point[1]) + '_' + str(src_point[0]) + '_feature.npy'), save_src_feature)
                    np.save(os.path.join(feature_folder, 'src_' + data['src_imname'] + '_' + str(src_point[1]) + '_' + str(src_point[0]) + '_gradient.npy'), save_src_gradient)
                    src_norm_feature = F.normalize(feature.view(1, -1))
                    src_norm_gradient = F.normalize(gradient.view(1, -1))
                    
                    trg_point = data['trg_kps'][idx]
                    features = self.dift.forward_grad2(trg_img,
                                                    category=cat,
                                                    img_size=self.img_size,
                                                    t=self.t,
                                                    up_ft_index=self.up_ft_index,
                                                    ensemble_size=1, 
                                                    image_size = trg_img_size,
                                                    point = trg_point)
                    feature, gradient = features
                    trg_norm_feature = F.normalize(feature.view(-1, 1), dim=0)
                    trg_norm_gradient = F.normalize(gradient.view(-1, 1), dim=0)
                    save_trg_feature = feature.cpu().numpy()
                    save_trg_gradient = gradient.cpu().numpy()
                    np.save(os.path.join(feature_folder, 'trg_' + data['trg_imname'] + '_' + str(trg_point[1]) + '_' + str(trg_point[0]) + '_feature.npy'), save_trg_feature)
                    np.save(os.path.join(feature_folder, 'trg_' + data['trg_imname'] + '_' + str(trg_point[1]) + '_' + str(trg_point[0]) + '_gradient.npy'), save_trg_gradient)
                    cos_map_feature = torch.mm(src_norm_feature, trg_norm_feature).cpu().numpy()[0, 0]
                    cos_map_gradient = torch.mm(src_norm_gradient, trg_norm_gradient).cpu().numpy()[0, 0]
                    res[cat].append([data['src_imname'], data['trg_imname'], src_point[1], src_point[0], trg_point[1], trg_point[0], cos_map_feature, cos_map_gradient])
                    xxx += 1
                    print(f'saving to results {xxx}')   
            #         if xxx >= 1:
            #             flag1 = True
            #             break
            #     if flag1 == True:
            #         flag2 = True
            #         break
            # if flag2 == True:
            #     break
        return res
    
    def evaluate_person_grad3(self, vocal=False):
        torch.cuda.set_device(0)
        torch.set_grad_enabled(True)

        res = defaultdict(list)
        for cat in tqdm(['person']):
            feature_folder = os.path.join(self.plot_path, cat + '_npys', str(self.t))
            os.makedirs(feature_folder, exist_ok=True)
            
            dicts = {}
            point_num = 0
            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                    
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                
                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]  
                if src_imname not in dicts:
                    dicts[src_imname] = []                  
                    dicts[src_imname].append(src_img_size)
                if trg_imname not in dicts:
                    dicts[trg_imname] = []
                    dicts[trg_imname].append(trg_img_size)
                for idx in range(len(data['src_kps'])):
                    src_point = data['src_kps'][idx]
                    trg_point = data['trg_kps'][idx]
                    dicts[src_imname].append(src_point)
                    assert src_point[1] < src_img_size[0] and src_point[0] < src_img_size[1]
                    dicts[trg_imname].append(trg_point)
                    assert trg_point[1] < trg_img_size[0] and trg_point[0] < trg_img_size[1]
                    point_num += 2
            
            print(f'{cat} {len(dicts)} {point_num}')

            res_dict = {}
            for image_path, points in tqdm(dicts.items()):
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                # print(points)
                feature, gradients = self.dift.forward_grad3(
                                            img,
                                            category=cat,
                                            img_size=self.img_size,
                                            t=self.t,
                                            up_ft_index=self.up_ft_index,
                                            ensemble_size=1, 
                                            points = points)
                pre_folder = os.path.join(self.plot_path, str(self.t))
                os.makedirs(pre_folder, exist_ok=True)
                save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.pth')
                torch.save(feature.cpu(), save_feature)
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_gradient.pth')
                torch.save(gradients.cpu(), save_gradient)
                with open(os.path.join(self.plot_path, str(self.t), image_path + '_bbox.json'), 'w') as f:
                    json.dump(points, f)
                    
        return res
    
    def evaluate_person_grad4(self, vocal=False):
        torch.cuda.set_device(0)
        torch.set_grad_enabled(True)

        res = defaultdict(list)
        for cat in tqdm(['person']):
            feature_folder = os.path.join(self.plot_path, cat + '_npys', str(self.t))
            os.makedirs(feature_folder, exist_ok=True)
            
            dicts = {}
            point_num = 0
            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                    
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                
                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]  
                if src_imname not in dicts:
                    dicts[src_imname] = []                  
                    dicts[src_imname].append(src_img_size)
                if trg_imname not in dicts:
                    dicts[trg_imname] = []
                    dicts[trg_imname].append(trg_img_size)
                for idx in range(len(data['src_kps'])):
                    src_point = data['src_kps'][idx]
                    trg_point = data['trg_kps'][idx]
                    dicts[src_imname].append(src_point)
                    assert src_point[1] < src_img_size[0] and src_point[0] < src_img_size[1]
                    dicts[trg_imname].append(trg_point)
                    assert trg_point[1] < trg_img_size[0] and trg_point[0] < trg_img_size[1]
                    point_num += 2
            
            print(f'{cat} {len(dicts)} {point_num}')

            res_dict = {}
            for image_path, points in tqdm(dicts.items()):
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                # print(points)
                import time
                time1 = time.time()
                feature, gradients = self.dift.forward_grad4(
                                            img,
                                            category=cat,
                                            img_size=self.img_size,
                                            t=self.t,
                                            up_ft_index=self.up_ft_index,
                                            ensemble_size=1, 
                                            image_size=points[0],
                                            points = None)
                # print(feature.shape, gradients.shape, time.time() - time1)
                pre_folder = os.path.join(self.plot_path, str(self.t))
                os.makedirs(pre_folder, exist_ok=True)
                save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.pth')
                torch.save(feature.cpu(), save_feature)
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_gradient.pth')
                torch.save(gradients.cpu(), save_gradient)
                with open(os.path.join(self.plot_path, str(self.t), image_path + '_bbox.json'), 'w') as f:
                    json.dump(points, f)
                    
        return res
    
    def evaluate_person_grad5(self, vocal=False):
        torch.cuda.set_device(0)
        torch.set_grad_enabled(True)

        res = defaultdict(list)
        for cat in tqdm(['person']):
            feature_folder = os.path.join(self.plot_path, cat + '_npys', str(self.t))
            os.makedirs(feature_folder, exist_ok=True)
            
            dicts = {}
            point_num = 0
            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                    
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                
                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]  
                if src_imname not in dicts:
                    dicts[src_imname] = []                  
                    dicts[src_imname].append(src_img_size)
                if trg_imname not in dicts:
                    dicts[trg_imname] = []
                    dicts[trg_imname].append(trg_img_size)
                for idx in range(len(data['src_kps'])):
                    src_point = data['src_kps'][idx]
                    trg_point = data['trg_kps'][idx]
                    dicts[src_imname].append(src_point)
                    assert src_point[1] < src_img_size[0] and src_point[0] < src_img_size[1]
                    dicts[trg_imname].append(trg_point)
                    assert trg_point[1] < trg_img_size[0] and trg_point[0] < trg_img_size[1]
                    point_num += 2
            
            # print(f'{cat} {len(dicts)} {point_num}')

            res_dict = {}
            for image_path, points in tqdm(dicts.items()):
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                # print(points)
                import time
                time1 = time.time()
                feature_sample, gradients_t, gradients_z_sam = self.dift.forward_grad5(
                                            img,
                                            category=cat,
                                            img_size=self.img_size,
                                            t=self.t,
                                            up_ft_index=self.up_ft_index,
                                            ensemble_size=1, 
                                            image_size=points[0],
                                            points = None)
                # print(feature.shape, gradients.shape, time.time() - time1)
                pre_folder = os.path.join(self.plot_path, str(self.t))
                os.makedirs(pre_folder, exist_ok=True)
                # save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.pth')
                # torch.save(feature_sample, save_feature)
                # save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_t_gradient.pth')
                # torch.save(gradients_t, save_gradient)
                # save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_z_gradient.pth')
                # torch.save(gradients_z_sam, save_gradient)
                # with open(os.path.join(self.plot_path, str(self.t), image_path + '_bbox.json'), 'w') as f:
                #     json.dump(points, f)
                save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.npy')
                np.save(save_feature, feature_sample.numpy())
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_t_gradient.npy')
                np.save(save_gradient, gradients_t.numpy())
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_z_gradient.npy')
                np.save(save_gradient, gradients_z_sam.numpy())
                del feature_sample
                del gradients_t
                del gradients_z_sam
                    
        return res

    def evaluate_person_grad6(self, vocal=False):
        torch.cuda.set_device(0)
        torch.set_grad_enabled(True)

        res = defaultdict(list)
        for cat in tqdm(['person']):
            feature_folder = os.path.join(self.plot_path, cat + '_npys', str(self.t))
            os.makedirs(feature_folder, exist_ok=True)
            
            dicts = {}
            point_num = 0
            for json_path in self.cat2json[cat]:
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                    
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                
                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]  
                if src_imname not in dicts:
                    dicts[src_imname] = []                  
                    dicts[src_imname].append(src_img_size)
                if trg_imname not in dicts:
                    dicts[trg_imname] = []
                    dicts[trg_imname].append(trg_img_size)
                for idx in range(len(data['src_kps'])):
                    src_point = data['src_kps'][idx]
                    trg_point = data['trg_kps'][idx]
                    dicts[src_imname].append(src_point)
                    assert src_point[1] < src_img_size[0] and src_point[0] < src_img_size[1]
                    dicts[trg_imname].append(trg_point)
                    assert trg_point[1] < trg_img_size[0] and trg_point[0] < trg_img_size[1]
                    point_num += 2
            
            # print(f'{cat} {len(dicts)} {point_num}')

            res_dict = {}
            for image_path, points in tqdm(dicts.items()):
                img = Image.open(os.path.join(self.dataset_path, 'JPEGImages', cat, image_path))
                # print(points)
                import time
                time1 = time.time()
                feature_sample, gradients_t, gradients_z_sam = self.dift.forward_grad6(
                                            img,
                                            category=cat,
                                            img_size=self.img_size,
                                            t=self.t,
                                            up_ft_index=self.up_ft_index,
                                            ensemble_size=1, 
                                            image_size=points[0],
                                            points = None,
                                            repeat_direc = self.repeat_direc)
                # print(feature.shape, gradients.shape, time.time() - time1)
                pre_folder = os.path.join(self.plot_path, str(self.t))
                os.makedirs(pre_folder, exist_ok=True)
                # save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.pth')
                # torch.save(feature_sample, save_feature)
                # save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_t_gradient.pth')
                # torch.save(gradients_t, save_gradient)
                # save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_z_gradient.pth')
                # torch.save(gradients_z_sam, save_gradient)
                # with open(os.path.join(self.plot_path, str(self.t), image_path + '_bbox.json'), 'w') as f:
                #     json.dump(points, f)
                save_feature = os.path.join(self.plot_path,str(self.t), image_path + '_feature.npy')
                np.save(save_feature, feature_sample.numpy())
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_t_gradient.npy')
                np.save(save_gradient, gradients_t.numpy())
                save_gradient = os.path.join(self.plot_path, str(self.t), image_path + '_z_gradient.npy')
                np.save(save_gradient, gradients_z_sam.numpy())
                del feature_sample
                del gradients_t
                del gradients_z_sam
                    
        return res
    
    def evaluate_person_time_diff(self, vocal=False):
        torch.cuda.set_device(0)
        self.infer_and_save_features_persons_time_diff(self.t)
        self.infer_and_save_features_persons_time_diff(self.t-self.time_diff)
        
        total_pck = []
        all_correct = 0
        all_total = 0
        per_point_pck_dict = {}
        per_image_pck_dict = {}

        for cat in tqdm(['person']):
            output_dict = torch.load(os.path.join(self.args.save_path, f'{cat}_{self.t}.pth'))
            output_dict2 = torch.load(os.path.join(self.args.save_path, f'{cat}_{self.t-self.time_diff}.pth'))

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
                src_ft2 = output_dict2[data['src_imname']]
                trg_ft = output_dict[data['trg_imname']]
                trg_ft2 = output_dict2[data['trg_imname']]
                src_ft = src_ft2 + self.extrap_coef * (src_ft2 - src_ft)
                trg_ft = trg_ft2 + self.extrap_coef * (trg_ft2 - trg_ft)
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
                    if (dist / threshold) <= self.threshold:
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
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    evaluator = SPairEvaluator(args)
    evaluator.evaluate()
