import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_spair import SPairEvaluator
import os

def main(args):
    # Initialize the SpairEvaluator
    # args.save_path = args.save_path + f'_{args.time_diff}/'
    args.t = 0
    evaluator = SPairEvaluator(args)
    # cat_list = evaluator.get_cat_list()
    # cat_list.append('overall')
    cat_list = ['person']
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # Define the range of 't' values to evaluate

    t_values = range(args.t_range[0], args.t_range[1], args.time_diff)

    # Initialize dict of lists to store evaluation results
    per_image_pck = {}
    per_point_pck = {}
    for cat in cat_list:
        per_image_pck[cat] = []
        per_point_pck[cat] = []

    evaluator.set_threshold(args.threshold)
    evaluator.set_timediff(args.time_diff)
    evaluator.set_plot_path(args.plot_path)
    # t_values = [0]
    for t in tqdm(t_values):
        evaluator.set_t(t)
        # print(f'start evaluation2')
        # Evaluate using the SpairEvaluator
        res = evaluator.evaluate_person_grad4()

        save_folder = os.path.join(args.plot_path, cat)
        os.makedirs(save_folder, exist_ok=True)
        
        with open(os.path.join(save_folder, f'{t}' + '.txt'), 'w') as f:
            for rr in res[cat]:
                src_imname, trg_imname, src_point1, src_point0, trg_point1, trg_point0, cos_map_feature, cos_map_gradient = rr
                f.write(src_imname + '\t' + trg_imname + '\t' + str(src_point1) + '\t' + str(src_point0) + '\t' + str(trg_point1) + '\t' + str(trg_point0) + '\t' + str(cos_map_feature) + '\t' + str(cos_map_gradient) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='./temp_features/spair_ft_person_grad4_timediff5', help='path to save features')
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
    parser.add_argument('--plot_path', type=str, default='./plot_res/plots_person_grad4_timediff5', help='path to save plots')
    args = parser.parse_args()
    main(args)

"""
The feature used for computing performance is the gradient of the feature map to t
"""