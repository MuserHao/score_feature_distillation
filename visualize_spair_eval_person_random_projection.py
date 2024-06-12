import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_spair import SPairEvaluator
import os

def main(args):
    # Initialize the SpairEvaluator
    args.t = 0
    evaluator = SPairEvaluator(args)
    # cat_list = evaluator.get_cat_list()
    # cat_list.append('overall')
    cat_list = ['person']
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)


    # Define the range of 't' values to evaluate
    t_values = range(args.t_range[0], args.t_range[1], 10)

    # Initialize dict of lists to store evaluation results
    per_image_pck = {}
    per_point_pck = {}
    for cat in cat_list:
        per_image_pck[cat] = []
        per_point_pck[cat] = []

    evaluator.set_threshold(args.threshold)
    
    for t in tqdm(t_values):
        evaluator.set_t(t)
        # Evaluate using the SpairEvaluator
        for proj_dim in args.proj_dim:
            point_metric_dict, image_metric_dict = evaluator.evaluate_person_random_proj(proj_dim)
            for cat in cat_list:
                per_image_pck[cat].append(image_metric_dict[cat])
                per_point_pck[cat].append(point_metric_dict[cat])

        xxx = os.path.join(args.plot_path,f'{proj_dim}')
        os.makedirs(xxx, exist_ok=True)
        with open(os.path.join(args.plot_path,f'{proj_dim}', f'{cat}_evaluation_curve_Per_Image_PCK@0.1.txt'), 'w') as f:
            for tt, ipck in zip(t_values, per_image_pck[cat]):
                f.write(str(tt) + '\t' + str(ipck) + '\n')
        
        with open(os.path.join(args.plot_path, f'{proj_dim}',f'{cat}_evaluation_curve_Per_Point_PCK@0.1.txt'), 'w') as f:
            for tt, ipck in zip(t_values, per_point_pck[cat]):
                f.write(str(tt) + '\t' + str(ipck) + '\n')
        

    # Plot and save the evaluation curve
    for cat in cat_list:
        plt.figure()
        plt.plot(t_values, per_image_pck[cat], label='Per Image PCK@0.1')
        plt.plot(t_values, per_point_pck[cat], label='Per Point PCK@0.1')
        plt.xlabel('t')
        plt.ylabel('PCK')
        plt.title(f'PCK Metric vs. t - Category: {cat}')
        plt.legend()
        plt.savefig(os.path.join(args.plot_path, f'{cat}_evaluation_curve_{args.threshold}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='./temp_features/spair_ft_person/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t_range', nargs='+', type=int, default=[0, 999], help='range of t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--plot_path', type=str, default='./plot_res/plots_person_random_proj', help='path to save plots')
    parser.add_argument('--proj_dim', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    main(args)