import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_spair import SPairEvaluator
import os


def increment(initial_list, step):
    incremented_list = []    
    for value in initial_list:
        incremented_list.append(value + step)
    return incremented_list

def main(args):
    # Initialize the SpairEvaluator
    evaluator = SPairEvaluator(args)
    cat_list = evaluator.get_cat_list()
    cat_list.append('overall')
    os.makedirs(args.plot_path, exist_ok=True)

    # Define the range of 't' values to evaluate
    t_vals = [args.t]
    while t_values[-1][-1] < args.step_end[1]:
        t_vals.append(increment(t_vals[-1], args.step_end[0]))
    t_values = [sublist[0] for sublist in t_vals]

    # Initialize dict of lists to store evaluation results
    per_image_pck = {}
    per_point_pck = {}
    for cat in cat_list:
        per_image_pck[cat] = []
        per_point_pck[cat] = []

    for t in tqdm(t_vals):
        evaluator.set_t(t)
        # Evaluate using the SpairEvaluator
        point_metric_dict, image_metric_dict = evaluator.evaluate()
        for cat in cat_list:
            per_image_pck[cat].append(image_metric_dict[cat])
            per_point_pck[cat].append(point_metric_dict[cat])

    # Plot and save the evaluation curve
    for cat in cat_list:
        plt.figure()
        plt.plot(t_values, per_image_pck[cat], label='Per Image PCK@0.1')
        plt.plot(t_values, per_point_pck[cat], label='Per Point PCK@0.1')
        plt.xlabel('t')
        plt.ylabel('PCK')
        plt.title(f'PCK Metric vs. t - Category: {cat}')
        plt.legend()
        plt.savefig(os.path.join(args.plot_path, f'{cat}_evaluation_curve.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm', 'sdh'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', nargs='+', default=[0], type=int, help='initial t for diffusion')
    parser.add_argument('--step_end', nargs='+', type=int, default=[10, 1000], help='step for t increment and end t value: [step, end]')
    parser.add_argument('--cat_list', nargs='+', default=[], type=str, help='Category list to process')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--plot_path', type=str, default='./plots', help='path to save plots')
    args = parser.parse_args()
    main(args)
