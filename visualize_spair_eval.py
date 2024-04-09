import argparse
import matplotlib.pyplot as plt
from eval_spair import SpairEvaluator

def main(args):
    # Initialize the SpairEvaluator
    args.t = 0
    evaluator = SpairEvaluator(args)

    # Define the range of 't' values to evaluate
    t_values = range(0, 1001, 20)

    # Initialize lists to store evaluation results
    per_image_pck = []
    pck = []

    for t in t_values:
        evaluator.set_t(t)
        # Evaluate using the SpairEvaluator
        metric_dict, per_image_metric_dict = evaluator.evaluate()
        per_image_pck.append(metric_dict['overall'])
        pck.append(per_image_metric_dict['overall'])

    # Plot and save the evaluation curve
    plt.figure()
    plt.plot(t_values, per_image_pck, label='Per Image PCK@0.1')
    plt.plot(t_values, pck, label='Per Point PCK@0.1')
    plt.xlabel('t')
    plt.ylabel('PCK')
    plt.title('Evaluation Metric vs. t')
    plt.legend()
    plt.savefig('evaluation_curve.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Visualize Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t_range', nargs='+', type=int, default=[0, 1001] help='range of t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    main(args)
