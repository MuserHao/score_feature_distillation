import argparse
import matplotlib.pyplot as plt
from SpairEvaluator import SpairEvaluator

def main(args):
    # Initialize the SpairEvaluator
    evaluator = SpairEvaluator(args.dataset_path, args.save_path, args.dift_model, args.img_size, args.up_ft_index, args.ensemble_size)

    # Define the range of 't' values to evaluate
    t_values = range(0, 1001, 100)

    # Initialize lists to store evaluation results
    per_image_pck = []
    per_point_pck = []

    for t in t_values:
        args.t = t
        # Evaluate using the SpairEvaluator
        per_image_pck.append(evaluator.evaluate_per_image_pck())
        per_point_pck.append(evaluator.evaluate_per_point_pck())

    # Plot and save the evaluation curve
    plt.figure()
    plt.plot(t_values, per_image_pck, label='Per Image PCK@0.1')
    plt.plot(t_values, per_point_pck, label='Per Point PCK@0.1')
    plt.xlabel('t')
    plt.ylabel('PCK')
    plt.title('Evaluation Metric vs. t')
    plt.legend()
    plt.savefig('evaluation_curve.png')

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
    main(args)
