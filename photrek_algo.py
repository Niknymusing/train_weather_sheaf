import torch
import numpy as np
import argparse
import os
from typing import Union, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compute_metrics(p, q):
    """
    Compute the three comparison metrics (accuracy, decisiveness, robustness) between two probability distributions.
    """
    # Ensure inputs are tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    else:
        p = p.clone().detach()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.clone().detach()
    
    # Ensure numerical stability
    eps = 1e-10
    p = torch.clamp(p, min=eps)
    p = p / p.sum()
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    
    # Compute metrics
    # Accuracy (Cross-Entropy)
    accuracy = torch.exp(torch.sum(p * torch.log(q)))
    
    # Decisiveness (Arithmetic Mean)
    decisiveness = torch.sum(p * q)
    
    # Robustness (-2/3 Mean)
    r = -2/3
    sum_p_q_r = torch.sum(p * torch.pow(q, r))
    robustness = torch.pow(sum_p_q_r, 1 / r)
    
    return {
        "accuracy": accuracy.item(),
        "decisiveness": decisiveness.item(),
        "robustness": robustness.item()
    }

def generate_probability_distribution(dist_type: str, N: int, x_values=None, **params) -> torch.Tensor:
    """
    Generate a discrete probability distribution by evaluating the PDF of the specified distribution.
    """
    if x_values is not None:
        x = x_values
    else:
        if dist_type == 'normal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            x = torch.linspace(mean - 4*std, mean + 4*std, steps=N)
        elif dist_type == 'student_t':
            x = torch.linspace(-10, 10, steps=N)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    if dist_type == 'normal':
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        pdf_values = torch.exp(-0.5 * ((x - mean)/std)**2) / (std * np.sqrt(2 * np.pi))
    elif dist_type == 'student_t':
        df = params.get('df', 1.0)
        pdf_values = torch.tensor(stats.t.pdf(x.numpy(), df))
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    p = pdf_values / pdf_values.sum()
    return p, x

def visualize_distributions(distribution_p: torch.Tensor, distribution_q: torch.Tensor,
                            x_values: torch.Tensor,
                            params_p: dict, params_q: dict,
                            metrics: Dict[str, float],
                            dist_type_p: str, dist_type_q: str):
    """
    Create comprehensive visualization of the probability distributions.
    """
    plt.figure(figsize=(12, 8))

    # Plot the probability distributions
    ax1 = plt.subplot(2, 1, 1)
    values_p = distribution_p.numpy()
    values_q = distribution_q.numpy()
    x = x_values.numpy()
    ax1.plot(x, values_p, label=f'{dist_type_p.capitalize()} Distribution P')
    ax1.plot(x, values_q, label=f'{dist_type_q.capitalize()} Distribution Q')
    ax1.set_title(f'Probability Distributions\nP: {dist_type_p.capitalize()}, Params: {params_p}\nQ: {dist_type_q.capitalize()}, Params: {params_q}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability')
    ax1.legend()

    # Metrics text
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    metrics_text = (
        f"Comparison Metrics between P and Q:\n"
        f"Accuracy (Geometric Mean):     {metrics['accuracy']:.6f}\n"
        f"Decisiveness (Arithmetic Mean): {metrics['decisiveness']:.6f}\n"
        f"Robustness (-2/3 Mean):        {metrics['robustness']:.6f}"
    )
    ax2.text(0.1, 0.8, metrics_text, va='top', fontfamily='monospace', fontsize=12)

    plt.tight_layout()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Calculate comparison metrics between two probability distributions')
    parser.add_argument('--distribution', type=str, choices=['normal', 'student_t'],
                        required=True, help='Type of distribution P to generate')
    parser.add_argument('--N', type=int, default=1000, help='Number of discrete points')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for normal distribution P')
    parser.add_argument('--std', type=float, default=1.0, help='Standard deviation for normal distribution P')
    parser.add_argument('--df', type=float, default=1.0, help='Degrees of freedom for Student t distribution P')
    parser.add_argument('--distribution2', type=str, choices=['normal', 'student_t'],
                        help='Type of distribution Q to generate')
    parser.add_argument('--mean2', type=float, help='Mean for normal distribution Q')
    parser.add_argument('--std2', type=float, help='Standard deviation for normal distribution Q')
    parser.add_argument('--df2', type=float, help='Degrees of freedom for Student t distribution Q')
    parser.add_argument('--output', type=str, help='Path to save visualization plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    try:
        # Generate probability distribution P with exact parameters
        params_p = {
            'mean': args.mean,
            'std': args.std,
            'df': args.df
        }

        if args.distribution == 'normal':
            mean_p = args.mean
            std_p = args.std
            x_values = torch.linspace(mean_p - 4*std_p, mean_p + 4*std_p, steps=args.N)
        elif args.distribution == 'student_t':
            x_values = torch.linspace(-10, 10, steps=args.N)
        else:
            raise ValueError(f"Unsupported distribution type: {args.distribution}")

        distribution_p, _ = generate_probability_distribution(args.distribution, args.N, x_values=x_values, **params_p)
        print(f"\nGenerated distribution P ({args.distribution}) with N={args.N}")
        print("Parameters P:", params_p)

        # Generate probability distribution Q
        if args.distribution2:
            distribution_type_q = args.distribution2
        else:
            distribution_type_q = args.distribution  # Use the same distribution type if not specified

        params_q = {}
        if distribution_type_q == 'normal':
            params_q['mean'] = args.mean2 if args.mean2 is not None else args.mean + 0.1
            params_q['std'] = args.std2 if args.std2 is not None else args.std + 0.1
        elif distribution_type_q == 'student_t':
            params_q['df'] = args.df2 if args.df2 is not None else args.df + 1
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type_q}")

        distribution_q, _ = generate_probability_distribution(distribution_type_q, args.N, x_values=x_values, **params_q)
        print(f"\nGenerated distribution Q ({distribution_type_q}) with N={args.N}")
        print("Parameters Q:", params_q)

        # Calculate metrics
        metrics = compute_metrics(distribution_p, distribution_q)

        # Print detailed statistics
        print("\n=== Comparison Metrics between P and Q ===")
        print(f"Accuracy (Geometric Mean):     {metrics['accuracy']:.6f}")
        print(f"Decisiveness (Arithmetic Mean): {metrics['decisiveness']:.6f}")
        print(f"Robustness (-2/3 Mean):        {metrics['robustness']:.6f}")
        print("\nDistribution P Statistics:")
        print(f"Sum of probabilities:          {distribution_p.sum().item():.6f}")
        print(f"Min probability:               {distribution_p.min().item():.6e}")
        print(f"Max probability:               {distribution_p.max().item():.6e}")
        print("\nDistribution Q Statistics:")
        print(f"Sum of probabilities:          {distribution_q.sum().item():.6f}")
        print(f"Min probability:               {distribution_q.min().item():.6e}")
        print(f"Max probability:               {distribution_q.max().item():.6e}")
        print("\n=========================")

        # Create and show/save visualization
        fig = visualize_distributions(distribution_p, distribution_q, x_values,
                                      params_p, params_q, metrics,
                                      args.distribution, distribution_type_q)
        if args.output:
            fig.savefig(args.output)
            print(f"Saved visualization to {args.output}")
        else:
            plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
