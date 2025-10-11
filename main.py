import warnings, argparse
import random
import torch
import numpy as np
from time import time

from zen import ZEN



def create_splits(Y, k):
    """
    Generate train/validation/test indices for few-shot node classification.
    Each class contributes exactly k labeled nodes for training.

    Args:
        Y: Ground-truth labels
        k: Number of labeled samples per class

    Returns:
        (train_idx, val_idx, test_idx): Index splits
    """
    label_to_indices = {}
    for idx, label in enumerate(Y):
        label = label.item()
        label_to_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    for indices in label_to_indices.values():
        if len(indices) < 2 * k:
            if len(indices) < k:
                train = indices
                val = []
            else:
                train = random.sample(indices, len(indices))[:k]
                val = random.sample(indices, len(indices))[k:]
        else:
            train = random.sample(indices, 2 * k)[:k]
            val = random.sample(indices, 2 * k)[k:]
        train_indices.extend(train)
        val_indices.extend(val)

    all_indices = set(range(len(Y)))
    test_indices = all_indices - set(train_indices) - set(val_indices)
    
    return sorted(train_indices), sorted(val_indices), sorted(test_indices)



def generate_simplex_grid(n):
    """
    Generate a grid of (a0, a1, a2) coefficients on the 2-simplex:
    { (a0, a1, a2) | a0 + a1 + a2 = 1, a_i >= 0 }, discretized with denominator n.

    Total number of grid points: (n+2)(n+1)/2
    """
    grid = []
    for i in range(n+1):
        for j in range(n+1 - i):
            k = n - i - j
            a0, a1, a2 = i / n, j / n, k / n
            grid.append((a0, a1, a2))
    
    return grid



def hyperparam_tuning(H, X, Y, data_spilts, n):
    """
    Perform grid search over the 2-simplex to tune (a0, a1, a2) coefficients
    for each split based on validation accuracy.

    Args:
        H:            Incidence matrix
        X:            Node features
        Y:            Ground truth labels
        data_splits:  List of (train_idx, val_idx, test_idx) tuples
        n:            Grid resolution (discretization denominator)

    Returns:
        best_hyperparams: List of optimal coefficients per split
    """
    grid_points = generate_simplex_grid(n)

    best_hyperparams = []
    for idx, (train_idx, valid_idx, test_idx) in enumerate(data_splits):
        valid_accs = []
        hyperparams = []
        for hyperparam in grid_points:
            Y_hat = ZEN(H, X, Y, train_idx, hyperparam)
            pred = torch.argmax(Y_hat, dim=1)
            valid_acc = (pred == Y)[valid_idx].float().mean().item()

            valid_accs.append(valid_acc)
            hyperparams.append(hyperparam)

        best_hyperparams.append(hyperparams[torch.tensor(valid_accs).argmax()])

    return best_hyperparams



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Set random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Argument parsing
    parser = argparse.ArgumentParser('zen')
    parser.add_argument('-data', '--data', type=str, default='cora')
    parser.add_argument('-n', '--n', type=int, default=9)
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('-run', '--run', type=int, default=10)
    parser.add_argument('-device', '--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Load preprocessed hypergraph data
    H = torch.load('data/{0}/H.pt'.format(args.data)).to(args.device)
    X = torch.load('data/{0}/X.pt'.format(args.data)).to(args.device)
    Y = torch.load('data/{0}/Y.pt'.format(args.data)).to(args.device)
    data_splits = [create_splits(Y, args.k) for _ in range(args.run)]
    print(f"Dataset '{args.data}' successfully loaded.")

    # Tune hyperparameters on each split using validation set
    hyperparams = hyperparam_tuning(H, X, Y, data_splits, args.n)
    print("Hyperparameter tuning completed.")

    # Evaluate on test sets
    test_accs = []
    test_time = []
    for idx, (train_idx, valid_idx, test_idx) in enumerate(data_splits):
        start_time = time()
        Y_hat = ZEN(H, X, Y, train_idx, hyperparams[idx])
        pred = torch.argmax(Y_hat, dim=1)
        test_acc = (pred == Y)[test_idx].float().mean().item()

        test_accs.append(test_acc)
        test_time.append(time() - start_time)

    # Print averaged performance
    print("{0} | Accuracy: {1:.1f}+-{2:.1f} | Time: {3:.4f}".format(
        args.data,
        np.mean(test_accs) * 100,
        np.std(test_accs) * 100,
        np.mean(test_time)
    ))
