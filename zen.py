import torch
from torch_scatter import scatter



def degree_norms(H):
    """
    Compute degree-based normalization terms for nodes and hyperedges.

    Args:
        H:      Incidence matrix

    Returns:
        Dv:     (deg(v))^(-0.5)   (|V|, 1)
        Dv_1:   (deg(v) - 1)^(-1) (|V|, 1)
        De_1:   (deg(e) - 1)^(-1) (|E|, 1)
    """
    ones = torch.ones(H.shape[1], dtype=torch.int64).to(H.device)

    Dv = scatter(src=ones, index=H[0], dim=0, reduce='sum')
    Dv_1 = Dv - 1
    De_1 = scatter(src=ones, index=H[1], dim=0, reduce='sum') - 1

    Dv = Dv ** (-0.5)
    Dv_1 = Dv_1 ** (-1.0)
    De_1 = De_1 ** (-1.0)

    Dv[Dv.isinf()] = 0
    Dv_1[Dv_1.isinf()] = 0
    De_1[De_1.isinf()] = 0

    return Dv.unsqueeze(1), Dv_1.unsqueeze(1), De_1.unsqueeze(1)



def RSIs(H, Dv_1, De_1):
    """
    Compute residual self-information (RSI) terms.

    Args:
        H:       Incidence matrix
        Dv_1:    (deg(v) - 1)^(-1)
        De_1:    (deg(e) - 1)^(-1)

    Returns:
        RSI_1:   First-order residual term
        RSI_2:   Second-order residual term
    """
    RSI_1 = scatter(src=De_1[H[1]], index=H[0], dim=0, reduce='sum')

    De2 = De_1 ** (2.0)

    RSI_2 = scatter(src=Dv_1[H[0]], index=H[1], dim=0, reduce='sum')
    RSI_2 = De2 * RSI_2
    RSI_2 = scatter(src=RSI_2[H[1]], index=H[0], dim=0, reduce='sum')

    correction_term = scatter(src=De2[H[1]], index=H[0], dim=0, reduce='sum')
    RSI_2 = RSI_2 - Dv_1 * correction_term

    return RSI_1, RSI_2



def propagation(H, X, De_1, RSI_1):
    """
    Perform normalized hypergraph message passing with residual correction.

    Args:
        H:       Incidence matrix
        X:       Node features
        De_1:    (deg(e) - 1)^(-1)
        RSI_1:   First-order residual self-information term

    Returns:
        Z: Updated node features after propagation
    """
    Z = scatter(src=X[H[0]], index=H[1], dim=0, reduce='sum')
    Z = De_1 * Z
    Z = scatter(src=Z[H[1]], index=H[0], dim=0, reduce='sum')

    Z = Z - RSI_1 * X

    return Z



def ZEN(H, X, Y, train_idx, hyperparams):
    """
    ZEN: Zero-parameter HypErgraph Neural network

    Computes class logits via a closed-form prediction formula without learnable parameters.

    Args:
        H:            Incidence matrix
        X:            Node features
        Y:            Ground truth labels
        train_idx:    Indices used for training
        hyperparams:  Tuple (a0, a1, a2) in 2-simplex

    Returns:
        Y_hat: Logits for all nodes
    """
    Dv, Dv_1, De_1 = degree_norms(H)
    RSI_1, RSI_2 = RSIs(H, Dv_1, De_1)

    Z1 = Dv * X
    AX = propagation(H, Z1, De_1, RSI_1)
    
    Z2 = Dv_1 * AX
    AAX = propagation(H, Z2, De_1, RSI_1)

    AAX = AAX - RSI_2 * Z1

    AX = Dv * AX
    AAX = Dv * AAX

    Z = torch.nn.functional.normalize(hyperparams[0] * X + hyperparams[1] * AX + hyperparams[2] * AAX)

    W = scatter(src=Z[train_idx], index=Y[train_idx], dim=0, reduce='sum')
    W = torch.nn.functional.normalize(W)
    
    Y_hat = Z @ W.T

    return Y_hat
