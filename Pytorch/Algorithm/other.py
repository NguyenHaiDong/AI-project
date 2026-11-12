import torch




def get_accuracy_rate(A, B):
    return (A == B).float().mean().item() * 100.0

def get_full_accuracy_rate(A, B):
    sum1 = torch.sum(A, dim=0)
    sum0 = A.shape[0] - sum1
    total = torch.sum(2.0 * sum0 * sum1)

    one = ((A == 1) & (B == 1)).sum(dim = 0)
    zero = ((A == 0) & (B == 0)).sum(dim = 0)
    accuracy = ((one * sum0) + (zero * sum1)).sum() / total
    return (accuracy * 100.0).item()


def clamp_grad(list_g, max_norm = 1.0):
    cur_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in list_g))
    clip_coef = max_norm / (cur_norm + 1e-6)
    if clip_coef < 1.0:
        for g in list_g:
            g.mul_(clip_coef)

def get_better_randn(n):
    return (2.0 / n) ** 0.5