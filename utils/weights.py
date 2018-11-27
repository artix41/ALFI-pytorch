import torch

def weight_last(loss_t, device=torch.device("cpu")):
    return loss_t[-1]
    
def weight_const(loss_t, device=torch.device("cpu")):
    return loss_t.mean()
    
def weight_exp(loss_t, beta=3., device=torch.device("cpu")):
    beta = torch.tensor(beta).to(device)
    def weight(x):
        return (torch.exp(beta * x) - 1) / (torch.exp(beta) - 1)
    T = loss_t.shape[0]
    weights = weight(torch.linspace(0,1,T).to(device))
    weights = weights / weights.sum()
    
    return (weights * loss_t).sum()

def weight_oi(loss_t, device=torch.device("cpu")):
    n = loss_t.shape[0]
    ones_tril = torch.ones(n, n).tril() - torch.eye(n)
    ones_tril = ones_tril.to(device)
    loss_matrix = loss_t.repeat(loss_t.shape[0], 1) * ones_tril
    loss_matrix = loss_matrix + loss_matrix[1, 0] * torch.ones(n, n).triu().to(device)
    loss_matrix[0] = 0
    min_t = torch.min(loss_matrix, dim=1)[0]
    
    return torch.mean(loss_t - min_t)
