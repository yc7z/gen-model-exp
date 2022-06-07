
import torch
import torch.nn.functional as F

def ELBO(x, x_hat, mu, logvar, gamma, reduction='sum'):
    """
    x.shape = (B, C, H, W)
    
    Returns 
        L(x) = 1/B sum_i L(x_i)
    where
        L(x_i) = E_q[log(q(z|x)) - log(p(z))] - E_q[log(p(x|z))]
               = D_KL(q(z|x_i)||p(z)) + L_rec(x_i, x_hat_i, gamma)
               = 1/2 * sum_k [mu_k^2 + exp(logvar_k) - logvar_k - 1]
                + gamma * ||x_i - x_hat_i||^2
    """
    # get KL divergence
    kl_div = KL_div_qp(mu, logvar) # shape (B)
    
    # Get reconstruction loss
    L_rec = F.mse_loss(x, x_hat, reduction='none') # shape (B, C, H, W)
    L_rec = torch.sum(L_rec, dim=(1, 2, 3)) # shape (B)
    
    # get ELBO loss
    L_elbo = kl_div + gamma * L_rec # shape (B)
    
    if reduction == 'sum':
        L_elbo = L_elbo.sum()
    elif reduction == 'mean':
        L_elbo = L_elbo.mean()
    
    return L_elbo


def KL_div_qp(mu, logvar):
    """
    mu.shape = logvar.shape = (B, latent_dim)
    
    Assumptions
        q(z|x) = N(z; mu, exp(logvar)*I) (approximate posterior)
        p(z) = N(z; 0, I) (prior)
        
    Returns 
        D_KL(q(z|x)||p(z)) = E_q[log(q(z|x)) - log(p(z))]
                           = 1/2 * sum_i mu_i^2 + exp(logvar_i) - logvar_i - 1
    """

    kl_div = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1, dim=-1) 
    
    return kl_div # shape (B)