
import numpy as np


def MMD_RBF(x, y):
    num_data, num_feats = x.shape
    
    xx = np.matmul(x, x.T)
    yy = np.matmul(y, y.T)
    xy = np.matmul(x, y.T)
    
    rx = np.diag(xx).unsqueeze(0).expand_as(xx)
    ry = np.diag(yy).unsqueeze(0).expand_as(yy)
    
    dxx = rx.T + rx - 2*xx
    dyy = ry.T + ry - 2*yy
    dxy = rx.T + ry - 2*xy
    
    kxx = np.zeros(xx.shape)
    kyy = np.zeros(yy.shape)
    kxy = np.zeros(xy.shape)
    
    bandwidths = [10, 15, 20, 50]
    for bw in bandwidths:
        kxx += np.exp(-alpha*dxx/bw)
        kyy += np.exp(-alpha*dyy/bw)
        kxy += np.exp(-alpha*dxy/bw)

    mn = 1./(num_data*(num_data-1)*len(bandwidths))
    mm = 2./(num_data*num_data*len(bandwidths))
    score = mn*np.sum(kxx) - mm*np.sum(kxy) + mn*torch.sum(kyy)

    return score


def MMD2(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    
    return torch.mean(XX + YY - 2. * XY)
