import torch
from torchTools import * 

def signed_area_triangle (x) : 
    """ 
    x is a tensor of shape [N, 3, 2]
    """ 
    pB = x[:, 1, :] - x[:, 0, :] # [N, 2]
    pC = x[:, 2, :] - x[:, 0, :] # [N, 2]

    xB = pB[:, 0]
    yB = pB[:, 1]

    xC = pC[:, 0]
    yC = pC[:, 1]

    return 0.5 * (xB * yC - xC * yB) # [N]

def signed_area_oriented_polygon (x) : 
    """ 
    x is a tensor of shape [N, K, 2]
    """
    N, K, *_ = x.shape
    areas = []
    for i in range(K - 2) : 
        A = x[:, 0:1, :] 
        BC = x[:, i+1:i+3,:] 
        ABC = torch.cat((A, BC), 1)
        areas.append(signed_area_triangle(ABC))
    all_areas = torch.stack(areas) # [K - 2, N]
    area = all_areas.sum(0) # [N] 
    return area

def rel_orientation_loss (x, y) : 
    """ 
    x - [N, 4, 2]
    y - [N, 4, 2]
    """ 
    x_cyc = torch.cat((x, x[:, :1, :]), 1)
    y_cyc = torch.cat((y, y[:, :1, :]), 1)

    x_deltas = x_cyc[:, 1:, :] - x_cyc[:, :-1, :] 
    y_deltas = y_cyc[:, 1:, :] - y_cyc[:, :-1, :] 

    x_del_norm = unitNorm(x_deltas, 2) 
    y_del_norm = unitNorm(y_deltas, 2) 

    dot_product = (x_del_norm * y_del_norm).sum(2) 

    one_minus_dot_product = (1 - dot_product).max(1)[0]

    loss = one_minus_dot_product.mean()

    return loss

if __name__ == "__main__" : 
    x = torch.tensor([[0., 0.], [0., 10.0], [5.0, 10.0], [5.0, 0.0]]).unsqueeze(0)
    print(rel_orientation_loss(x, x))


