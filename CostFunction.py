import numpy as np
def cost(h,y,m):
    c = - 1/m * np.sum( (y * np.log(h)) + ( (1-y) * np.log(1-h)) )
    return c
