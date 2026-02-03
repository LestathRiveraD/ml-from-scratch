import numpy as np

def compute_cost(x, y, w, b):
    '''
    Input: 
    Training samples (x,y) and parameterers w and b
    Output:
    The total cost for parameters w & b
    '''
    
    m = x.shape[0] # Total number of samples
    result = 0
    
    for i in range(m):
        result += ((w * x[i] + b) - y[i]) ** 2
    result *= (1 / (2 * m))
    return result