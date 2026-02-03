import numpy as np

def compute_gradient(x, y, w, b):
    '''
    Input: 
    Training samples (x,y)
    parameters w & b for the current model
    Output:
    dj_dw and dj_db where j is the error function and dj is the partial derivative of the error function with respect to w or b
    '''
    m = x.shape[0]
    result = 0
    dj_dw, dj_db = 0, 0
    for i in range(m):
        f_wb = (w * x[i] + b)
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return (dj_dw,  dj_db)