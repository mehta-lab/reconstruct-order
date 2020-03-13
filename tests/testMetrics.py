
import numpy as np

def mse(y_pred, y_target):
    sqer = (y_target - y_pred)**2
    mse = (np.mean(sqer))**(1/2) / np.mean(np.abs(y_target))
    return mse
