
import numpy as np

def mse(y_pred, y_target):
    sqer = (y_target - y_pred)**2
    mse = np.mean(sqer)
    return mse
