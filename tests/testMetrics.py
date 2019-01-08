#!/usr/bin/env python
# title           : this_python_file.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :12/12/18
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

from sklearn.metrics import mean_squared_error
from datetime import datetime

def mse(y_pred, y_target):
    mse = mean_squared_error(y_target, y_pred)
    print("\tmse value = "+str(mse))
    return mse