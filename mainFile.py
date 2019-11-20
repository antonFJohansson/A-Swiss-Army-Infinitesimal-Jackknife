# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:21:35 2019

@author: johaant
"""

from generate_data import generate_data_points
from optimize_theta import get_widehat_theta
from autograd import grad

num_data_train = 500
num_features = 2
x_data, y_data = generate_data_points(num_data_train, num_features)
widehat_theta = get_widehat_theta(x_data, y_data)
print(widehat_theta)










