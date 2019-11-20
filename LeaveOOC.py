# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:59:40 2019

@author: johaant
"""

import numpy as np


def LOOCV(x_data):

    num_data_train = x_data.shape[0]
    num_features = x_data.shape[1] + 1
    
    store_loss = np.zeros(num_data_train)
    store_theta = np.zeros((x_data.shape[0], num_features))
    
    
    for rem_ind in range(num_data_train):
      
      if (rem_ind % (num_data_train // 10)) == 0 or rem_ind == (num_data_train - 1) or rem_ind == 0:
        print('Iteration {}/{}'.format(rem_ind,num_data_train))
      loss_n, theta_n = LOOCV(rem_ind)
      store_loss[rem_ind] = loss_n
      store_theta[rem_ind,:] = theta_n 


    return store_loss, store_theta


