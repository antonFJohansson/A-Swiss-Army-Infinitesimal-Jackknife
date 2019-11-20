# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:42:32 2019

@author: johaant
"""


## So we fit the loglike loss here
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
def logistic(x):
  return 1 / (1 + np.exp(-x))
    

def loss_func(theta,x,y):
  EPS = 1e-8
  probs = logistic(np.sum(x * theta[:-1], axis = 1) + theta[-1])
  loglike = np.sum(np.log(probs[y==1]+ EPS)) + np.sum(np.log(1-probs[y==0]+ EPS))
  return -loglike

def get_predictions(x, theta):

  y_prob = logistic(np.sum(x * theta[:-1], axis = 1) + theta[-1])
  one_class = np.where(y_prob > 0.5)
  class_vec = np.zeros(x.shape[0])
  class_vec[one_class] = 1
  y_data = class_vec
  return y_data


def get_widehat_theta(x_data, y_data, max_epoch = 150, lr = 1e-3, plot_res = True):
    
    ## So this should work in some way

    ## So this can train my method now
    
    theta_prop = np.random.normal(0,1,x_data.shape[1] + 1)
    grad_loss = grad(loss_func)
    
    for epoch in range(max_epoch):
      gradient = grad_loss(theta_prop, x_data, y_data) ## but maybe I can just insert what I want here?
      theta_prop -= gradient*lr
    
    if plot_res == True:
        y_data_prop = get_predictions(x_data, theta_prop)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].scatter(x_data[y_data == 0,0],x_data[y_data == 0,1], c = 'r')
        ax[0].scatter(x_data[y_data == 1,0],x_data[y_data == 1,1], c = 'b')
        ax[1].scatter(x_data[y_data_prop == 0,0],x_data[y_data_prop == 0,1], c = 'r')
        ax[1].scatter(x_data[y_data_prop == 1,0],x_data[y_data_prop == 1,1], c = 'b')
    return theta_prop