# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:23:48 2019

@author: johaant
"""

##
import numpy as np

def generate_data_points(num_data_train, num_features, plot_points = False):
    
    #num_data_train = 500
    #num_features = 2
    x_data = np.random.normal(0,10,(num_data_train, num_features))
    theta = np.random.normal(0,1,num_features + 1)
    
    
    def logistic(x):
      return 1 / (1 + np.exp(-x))
    
    def generate_y(x, theta):
      y_prob = logistic(np.sum(x_data * theta[:-1], axis = 1) + theta[-1])
      class_vector = np.random.binomial(1,y_prob)
      return class_vector
    
    
    y_data = generate_y(x_data, theta)
    
    if plot_points == True:
        x_plot = np.linspace(-3,3,100)
        y_plot = -theta[-1]/theta[1] - theta[0]/theta[1]*x_plot
        import matplotlib.pyplot as plt
        plt.scatter(x_data[y_data == 0,0],x_data[y_data == 0,1], c = 'r')
        plt.scatter(x_data[y_data == 1,0],x_data[y_data == 1,1], c = 'b')
        plt.plot(x_plot, y_plot)
    return (x_data, y_data)
