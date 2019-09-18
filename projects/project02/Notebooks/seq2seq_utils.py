
# coding: utf-8

# In[25]:


import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import unicodedata
import string 
import re
import time 
from torch.autograd import Variable
import math 
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[27]:


def seq_seq_plot(accuracy, epoch):
    t = np.arange(0, epoch, 1)
    plt.plot(t, accuracy, 'r--')
    plt.show()


# In[22]:


def confusion_parameters(scores,target_tensor,inpute_tensor):
    
    conf_counter= torch.zeros(4)

  
    if torch.all(torch.eq(target_tensor, scores.argmax(dim=1)))==1 and torch.all(torch.eq(inpute_tensor,target_tensor))==0: ### making incorrect->correct
        conf_counter[0] +=1

    if torch.all(torch.eq(inpute_tensor, scores.argmax(dim=1)))==0 and torch.all(torch.eq(inpute_tensor,target_tensor))==1: ### making correct->incorrect

        conf_counter[1] +=1    


    if torch.all(torch.eq(inpute_tensor, scores.argmax(dim=1)))==1 and torch.all(torch.eq(inpute_tensor,target_tensor))==1: ### making correct->correct

        conf_counter[2] +=1                                                               

    if torch.all(torch.eq(inpute_tensor, scores.argmax(dim=1)))==1 and torch.all(torch.eq(inpute_tensor,target_tensor))==0: ### making inccorrect->correct or regenerate the input 

        conf_counter[3] +=1  
    return conf_counter


# In[30]:


def plot_loss(all_parameters ):
    train_accuracy = all_parameters['train_accuracy']
    test_accuracy  = all_parameters['test_accuracy']
    train_Loss     = all_parameters['train_loss']
    test_Loss      =all_parameters['test_loss']
    epoch= all_parameters['epoch']

    t1=np.arange(0,epoch, 1)
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.plot(t1, train_Loss,'r-')
    plt.xlabel('number of epochs')
    plt.ylabel('Train loss')
    # plt.figure(figsize=(18, 6))
    plt.subplot(122)
    plt.plot(t1, test_Loss,'k')
    plt.xlabel('number of epochs')
    plt.ylabel('test loss')
    # plt.plot(t1, train_accuracy,'r-', t1,test_accuracy,'k', label='test Accuracy' )
    plt.show()
    # plt.legend(['train_accuracy', ' test_accuracy'], loc='upper right', prop={'size': 10})
    # plt.subplot(212)
    # plt.plot(t1, train_Loss,'r-', t1,test_Loss,'k' )
    # plt.legend(['train_Loss', ' test_Loss'], loc='upper right', prop={'size': 10})


# In[33]:


def pie_plot_lstm(all_parameters):
    train_accuracy = all_parameters['train_accuracy']
    test_accuracy  = all_parameters['test_accuracy']
    train_Loss     = all_parameters['train_loss']
    test_Loss      =all_parameters['test_loss']
    epoch= all_parameters['epoch']

    a=train_accuracy[epoch-1]
    b=test_accuracy[epoch-1]
    b=torch.tensor(b)

    labels = 'train_accuracy', 'error'
    labels2='test_accuracy', 'error'
    sizes = [a, 100-a]
    sizes2 = [b, 100-b]

    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

    plt.subplot(122)
    plt.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.show()


# In[ ]:


#  def plot_lstm(model_param):
    
#     plt.figure(figsize=(18, 6))
#     plt.subplot(1,2,1)
#     plt.plot(train_history_over_CV[0][0], train_history_over_CV[0][1])
#     plt.plot(val_history_over_CV[0][0], val_history_over_CV[0][1])
#     plt.legend(['Average Train Loss', ' AverageValidation Loss'], loc='upper right', prop={'size': 15})

#     plt.subplot(1,2,2)
#     plt.plot(train_history_over_CV[0][0], train_history_over_CV[0][2])
#     plt.plot(val_history_over_CV[0][0], val_history_over_CV[0][2])
#     plt.legend(['Average Train Error', 'Average Validation Error'], loc='upper right', prop={'size': 15})
#     plt.yticks(np.arange(0, 100, step=5))

#     plt.suptitle('Cross Validation Performance', fontsize=15.0, y=1.08, fontweight='bold')

