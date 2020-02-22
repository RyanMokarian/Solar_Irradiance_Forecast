import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

sns.set() # matplotlib plots are more beautiful with seaborn ッ

VISUALIZATION_DIR = 'visualizations'

def check_dir_exist():
    if not os.path.exists(VISUALIZATION_DIR):
        os.mkdir(VISUALIZATION_DIR)

def plot_loss(train_losses: list, valid_losses: list, csky_loss_valid: float = None, title: str = 'Training Loss', loss: str = 'RMSE'):
    """
    This function plots the training and validation losses. It can also plot the loss 
    of the clear sky model as a baseline. 
    """
    
    assert len(train_losses) == len(valid_losses), f'Number of train losses ({len(train_losses)}) \
                                                    is not equal to validation losses ({len(validation_losses)})'
    
    # Plot
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    if csky_loss_valid != None:
        plt.plot([csky_loss_valid]*len(train_losses), label='csky_valid', linestyle='--', color='red')
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss}')
    plt.title(title)
    plt.legend()
    
    # Save plot
    check_dir_exist()
    filename = title + '_' + loss + '_' + str(datetime.now())
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename+'.png'))
    data = {'train_losses':train_losses, 'valid_losses':valid_losses, 'csky_loss':csky_loss_valid}
    pickle.dump(data, open(os.path.join(VISUALIZATION_DIR,filename+'.pkl'), 'wb'))