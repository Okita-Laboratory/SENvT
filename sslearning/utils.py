import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
import argparse

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")



def plot_fig(log, figsize=(16,8)):
    fig = plt.figure(figsize=figsize)

    plt.plot(log['train_losses'], label='train')
    plt.plot(log['valid_losses'], label='valid')
    plt.title('loss')
    plt.legend()
    return plt



def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights