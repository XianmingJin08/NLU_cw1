import matplotlib as plot
import seaborn as sns
import numpy as np


def get2DResults (dict):
    p1 = {}
    p2 = {}
    p3 = {}
    output = {}

    for (a,b,c) in dict.keys():
        p1.

    return p1,p2,p3


if __name__ == "__main__":
    dict2a = np.load('dict2a.npy', allow_pickle='TRUE').item()
    dict2aExtra = np.load('dict2aExtra.npy', allow_pickle='TRUE').item()
    dict3 = np.load('dict3.npy', allow_pickle='TRUE').item()
    dict_acc_a = np.load('dict_acc_a.npy', allow_pickle='TRUE').item()
    dict_acc_b = np.load('dict_acc_b.npy', allow_pickle='TRUE').item()
    dict_acc_b = np.load('dict_acc_c.npy', allow_pickle='TRUE').item()
    dict_loss_a = np.load('dict_loss_a.npy', allow_pickle='TRUE').item()
    dict_loss_b = np.load('dict_loss_b.npy', allow_pickle='TRUE').item()
    dict_loss_c = np.load('dict_loss_c.npy', allow_pickle='TRUE').item()

    lrs,hUnits,steps = getParameters(dict2a)
    print (lrs)