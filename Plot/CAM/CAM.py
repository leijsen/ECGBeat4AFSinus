import os.path
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchcam
import csv

import matplotlib
matplotlib.rcParams.update({'font.size':30
                            , 'font.family': 'Times New Roman'
                            , 'axes.labelsize': 30
                            , 'axes.titlesize': 30
                            , 'xtick.labelsize': 30
                            , 'ytick.labelsize': 30
                            , 'legend.fontsize': 30
                            , 'figure.titlesize': 30
                            , 'figure.dpi': 300})

from wfdb import processing
from matplotlib.collections import LineCollection

def cal_pred_res(prob):
    test_pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(item[0])
        tmp_label.append(item[1])
        # tmp_label.append(item[0] - item[1])
        # tmp_label.append(item[1] - item[2])
        # tmp_label.append(item[2])
        test_pred.append(tmp_label)
    return test_pred

value = []
weights = []
def forward_hook(module, args, output):
    # value.append(args)
    value.append(output)


def hook_grad(module, grad_input, grad_output):
    # weights.append(grad_input)
    weights.append(grad_output)


# cam_test(model, show_case[i], (show_res[i], target), 'stage_list.4.block_list.0.conv1.conv', True)
def cam_test(
        index, # png path
        model, # Model
        data,  # ECG data signal my shape is torch.Size([1, 187])
        clas,  # a tuple with (predict, ground_truth)
        layer, # target layer to compute the graph like 'stage_list.4.block_list.0.conv1.conv'
        show_overlay=False):# if show the heatmap of signal

    #####################################################
    # register hooks

    # data = data.reshape(1, 1, 187)
    data= data.reshape(1, 1, 200)
    output_len = data.shape[-1]
    target_layer = model.get_submodule(layer)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(hook_grad)
    # clear list
    weights.clear()
    model.zero_grad()
    # forward
    value.clear()
    out = model(torch.tensor(data))
    # out = model(data)
    # softmax = torch.nn.Softmax()
    # out = softmax(out)
    out = cal_pred_res(out)
    # get the specific loss at the class cls
    # because the masked code can process the difference between predict and ground truth
    # so the clas was transfered as a tuple of size 2
    loss = out[0][int(clas[0])]
    loss.backward(retain_graph=True)
    # calculate the GradCAM actimap at clas[0]
    # GAP at the grad_map then fill the missing dimension
    # weight = weight[(...,) + (None,) * missing_dims]
    weight = weights[0][0].mean(-1)
    missing_dims = value[0].ndim - weight.ndim
    weight = weight[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    p_acti_map = F.relu(cam.sum(1))
    # print(p_acti_map)

    # gt class actimap
    weights.clear()
    model.zero_grad()
    loss = out[0][int(clas[1])]
    loss.backward()
    weight = weights[0][0].mean(-1)[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    g_acti_map = F.relu(cam.sum(1))

    forward_handle.remove()
    backward_handle.remove()

    #######################################################
    # build colored ECG
    #######################################################
    if show_overlay:
        p_acti_map = p_acti_map.detach().numpy()
        g_acti_map = g_acti_map.detach().numpy()
        p_new_acti = processing.resample_sig(p_acti_map.flatten(), p_acti_map.shape[-1], output_len)[0]
        g_new_acti = processing.resample_sig(g_acti_map.flatten(), g_acti_map.shape[-1], output_len)[0]

        p_new_acti = pd.Series(p_new_acti)
        g_new_acti = pd.Series(g_new_acti)
        p_new_acti = p_new_acti.interpolate()
        g_new_acti = g_new_acti.interpolate()

        x = np.arange(0, output_len)
        y = data.flatten()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        dydx = p_new_acti
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(3)
        line = axs.add_collection(lc)
        fig.colorbar(line, ax=axs)

        # dydx = g_new_acti
        # # Create a continuous norm to map from data points to colors
        # norm = plt.Normalize(dydx.min(), dydx.max())
        # lc = LineCollection(segments, cmap='viridis', norm=norm)
        # # Set the values used for colormapping
        # lc.set_array(dydx)
        # lc.set_linewidth(3)
        # line = axs[1].add_collection(lc)
        # fig.colorbar(line, ax=axs[1])

        axs.set_xlim(x.min(), x.max())
        axs.set_ylim(y.min()-0.3, y.max()+0.3)
        axs.set_title('predict class: ' + clas[0].__str__()[0])
        # axs[1].set_xlim(x.min(), x.max())
        # axs[1].set_ylim(y.min() - 0.3, y.max() + 0.3)
        # axs[1].set_title('ground truth class:' + clas[1].__str__())
        plt.tight_layout()
        plt.savefig(f'{png_path}/{index}.png')
        plt.close()
        # plt.show()
    ################################################################
    return p_acti_map

if __name__ == '__main__':
    model = torch.load('net1d.pth', map_location=torch.device('cpu'))
    model.eval()
    file_path = 'df_tp_data.csv'
    df = pd.read_csv(file_path)

    df = df.sample(frac=1).reset_index(drop=True)
    # print(df.head())
    train_X = df.iloc[:, 0:200]
    train_Y = df.iloc[:, 200]
    train_tensor_X = torch.tensor(train_X.values, dtype=torch.float32)
    train_tensor_Y = torch.tensor(train_Y.values, dtype=torch.float32)
    train_tensor_X = torch.unsqueeze(train_tensor_X, dim=1)
    # print(train_tensor_X[0].shape)
    global png_path
    for i in range(200):
        train_data = np.array(train_tensor_X[i])
        cls = np.array(train_tensor_Y[i])
        # layer_name = 'stage_list.1.block_list.1.conv1.conv' # good
        layer_name = 'stage_list.6.block_list.1.conv1' # good
        # layer_name = 'stage_list.3.block_list.1.conv1.conv' # good
        if 'tn' in file_path:
            png_path = f'.TNDenoiseattention_images{layer_name}'
        else:
            png_path = f'.TPDenoiseattention_images{layer_name}'
        if not os.path.exists(png_path):
            os.makedirs(png_path)
        cam_test(i, model, train_data, (cls, cls), layer_name, True)
        # cam_test(i, model, train_data, (cls, cls), layer_name, True)

