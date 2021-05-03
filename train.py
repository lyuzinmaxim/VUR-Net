import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # работает медленнее, но зато воспроизводимость!


def au_and_bem_torch(nn_output, ground_truth, calc_bem: bool):
    """
    difference from "au_and_bem' is converting to np.ndarray and abs()

    calculates Binary Error Map (BEM) and Accuracy of Unwrapping (AU)
    for batch [batch_images,0,width,heidth] and returns mean AU of a batch
    with list of AU for every image and may be with BEM (optionally)

    function returns:
    au_mean - float, mean AU for batch
    au_list - list, info about AU for every image in batch
    bem - 3d boolean tensor, shows BEM in format [images_in_batch,width,height]

    args:
    nn_output - ndarray or torch.tensor - tensor that goes forward the net
    ground_truth - ndarray or tensor - ground truth image (original phase)
    calc_bem - boolean, if needed, will calculate BEM


    with input as np.ndarray runs 10 times faster
    """
    nn_output = nn_output.numpy()
    ground_truth = ground_truth.numpy()

    au_list = []
    bem = np.empty([
        len(nn_output[:, 0, 0, 0]),
        len(nn_output[0, 0, :, 0]),
        len(nn_output[0, 0, 0, :])
    ])

    for k in range(len(nn_output[:, 0, 0, 0])):
        min_height = 0
        cnt = 0
        for i in range(len(nn_output[0, 0, :, 0])):
            for j in range(len(nn_output[0, 0, 0, :])):
                x = abs(nn_output[k, 0, i, j] - ground_truth[k, 0, i, j])

                if calc_bem:

                    if x <= (ground_truth[k, 0, i, j] - min_height) * 0.05:
                        bem[k, i, j] = 1
                        cnt += 1
                    else:
                        bem[k, i, j] = 0

                else:
                    if x <= (ground_truth[k, 0, i, j] - min_height) * 0.05:
                        cnt += 1

        au = cnt / (len(nn_output[0, 0, :, 0]) * len(nn_output[0, 0, 0, :]))
        # print(k,'au:',au)
        au_list.append(au)

    au_mean = sum(au_list) / len(au_list)

    if calc_bem:
        return au_mean, au_list, bem
    else:
        return au_mean, au_list


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

