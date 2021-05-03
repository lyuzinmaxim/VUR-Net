import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def create_dataset_element(base_size, end_size, magnitude_min, magnituge_max):
    array = np.random.rand(base_size, base_size)
    coef = np.random.permutation(np.arange(magnitude_min, magnituge_max, 0.1))[0]
    element = cv2.resize(array, dsize=(end_size, end_size), interpolation=cv2.INTER_CUBIC)
    element = element * coef
    if np.min(element) >= 0:
        min_value = np.min(element)
        element = element - min_value
    else:
        min_value = np.min(element)
        element = element + abs(min_value)
    return element


def make_gaussian(number_of_gaussians, sigma_min, sigma_max, shift_max, magnitude_min, magnitude_max):
    element = np.zeros([256, 256])
    x = np.arange(-3.14, 3.14, 0.0246)
    y = np.arange(-3.14, 3.14, 0.0246)
    xx, yy = np.meshgrid(x, y);

    for i in range(number_of_gaussians):
        sigma = np.random.permutation(np.arange(sigma_min, sigma_max, .5))[0]
        shift_x = np.random.permutation(np.arange(-shift_max, shift_max, .1))[0]
        shift_y = np.random.permutation(np.arange(-shift_max, shift_max, .1))[0]
        magnitude = np.random.permutation(np.arange(magnitude_min, magnitude_max, .5))[0]

        d = np.sqrt((xx - shift_x) ** 2 + (yy - shift_y) ** 2)
        element += np.exp(-((d) ** 2 / (2.0 * sigma ** 2))) * (1 / sigma * np.sqrt(6.28))

    element = element / np.max(element)
    element = element * magnitude

    if np.min(element) >= 0:
        min_value = np.min(element)
        element = element - min_value
    else:
        min_value = np.min(element)
        element = element + abs(min_value)

    return element


def wraptopi(input_image):
    pi = 3.1415926535897932384626433
    output = input_image - 2 * pi * np.floor((input_image + pi) / (2 * pi))
    return output


if __name__ == "__main__":
    n = 5
    dataset = np.empty([n, 256, 256])
    for i in range(n):
        if i % 2 == 0:
            size = np.random.permutation(np.arange(2, 15, 1))[0]
            dataset[i] = create_dataset_element(size, 256, 4, 20)
        else:
            num_gauss = np.random.permutation(np.arange(1, 7, 1))[0]
            dataset[i] = make_gaussian(
                num_gauss,
                sigma_min=1,
                sigma_max=4,
                shift_max=4,
                magnitude_min=2,
                magnitude_max=20)
