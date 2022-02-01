import os
import matplotlib.pyplot as plt
import torch

from dataset.facade_dataset import ECPDataset

DATASET_ROOT = '/mnt/hdd/datasets/facade'

dataset = ECPDataset(os.path.join(DATASET_ROOT, 'ecp', 'cvpr2010'))

for i, item in enumerate(dataset):

    plt.imshow(item[0])
    plt.show()
    plt.imshow(dataset.label2image(item[1]))
    plt.show()

    print(i)
