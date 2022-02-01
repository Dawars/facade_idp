import os
import matplotlib.pyplot as plt

from dataset.facade_dataset import ECPDataset, Graz50Dataset

DATASET_ROOT = '/mnt/hdd/datasets/facade'

# dataset = ECPDataset(os.path.join(DATASET_ROOT, 'ecp', 'cvpr2010'))
dataset = Graz50Dataset(os.path.join(DATASET_ROOT, 'graz50'))

for i, item in enumerate(dataset):

    plt.imshow(item[0])
    plt.show()
    label_image = dataset.label2image(item[1]).astype('uint8')
    plt.imshow(label_image)
    plt.show()

    print(i)
