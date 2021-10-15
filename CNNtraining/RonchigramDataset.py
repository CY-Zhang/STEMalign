from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class RonchigramDataset(Dataset):
    def __init__(self, data, labels, transform = None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.scale_range(self.data[index,:,:].astype('float'), 0, 1)
        new_channel = np.zeros(img.shape)
        img = np.dstack((img, new_channel, new_channel))
        img = Image.fromarray(np.uint8(img * 255))
        y_label = torch.tensor(float(self.labels[index]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)
    
    def scale_range (self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input    

    def aperture_generator(self, px_size, simdim, ap_size):
        x = np.linspace(-simdim, simdim, px_size)
        y = np.linspace(-simdim, simdim, px_size)
        xv, yv = np.meshgrid(x, y)
        apt_mask = np.sqrt(xv*xv + yv*yv) < ap_size # aperture mask
        return apt_mask