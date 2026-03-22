import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
import scipy.io as sio

DATASET_SCALE_FACTOR = 0.1

class HADTestDataset(Dataset):
    def __init__(self,
                 dataset_path='./',
                 ):
        self.dataset_path = dataset_path
        # load dataset
        self.test_img = self.load_dataset_folder()

        # set transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, idx):
        img_path = self.test_img[idx]
        # load test image
        mat = sio.loadmat(img_path)
        x = mat["data"].astype(np.float32)

        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        x = x * DATASET_SCALE_FACTOR
        x = self.transform(x)
        x = x.type(torch.FloatTensor)

        # load gt
        gt = np.asarray(mat["map"]).astype(bool)
        gt = Image.fromarray(gt)
        gt = self.transform(gt)
        return x,gt,os.path.splitext(os.path.basename(img_path))[0]

    def __len__(self):
        return len(self.test_img)

    def load_dataset_folder(self):
        test_list = []
        for fname in sorted(os.listdir(self.dataset_path)):
            if fname.endswith(".mat"):
                test_list.append(os.path.join(self.dataset_path, fname))
        return test_list
