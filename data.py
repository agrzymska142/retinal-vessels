import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class ChasedDataset(Dataset):
    def __init__(self, images_path, answers_path):
        self.images_path = images_path
        self.answers_path = answers_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        img = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        ans = cv2.imread(self.answers_path[index], cv2.IMREAD_GRAYSCALE)
        ans = cv2.resize(ans, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        ans = ans / 255.0
        ans = ans[np.newaxis, ...]
        ans = torch.from_numpy(ans.astype(np.float32))

        return img, ans

    def __len__(self):
        return self.n_samples