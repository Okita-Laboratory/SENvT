import os
import numpy as np
import pickle
import torch
import time
import glob
import torch.nn.functional as F

def worker_init_fn(worker_id):
    np.random.seed(int(time.time()))


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class SSL_dataset:
    def __init__(self, data_path_list, transform=None, transform_aug=None, label=False):
        self.transform = transform
        self.transform_aug = transform_aug
        self.label = label
        self.data = np.concatenate([np.load(path, allow_pickle=True).astype(np.float16) for path in data_path_list], axis=0)
        self.task_num = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        window = self.data[idx]
        if len(window.shape) == 2:
            x = window
            y = None
        elif len(window.shape) == 1:
            x, y = window
        
        if self.transform is not None:
            x = self.transform(np.expand_dims(x, 0))
            if isinstance(x, list):
                x = [xi.squeeze(0) for xi in x]
            else:
                x = x.squeeze(0)
        
        if self.transform_aug is not None:
            x_aug, task_variabel = self.transform_aug(np.expand_dims(x, 0))
            (task_pos, task_num, r_true) = task_variabel
            self.task_num = task_num
            if isinstance(x, list):
                x_aug = [xi.squeeze(0) for xi in x_aug]
            else:
                x_aug = x_aug.squeeze(0)
            
            s = (x, y) if self.label and y is not None else x
            batch = dict(origin=s, aug=x_aug, pos=task_pos.squeeze(0))
            if r_true is not None:
                batch['rotation'] = r_true
                return batch
            else:
                return batch
        
        if self.label and y is not None:
            return (x, y)
        else:
            return x


class NormalDataset:
    def __init__(
        self,
        X,
        y=[],
        pid=[],
        name="",
        isLabel=True,
        transform=None,
        target_transform=None,
    ):
        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        self.pid = pid
        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = []
        if self.isLabel:
            y = self.y[idx]
            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        if len(self.pid) >= 1:
            return sample, y, self.pid[idx]
        else:
            return sample, y

