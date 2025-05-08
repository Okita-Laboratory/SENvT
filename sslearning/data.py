import os
import random
import numpy as np
import pickle
import torch
import time
import glob
import torch.nn.functional as F

def worker_init_fn(worker_id):
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)



class SSL_dataset:
    def __init__(self, data_path_list, transform=None):
        self.transform = transform
        self.data = np.concatenate([np.load(path, allow_pickle=True).astype(np.float16) for path in data_path_list], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        x = self.data[idx]
        x = self.transform(x).squeeze(0) if self.transform is not None else torch.from_numpy(x)
        
        return x



class NormalDataset:
    def __init__(self, x, y=[], name="", transform=None):
        self.x = torch.from_numpy(x)
        self.y = y
        self.transform = transform
        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xi = self.x[idx]
        yi = self.y[idx]

        if self.transform:
            xi = self.transform(xi)
        
        return xi, yi











# class SSL_dataset:
#     def __init__(self, data_path_list, transform=None, augmentation=None):
#         self.transform = transform
#         self.augmentation = augmentation
#         self.data = np.concatenate([np.load(path, allow_pickle=True).astype(np.float16) for path in data_path_list], axis=0)
#         self.task_num = None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         idx = idx.tolist() if torch.is_tensor(idx) else idx
        
#         x = self.data[idx]
#         y = None
        
#         if self.transform is not None:
#             x = self.transform(x)
#             x = [xi.squeeze(0) for xi in x] if isinstance(x, list) else x.squeeze(0)
        
#         if self.augmentation is not None:
#             x_aug, task_variabel = self.augmentation(np.expand_dims(x, 0))
#             (task_pos, task_num, r_true) = task_variabel
#             self.task_num = task_num
#             if isinstance(x, list):
#                 x_aug = [xi.squeeze(0) for xi in x_aug]
#             else:
#                 x_aug = x_aug.squeeze(0)
            
#             s = (x, y) if self.label and y is not None else x
#             batch = dict(origin=s, aug=x_aug, pos=task_pos.squeeze(0))
#             if r_true is not None:
#                 batch['rotation'] = r_true
#                 return batch
#             else:
#                 return batch
        
#         if self.label and y is not None:
#             return (x, y)
#         else:
#             return x