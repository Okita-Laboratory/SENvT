import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sslearning.augmentation as aug

class SignalAugmentation(object):
    def __init__(self, task, task_num, window_size=300, delta=25):
        self.window_size = window_size
        self.delta = delta
        self.num_patches = window_size//delta
        self.task = task
        self.task_num = task_num
        self.mask = None
        self.rotation_true = None
        self.nc = {'mask': -1, 'permute': -1, 'timewarp': -1, 'rotation': -1, 'wdba': -1, 'rgw': -1, 'dgw': -1}
        c = 0
        for task in ['mask', 'permute', 'timewarp', 'rotation', 'wdba', 'rgw', 'dgw']:
            if self.task[task]:
                self.nc[task] = c
                c+=1
        assert c == task_num, 'Error: The number of boolean values of task and task_num does not match.'

    def __call__(self, x, y=None):
        B, C, W = x.shape
        x = x.transpose(0, 2, 1) # BCW → BWC
        start = -1 if self.task_num == 1 and self.task['mask'] else 0
        r_split = np.random.randint(start, self.task_num, (B, self.num_patches,))
        r = np.repeat(r_split, self.delta, axis=1)
        
        c = 0
        if self.task['mask']:
            self.mask = (r == c)
            s0, s1 = x[self.mask].shape
            # x[self.mask] = np.random.randn(s0, s1)
            x[self.mask] = np.zeros((s0, s1)) # こっちの方が安定する．
            self.nc['mask'] = c
            c+=1
        if self.task['permute']:
            permute_mask = (r == c)
            target = x[permute_mask].reshape(-1, self.delta, C)
            target_aug = aug.permutation(target, max_segments=5)
            x[permute_mask] = target_aug.reshape(-1, C)
            self.nc['permute'] = c
            c+=1
        if self.task['timewarp']:
            warp_mask = (r == c)
            target = x[warp_mask].reshape(-1, self.delta, C)
            target_aug = aug.time_warp(target, sigma=0.2, knot=4)
            x[warp_mask] = target_aug.reshape(-1, C)
            self.nc['timewarp'] = c
            c+=1
        if self.task['rotation']:
            rotate_mask = (r == c)
            target = x[rotate_mask].reshape(-1, self.delta, C)
            target_aug, true_channel = aug.rotation(target)
            x[rotate_mask] = target_aug.reshape(-1, C)
            self.rotation_true = true_channel
            self.nc['rotation'] = c
            c+=1
        if self.task['wdba']:
            wdba_mask = (r == c)
            target = x[wdba_mask].reshape(-1, self.delta, C)
            self.nc['wdba'] = c
            c+=1
        if self.task['rgw']:
            rgw_mask = (r == c)
            target = x[rgw_mask].reshape(-1, self.delta, C)
            self.nc['rgw'] = c
            c+=1
        if self.task['dgw']:
            dgw_mask = (r == c)
            target = x[dgw_mask].reshape(-1, self.delta, C)
            self.nc['dgw'] = c
            c+=1
        
        x = x.transpose(0, 2, 1) # BWC → BCW
        return x, (r, self.nc, self.rotation_true)


class MultiTaskLoss:
    def __init__(self, task_dict, task_num):
        self.task_dict = task_dict
        self.task_num = task_num
        self.loss_dict = {'mask': 0., 'permute': 0., 'timewarp': 0., 'rotation': 0., 'wdba': 0., 'rgw': 0., 'dgw': 0.}
        self.rotation_loss_fn = nn.CrossEntropyLoss()
        self.rotation_loss_ratio = 0.333
        
    def __call__(self, pred, ground, r_pred=None, r_true=None, task_pos=None):
        squared_error = (pred - ground)**2
        se = squared_error.transpose(1,2)
        
        loss = 0.
        for task in self.loss_dict.keys():
            if self.task_dict[task]:
                if task=='rotation':
                    r_mask = (task_pos==self.task_num['rotation'])
                    self.loss_dict[task] = self.calc_rotation_loss(r_pred, r_true, mask=r_mask)
                else:
                    self.loss_dict[task] = torch.mean(se[task_pos==self.task_num[task]])
            else:
                self.loss_dict[task] = 0.
            loss += self.loss_dict[task]
        return loss, self.loss_dict
    
    def calc_rotation_loss(self, pred, true, mask=None):
        (N, nsplits, C,_), (_, L) = pred.shape, mask.shape
        delta = L // nsplits
        
        mask = mask.reshape(N, -1, delta).to(dtype=torch.float16).mean(dim=-1).to(dtype=torch.bool)
        true_idx, _ = torch.where(mask)
        
        pred = pred[mask]
        true = true[true_idx].to(pred.device)
        
        loss = self.rotation_loss_fn(pred.reshape(-1, C), true.reshape(-1))
        return loss * self.rotation_loss_ratio
