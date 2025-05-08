import sys
import numpy as np
import torch
from torch import nn
from scipy.stats import norm
from .augmentation import permutation_torch, time_warp_torch, rotation_torch


def set_task_label(task_dict):
    task_label = {k: -1 for k in task_dict.keys()}
    c = 0
    for task in task_label.keys():
        if task_dict[task]:
            task_label[task] = c
            c+=1
    return task_label


class SignalAugmentation:
    def __init__(self, task_dict, window_size=300, chunk=50):
        self.window_size = window_size
        self.chunk = chunk
        self.num_chunks = window_size//chunk
        self.task_dict = task_dict
        self.n_tasks = sum(task_dict.values())
        self.task_label = set_task_label(task_dict)
        
        # wdba, rgw, dgwは未実装のため，エラーを出す
        if self.task_dict['wdba'] or self.task_dict['rgw'] or self.task_dict['dgw']:
            assert False, 'Error: wdba, rgw, dgw are not implemented yet.'

    def __call__(self, x):
        x = x.clone()

        B, C, L = x.shape
        x = x.permute(0, 2, 1) # BCL → BLC
        # タスク割り当てのリストを作成(チャンク数分のランダムな整数配列) 例: [-1, 0, 1, 1, -1, 2, 0]
        task_split = torch.randint(-1, self.n_tasks, (B, self.num_chunks,), device=x.device)
        # サイズを実際のデータに合わせる 例: (1, 6) -> (1, 6*50)
        task_split_expand = task_split.repeat_interleave(self.chunk, dim=1)
        
        if self.task_dict['mask']:
            x = self.aug_adaption(x, mask=(task_split_expand == self.task_label['mask']), C=C, fn=lambda x: x)
        if self.task_dict['permute']:
            x = self.aug_adaption(x, mask=(task_split_expand == self.task_label['permute']), C=C, fn=permutation_torch)
        if self.task_dict['timewarp']:
            x = self.aug_adaption(x, mask=(task_split_expand == self.task_label['timewarp']), C=C, fn=time_warp_torch)
        if self.task_dict['rotation']:
            x, r_truth = self.aug_adaption(x, mask=(task_split_expand == self.task_label['rotation']), C=C, fn=rotation_torch)
        
        x = x.permute(0, 2, 1) # BWC → BCW
        return (
            x,
            task_split_expand,
            r_truth if self.task_dict['rotation'] else None
        )
    
    def aug_adaption(self, x, mask, C, fn):
        target = x[mask].reshape(-1, self.chunk, C)
        aug = fn(target)
        if isinstance(aug, tuple):
            aug, rotation_truth = aug
            x[mask] = aug.reshape(-1, C)
            return x, rotation_truth
        
        x[mask] = aug.reshape(-1, C)
        return x




class MultiTaskLoss(nn.Module):
    def __init__(self, task_dict):
        super().__init__()
        self.task_dict = task_dict
        self.task_label = set_task_label(task_dict)
        self.loss_dict = {'mask': 0., 'permute': 0., 'timewarp': 0., 'rotation': 0.}
        self.scale = 1.
    
    def prepare(self, x):
        return x

    def forward(self, prediction, ground_truth, task_pos, rotation_truth=None):
        # prediction: (b, w, c, v)
        # ground_truth: (b, c, w)
        
        gt = self.prepare(ground_truth).transpose(1, 2)
        loss = 0.
        for task in self.loss_dict.keys():
            if self.task_dict[task]:
                if task == 'rotation':
                    self.loss_dict[task] = self.calc_rotation_loss(prediction, rotation_truth, mask=(task_pos == self.task_label[task]))
                else:
                    self.loss_dict[task] = self.calc_loss(prediction, gt, mask=(task_pos == self.task_label[task])) * self.scale
            
            loss += self.loss_dict[task]
        return loss, self.loss_dict

    def calc_loss(self):
        pass

    def calc_rotation_loss(self, p, g, mask):
        p = p[mask]
        N, C, _ = p.shape
        n, _ = g.shape
        g = g.repeat_interleave(N//n, dim=0)
        return nn.functional.cross_entropy(p[:,:,:C].view(-1, C), g.view(-1))



class MultiTaskLossforCE(MultiTaskLoss):
    def __init__(self, task_dict, vocabulary=256):
        super().__init__(task_dict)
        self.loss_fn = nn.CrossEntropyLoss()
        self.vocabulary = vocabulary
        self.scale = 0.2

        # 標準正規分布(μ=0., σ=1.)に従ったvocabulary−1個の理論分位数を作成
        q = torch.linspace(0, 1, vocabulary+1)
        quantile_edges = torch.from_numpy(norm.ppf(q, loc=0., scale=1.)).to(dtype=torch.float)
        # Remove -inf and inf for bucketize, use only finite edges
        quantile_edges[0] = quantile_edges[1]
        quantile_edges[-1] = quantile_edges[-2]
        self.quantize = lambda x: torch.bucketize(x, quantile_edges[1:-1].to(device=x.device, dtype=x.dtype))

        quantile_midpoints = (quantile_edges[:-1] + quantile_edges[1:]) / 2
        self.dequantize = lambda x: quantile_midpoints.to(device=x.device)[x]

    def prepare(self, x):
        return self.quantize(x)
        
    def forward(self, prediction, ground_truth, task_pos, rotation_truth=None):
        return super().forward(prediction, ground_truth, task_pos, rotation_truth=rotation_truth)
    
    def calc_loss(self, p, g, mask):
        return self.loss_fn(p[mask].view(-1, self.vocabulary), g[mask].view(-1))



class MultiTaskLossforMSE(MultiTaskLoss):
    def __init__(self, task_dict):
        super().__init__(task_dict)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, prediction, ground_truth, task_pos, rotation_truth=None):
        return super().forward(prediction, ground_truth, task_pos, rotation_truth=rotation_truth)
    
    def calc_loss(self, p, g, mask):
        # prediction: (b, w, c, v)
        # ground_truth: (b, w, c)
        p = p[mask] # n, c, v
        g = g[mask] # n, c
        return self.loss_fn(p[:,:,0], g)









# class SignalAugmentation:
#     def __init__(self, task, task_num, window_size=300, chunk=50):
#         self.window_size = window_size
#         self.chunk = chunk
#         self.num_chunks = window_size//chunk
#         self.task = task
#         self.task_num = task_num
#         self.mask = None
#         self.rotation_true = None
#         self.nc = {'mask': -1, 'permute': -1, 'timewarp': -1, 'rotation': -1, 'wdba': -1, 'rgw': -1, 'dgw': -1}
#         c = 0
#         for task in self.nc.keys():
#             if self.task[task]:
#                 self.nc[task] = c
#                 c+=1
        
#         assert c == task_num, 'Error: The number of boolean values of task and task_num does not match.'

#     def __call__(self, x, y=None):
#         B, C, W = x.shape
#         x = x.transpose(0, 2, 1) # BCW → BWC
#         start = -1 if self.task_num == 1 and self.task['mask'] else 0
#         r_split = np.random.randint(start, self.task_num, (B, self.num_chunks,))
#         r = np.repeat(r_split, self.chunk, axis=1)
        
#         c = 0
#         if self.task['mask']:
#             self.mask = (r == c)
#             s0, s1 = x[self.mask].shape
#             # x[self.mask] = np.random.randn(s0, s1)
#             x[self.mask] = np.zeros((s0, s1)) # こっちの方が安定する．
#             self.nc['mask'] = c
#             c+=1
#         if self.task['permute']:
#             permute_mask = (r == c)
#             target = x[permute_mask].reshape(-1, self.chunk, C)
#             target_aug = aug.permutation(target, max_segments=5)
#             x[permute_mask] = target_aug.reshape(-1, C)
#             self.nc['permute'] = c
#             c+=1
#         if self.task['timewarp']:
#             warp_mask = (r == c)
#             target = x[warp_mask].reshape(-1, self.chunk, C)
#             target_aug = aug.time_warp(target, sigma=0.2, knot=4)
#             x[warp_mask] = target_aug.reshape(-1, C)
#             self.nc['timewarp'] = c
#             c+=1
#         if self.task['rotation']:
#             rotate_mask = (r == c)
#             target = x[rotate_mask].reshape(-1, self.chunk, C)
#             target_aug, true_channel = aug.rotation(target)
#             x[rotate_mask] = target_aug.reshape(-1, C)
#             self.rotation_true = true_channel
#             self.nc['rotation'] = c
#             c+=1
#         if self.task['wdba']:
#             wdba_mask = (r == c)
#             target = x[wdba_mask].reshape(-1, self.chunk, C)
#             self.nc['wdba'] = c
#             c+=1
#         if self.task['rgw']:
#             rgw_mask = (r == c)
#             target = x[rgw_mask].reshape(-1, self.chunk, C)
#             self.nc['rgw'] = c
#             c+=1
#         if self.task['dgw']:
#             dgw_mask = (r == c)
#             target = x[dgw_mask].reshape(-1, self.chunk, C)
#             self.nc['dgw'] = c
#             c+=1
        
#         x = x.transpose(0, 2, 1) # BWC → BCW
#         return x, (r, self.nc, self.rotation_true)


# class MultiTaskLoss:
#     def __init__(self, task_dict, task_num):
#         self.task_dict = task_dict
#         self.task_num = task_num
#         self.loss_dict = {'mask': 0., 'permute': 0., 'timewarp': 0., 'rotation': 0., 'wdba': 0., 'rgw': 0., 'dgw': 0.}
#         self.rotation_loss_fn = nn.CrossEntropyLoss()
#         self.rotation_loss_ratio = 0.333
        
#     def __call__(self, pred, ground, r_pred=None, r_true=None, task_pos=None):
#         squared_error = (pred - ground)**2
#         se = squared_error.transpose(1,2)
        
#         loss = 0.
#         for task in self.loss_dict.keys():
#             if self.task_dict[task]:
#                 if task=='rotation':
#                     r_mask = (task_pos==self.task_num['rotation'])
#                     self.loss_dict[task] = self.calc_rotation_loss(r_pred, r_true, mask=r_mask)
#                 else:
#                     self.loss_dict[task] = torch.mean(se[task_pos==self.task_num[task]])
#             else:
#                 self.loss_dict[task] = 0.
#             loss += self.loss_dict[task]
#         return loss, self.loss_dict
    
#     def calc_rotation_loss(self, pred, true, mask=None):
#         (N, nsplits, C,_), (_, L) = pred.shape, mask.shape
#         delta = L // nsplits
        
#         mask = mask.reshape(N, -1, delta).to(dtype=torch.float16).mean(dim=-1).to(dtype=torch.bool)
#         true_idx, _ = torch.where(mask)
        
#         pred = pred[mask]
#         true = true[true_idx].to(pred.device)
        
#         loss = self.rotation_loss_fn(pred.reshape(-1, C), true.reshape(-1))
#         return loss * self.rotation_loss_ratio
