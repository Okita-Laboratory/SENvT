import os
import glob
import time
import sys
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sslearning.data import SSL_dataset, worker_init_fn
from sslearning.multitask import SignalAugmentation, MultiTaskLoss
from sslearning.utils import Logger, bool_flag, plot_fig
from sslearning.pytorchtools import EarlyStopping
import sslearning.augmentation as aug
import sslearning.models.senvt as SENvT


def evaluate_model(model, data_loader, loss_fn=nn.MSELoss(), mask_nc=0, device='cpu'):
    model.eval()
    d = list(model.state_dict().values())[0].dtype
    ys, preds = [], []
    running_loss = 0.

    for i, batch in enumerate(data_loader):
        data = batch['origin']
        x_aug = batch['aug']
        task_pos = batch['pos']
        r_true = batch['rotation'] if args.rotation else None

        x, y = data if isinstance(data, list) else [data, None]
        ground_x = Variable(x).to(device, dtype=d)
        x_aug = Variable(x_aug).to(device, dtype=d)
        mask = (task_pos == mask_nc).to(device, dtype=d)
        
        with torch.no_grad():
            pred_x, pred_rotation = model(x_aug, mask=mask)
            r_args = dict(r_pred=pred_rotation, r_true=r_true, task_pos=task_pos)
            loss, loss_dict = loss_fn(pred_x, ground_x, **r_args)
        
        running_loss += loss.item()
    return running_loss/i



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################
    #   Setting macros
    ####################
    task_num = int(args.mask+args.permute+args.timewarp+args.rotation+args.wdba+args.rgw+args.dgw)
    task_dict = dict(
        mask=args.mask, permute=args.permute, timewarp=args.timewarp, rotation=args.rotation,
        wdba=args.wdba, rgw=args.rgw, dgw=args.dgw,
    )
    os.makedirs(args.log_path, exist_ok=True)
    log_index = len(glob.glob(f"{args.log_path}/*"))
    log_dir = os.path.join(args.log_path, f'{log_index:03d}')
    
    model_path = os.path.join(log_dir, 'best.pt')
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, 'output.log'))
    
    print('\n## Configuration: ')
    for k, v in vars(args).items():
        print(k, v)
    print()
    
    
    ####################
    #   Set up data
    ####################
    augment = SignalAugmentation(
        task_dict,
        task_num=task_num,
        window_size=args.window_size,
        delta=args.augment_chunk_size,
    )
    
    train_dataset = SSL_dataset(glob.glob(os.path.join(args.data_path, 'train/*.npy')), transform_aug=augment, label=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
    )
    test_dataset = SSL_dataset(glob.glob(os.path.join(args.data_path, 'test/*.npy')), transform_aug=augment, label=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
    )
    print(f"Data loaded:\ntraining data: {len(train_dataset)} windows, test data: {len(test_dataset)} windows.")


    ####################
    #   Model const
    ####################
    model = SENvT.__dict__[args.model_size](
        window_size=args.window_size,
        patch_size=args.patch_size,
        is_rotation_task=args.rotation,
        augment_chunk_size=args.augment_chunk_size,
    ).to(dtype=args.dtype)

    
    ####################
    #   Set up Training
    ####################
    loss_fn = MultiTaskLoss(task_dict=task_dict, task_num=augment.nc)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    total_step = len(train_loader)

    print('Start training ...')
    early_stopping = EarlyStopping(patience=args.patience, path=model_path, verbose=True)

    model.to(device)
    train_losses, valid_losses = [], []
    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0.

        for i, batch in enumerate(train_loader):
            data = batch['origin']
            x_aug = batch['aug']
            task_pos = batch['pos']
            r_true = batch['rotation'] if args.rotation else None
            
            x, y = data if isinstance(data, list) else [data, None]
            ground_x = Variable(x).to(device, dtype=args.dtype)
            x_aug = Variable(x_aug).to(device, dtype=args.dtype)
            mask = (task_pos == augment.nc['mask']).to(device, dtype=args.dtype)
            
            pred_x, pred_rotation = model(x_aug, mask=mask)
            r_args = dict(r_pred=pred_rotation, r_true=r_true, task_pos=task_pos)
            loss, loss_dict = loss_fn(pred_x, ground_x, **r_args)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % args.log_interval == 0:
                msg = (
                    f'Train Epoch [{epoch}/{args.num_epoch}], Step [{i}/{total_step}], Loss: {loss.item():.8f} ['
                   +f'mask: {loss_dict["mask"]:.3f}, '
                   +f'permute: {loss_dict["permute"]:.3f}, '
                   +f'timewarp: {loss_dict["timewarp"]:.3f}, '
                   +f'rotation: {loss_dict["rotation"]:.3f}, '
                   +f'wdba: {loss_dict["wdba"]:.3f}, '
                   +f'rgw: {loss_dict["rgw"]:.3f}, '
                   +f'dgw: {loss_dict["dgw"]:.3f}]'
                )
                print(msg)

        test_loss = evaluate_model(
            model, test_loader,
            loss_fn=MultiTaskLoss(task_dict=task_dict, task_num=augment.nc),
            mask_nc=augment.nc['mask'], device=device
        )
        train_losses.append(running_loss/(i+1))
        valid_losses.append(test_loss)
        print('train loss:', train_losses[-1], 'valid loss:', valid_losses[-1])

        if epoch % 10 == 0:
            torch.save({'model': model.state_dict(), 'args': args}, os.path.join(log_dir, 'models', f'{epoch}.pt'))
        early_stopping(test_loss, {'model': model.state_dict(), 'args': args})

        if early_stopping.early_stop:
            print("Early stopping")
            break

    losses = {'train_losses': train_losses, 'valid_losses': valid_losses}
    np.save(os.path.join(log_dir, 'loss.npy'), losses)
    plt = plot_fig(losses)
    plt.savefig(os.path.join(log_dir, 'loss.png'))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # runtime #
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--log-path', type=str, default='./experiment_log/pre_train')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    # data #
    parser.add_argument('--data-path', type=str, default='<dataset path>')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--augment-chunk-size', type=int, default=50)
    parser.add_argument('--window-num', type=int, default=-1)
    # model #
    parser.add_argument('--window-size', type=int, default=300)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--patch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model-size', type=str, default='S', choices=['XS', 'S', 'B', 'L', 'XL'])
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float)
    # task #
    parser.add_argument('--mask', type=bool_flag, default=True)
    parser.add_argument('--permute', type=bool_flag, default=False)
    parser.add_argument('--timewarp', type=bool_flag, default=False)
    parser.add_argument('--rotation', type=bool_flag, default=False)
    parser.add_argument('--wdba', type=bool_flag, default=False)
    parser.add_argument('--rgw', type=bool_flag, default=False)
    parser.add_argument('--dgw', type=bool_flag, default=False)
    
    parser.set_defaults()
    args = parser.parse_args()
    main(args)