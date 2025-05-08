import os
import sys
from glob import glob
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from sslearning.data import SSL_dataset, worker_init_fn
from sslearning.multitask import SignalAugmentation, MultiTaskLossforCE, MultiTaskLossforMSE, set_task_label
from sslearning.utils import Logger, bool_flag, plot_fig
from sslearning.pytorchtools import EarlyStopping
import sslearning.models.senvt as SENvT


def evaluate_model(model, loader, augmentation, loss_fn, dtype=torch.float, device='cpu'):
    model.eval()
    running_loss = 0.

    for i, x in enumerate(loader):
        x = x.to(device, dtype=dtype)
        x_aug, task_position, r_truth = augmentation(x)
        mask = (task_position == augmentation.task_label['mask'])

        with torch.no_grad():
            pred = model(x_aug, mask=mask)
            loss, loss_dict = loss_fn(pred, x, task_pos=task_position, rotation_truth=r_truth)
        
        running_loss += loss.item()
    return running_loss/i, loss_dict



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################
    #   Setting macros
    ####################
    os.makedirs(args.log_path, exist_ok=True)
    log_index = len(glob(f"{args.log_path}/*"))
    log_dir = os.path.join(args.log_path, f'{log_index:03d}')
    model_path = os.path.join(log_dir, 'best.pt')
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)

    with open(os.path.join(log_dir, 'config'), 'w') as f:
        f.write(''.join([f'{k}: {v}\n' for k, v in vars(args).items()]))
    sys.stdout = Logger(os.path.join(log_dir, 'output.log'))
    
    
    ####################
    #   Set up data
    ####################
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    train_dataset = SSL_dataset(glob(os.path.join(args.data_path, 'train/*.npy')), transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
    )
    test_dataset = SSL_dataset(glob(os.path.join(args.data_path, 'test/*.npy')), transform=transform)
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
        vocabulary=args.vocabulary, 
    ).to(dtype=args.dtype)

    
    ####################
    #  Set up Training
    ####################
    task_dict = dict(
        mask=args.mask, permute=args.permute, timewarp=args.timewarp, rotation=args.rotation,
        wdba=args.wdba, rgw=args.rgw, dgw=args.dgw,
    )
    task_label = set_task_label(task_dict)
    augmentation = SignalAugmentation(task_dict, window_size=args.window_size, chunk=args.augment_chunk_size)

    loss_fn = (
        MultiTaskLossforCE(task_dict=task_dict, vocabulary=args.vocabulary) if args.loss_type == 'CE' else
        MultiTaskLossforMSE(task_dict=task_dict) if args.loss_type == 'MSE' else None
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    total_step = len(train_loader)

    print('Start training ...')
    early_stopping = EarlyStopping(patience=args.patience, path=model_path, verbose=True)

    model.to(device)
    train_losses, valid_losses = [], []
    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0.

        for i, x in enumerate(train_loader):
            x = x.to(device, dtype=args.dtype)
            x_aug, task_position, rotation_truth = augmentation(x)
            mask = (task_position == task_label['mask'])
    
            pred = model(x_aug, mask=mask)
            loss, loss_dict = loss_fn(pred, x, task_pos=task_position, rotation_truth=rotation_truth)
            
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
                   +f'rotation: {loss_dict["rotation"]:.3f}]'
                )
                print(msg)

        valid_loss, valid_loss_dict = evaluate_model(model, test_loader, augmentation, loss_fn, dtype=args.dtype, device=device)

        train_losses.append(running_loss/(i+1))
        valid_losses.append(valid_loss)
        print('train loss:', train_losses[-1], 'valid loss:', valid_losses[-1])

        if epoch % 10 == 0:
            torch.save({'model': model.state_dict(), 'args': args}, os.path.join(log_dir, 'models', f'{epoch}.pt'))
        early_stopping(valid_loss, {'model': model.state_dict(), 'args': args})

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
    parser.add_argument('--num-epoch', type=int, default=500)
    parser.add_argument('--log-path', type=str, default='./experiment_log/pre_train')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--patience', type=int, default=100)
    # data #
    parser.add_argument('--data-path', type=str, default='<dataset path>')
    parser.add_argument('--batch-size', type=int, default=1024)         #
    parser.add_argument('--num-workers', type=int, default=4)           #
    parser.add_argument('--augment-chunk-size', type=int, default=50)   # [30, 50, 100]
    # model #
    parser.add_argument('--window-size', type=int, default=300)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--patch-size', type=int, default=10)           # [1, 2, 3, 4, 5, 10, 30, 50]
    parser.add_argument('--lr', type=float, default=1e-3)               #
    parser.add_argument('--wd', type=float, default=1e-4)               # [1e-5, 1e-4, 1e-3]
    parser.add_argument('--model-size', type=str, default='S', choices=['XS', 'S', 'B', 'L', 'XL']) #
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float)
    # loss #
    parser.add_argument('--loss-type', type=str, default='CE', choices=['CE', 'MSE']) # CrossEntropy or MeanSquareError
    parser.add_argument('--vocabulary', type=int, default=256)          # [16, 32, 64, 128, 256, 1024]
    # augmentation #
    parser.add_argument('--mask', type=bool_flag, default=True)         #
    parser.add_argument('--permute', type=bool_flag, default=False)     #
    parser.add_argument('--timewarp', type=bool_flag, default=False)    #
    parser.add_argument('--rotation', type=bool_flag, default=False)    #
    parser.add_argument('--wdba', type=bool_flag, default=False)
    parser.add_argument('--rgw', type=bool_flag, default=False)
    parser.add_argument('--dgw', type=bool_flag, default=False)
    
    parser.set_defaults()
    args = parser.parse_args()
    main(args)