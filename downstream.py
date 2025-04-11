import os
import sys
import copy
import glob
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score
)
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

import sslearning.models.senvt as SENvT
from sslearning.data import NormalDataset, worker_init_fn
from sslearning.utils import bool_flag, Logger, get_class_weights


def train_model(model, train_loader, valid_loader, epochs, lr, wd, w, dtype=torch.float, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=dtype).to(device))

    train_losses, train_acces, train_f1s = [], [], []
    valid_losses, valid_acces, valid_f1s = [], [], []
    best_acc, best_f1 = 0., 0.

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, running_f1 = [], [], []
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x).to(device, dtype=dtype)
            y = Variable(y).to(device, dtype=torch.uint8)
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred_y = torch.argmax(pred, dim=1)
            train_acc = accuracy_score(y.cpu(), pred_y.cpu())
            train_f1 = f1_score(y.cpu(), pred_y.cpu(), average='macro')

            running_loss.append(loss.item())
            running_acc.append(train_acc)
            running_f1.append(train_f1)
        
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, valid_loader, loss_fn=nn.CrossEntropyLoss(), dtype=dtype, device=device)
        
        train_losses.append(np.mean(running_loss))
        train_acces.append(np.mean(running_acc))
        train_f1s.append(np.mean(running_f1))
        valid_losses.append(val_loss)
        valid_acces.append(val_acc)
        valid_f1s.append(val_f1)
        
        if best_f1 <= val_f1:
            best_f1 = val_f1
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: f-score'
            if best_acc <= val_acc:
                best_acc = val_acc
                update_msg += ' & accuracy'
        elif best_acc <= val_acc:
            best_acc = val_acc
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: accuracy'
        else:
            update_msg = ''
        
        print(
            f'[{epoch:>{len(str(epochs))}}/{epochs:>{len(str(epochs))}}] '
            + f'Loss: [train: {np.mean(running_loss):.4f}, valid: {val_loss:.4f}] '
            + f'Accuracy: [train: {np.mean(running_acc):.4f}, valid: {val_acc:.4f}] '
            + f'F-score: [train: {np.mean(running_f1):.4f}, valid: {val_f1:.4f}] '
            + update_msg
        )
    log = {
        'loss': {'train': train_losses, 'valid': valid_losses},
        'acc' : {'train': train_acces, 'valid': valid_acces}
    }
    return log, best


def evaluate_model(model, loader, loss_fn, dtype=torch.float, device='cpu'):
    model.eval()
    losses, trues, preds = [], [], []
    for i, (x, y) in enumerate(loader):
        x = Variable(x).to(device, dtype=dtype)
        y = Variable(y).to(device, dtype=torch.uint8)
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y)
            pred_y = torch.argmax(pred, dim=1)
        
        losses.append(loss.item())
        trues.append(y.detach().cpu())
        preds.append(pred_y.detach().cpu())
    trues = torch.cat(trues)
    preds = torch.cat(preds)
    return (
        np.mean(np.array(losses)),
        accuracy_score(trues, preds),
        f1_score(trues, preds, average='macro'),
        trues, preds,
    )


def plot_fig(log):
    fig = plt.figure(figsize=(16,4))

    fig.add_subplot(1,2,1)
    plt.plot(log['loss']['train'], label='train')
    plt.plot(log['loss']['valid'], label='valid')
    plt.title('loss')
    plt.legend()

    fig.add_subplot(1,2,2)
    plt.plot(log['acc']['train'], label='train')
    plt.plot(log['acc']['valid'], label='valid')
    plt.title('accuracy')
    plt.legend()
    return plt


def test_evaluation(trues, preds, best_epoch):
    print(f'best epoch: {best_epoch}')
    acc = accuracy_score(trues, preds)
    rec = recall_score(trues, preds, average='macro')
    pre = precision_score(trues, preds, average='macro')
    f1s = f1_score(trues, preds, average='macro')
    print('accuracy :', acc)
    print('recall   :', rec)
    print('precision:', pre)
    print('f1       :', f1s)
    print(f'confusion matrix:\n {confusion_matrix(trues, preds)}')
    return acc, rec, pre, f1s


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.results_dir = os.path.join(
        args.results_dir, args.dataset,
        'fine-tuning' if args.finetuning else 'transfer_learning',
    )
    os.makedirs(args.results_dir, exist_ok=True)
    log_index = len(glob.glob(f"{args.results_dir}/*"))
    args.results_dir = os.path.join(args.results_dir, f'{log_index:03d}')

    os.makedirs(os.path.join(args.results_dir, 'ckpt_bests'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    
    sys.stdout = Logger(os.path.join(args.results_dir, 'output.log'))
    print('\n## Configuration: ')
    for k, v in vars(args).items():
        print(k, v)
    print()
    
    ##############
    # setup data #
    ##############
    x_train = np.load(os.path.join(args.data_path, args.dataset, 'x_train.npy'))
    x_valid = np.load(os.path.join(args.data_path, args.dataset, 'x_valid.npy'))
    x_test  = np.load(os.path.join(args.data_path, args.dataset, 'x_test.npy'))
    y_train = np.load(os.path.join(args.data_path, args.dataset, 'y_train.npy'))
    y_valid = np.load(os.path.join(args.data_path, args.dataset, 'y_valid.npy'))
    y_test  = np.load(os.path.join(args.data_path, args.dataset, 'y_test.npy'))
    w_train = get_class_weights(y_train)

    train_dataset = NormalDataset(x_train, y_train, name="train")
    valid_dataset = NormalDataset(x_valid, y_valid, name="val")
    test_dataset  = NormalDataset(x_test, y_test, name="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    ###############
    # setup model #
    ###############
    ckpt = torch.load(args.ckpt, weights_only=False)
    model = SENvT.__dict__[ckpt['args'].model_size](
        window_size=ckpt['args'].window_size,
        patch_size=ckpt['args'].patch_size,
        num_classes={'adl':5, 'oppo':4, 'pamap':8, 'realworld':8, 'wisdm':18}[args.dataset],
        is_eva=True,
    ).to(device, dtype=args.dtype)
    model.load_state_dict(ckpt['model'], strict=False)
    
    if args.finetuning:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for n, p in model.named_parameters():
            p.requires_grad = True if 'head' in n else False

    
    ###########
    # Running #
    ###########
    accs, f1s = [], []
    for i in range(args.num_repeat):
        # run #
        print(f'\n************** {i} **************')
        model_ = copy.deepcopy(model)
        log, best = train_model(model_, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, wd=args.wd, w=w_train, dtype=args.dtype, device=device)
        torch.save(best['model'].state_dict(), os.path.join(args.results_dir, 'ckpt_bests', f'{i:02d}.pt'))
        ##########
        # figure #
        ##########
        import matplotlib.pyplot as plt
        plt = plot_fig(log)
        plt.savefig(os.path.join(args.results_dir, 'figures', f'{i:02d}.pdf'))
        plt.close()
        #############
        # test eval #
        #############
        loss, acc, _, trues, preds = evaluate_model(best['model'], test_loader, loss_fn=nn.CrossEntropyLoss(), device=device, dtype=args.dtype)
        acc, rec, pre, f1 = test_evaluation(trues, preds, best['epoch'])

        accs.append(acc)
        f1s.append(f1)
    print('\n**************')
    print('    result    ')
    print('**************')
    print('acc: {:.4f}±{:.4f}, min({:.4f}), max({:.4f})'.format(np.mean(accs), np.std(accs), np.min(accs), np.max(accs)))
    print('f1 : {:.4f}±{:.4f}, min({:.4f}), max({:.4f})'.format(np.mean(f1s), np.std(f1s), np.min(f1s), np.max(f1s)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./experiment_log/downstream')
    parser.add_argument("--ckpt", type=str, default='<pre-trained model path>')
    parser.add_argument('--data-path', type=str, default='<downstream data path>')
    parser.add_argument('--dataset', type=str, default='adl', choices=['adl','oppo','pamap','realworld','wisdm'])
    parser.add_argument("--finetuning", type=bool_flag, default=True) # True: fine-tuning, False: transfer learning
    parser.add_argument('--num-repeat', type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float)

    parser.set_defaults()
    args = parser.parse_args()
    main(args)