import glob
import re
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datetime import timedelta
from sklearn.model_selection import train_test_split
import warnings
import argparse
warnings.resetwarnings()
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def time_range(start, stop, step):
    current = start
    while current < stop:
        yield current
        current += step


class SaveFile:
    def __init__(self, datafiles, output_dir, ratio=0.1, save_per_file=True, random_state=0, is_SHL=False):
        self.spf = save_per_file
        self.output_dir = output_dir
        self.ratio = ratio
        self.random_state = random_state
        self.is_SHL = is_SHL
        if self.spf:
            file_idx = np.arange(len(datafiles))
            self.train_idxs, self.test_idxs = train_test_split(
                file_idx,
                test_size=int(len(file_idx)*ratio),
                shuffle=True,
                random_state=self.random_state,
            ) if len(datafiles) > 1 and ratio > 0.0 else [file_idx, 0]
            
            os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        else:
            self.windows = []
            self.labels = []
    def __call__(self, idx, datafile, signals, labels=None):
        if self.spf:
            signals = np.transpose(signals, (0, 2, 1))
            data = np.array([(signal, label) for signal, label in zip(signals, labels)], dtype=object) if labels is not None else signals
            
            if self.is_SHL:
                split_list = datafile.split('/')
                filename_last = split_list[-2]+'_'+split_list[-1].split('.txt')[0]
            else:
                filename_last = datafile.split('/')[-1].split('.csv')[0]
            filename = os.path.join(
                self.output_dir,
                'train' if idx in self.train_idxs else 'test',
                filename_last,
            )
            np.save(filename, data)
            print(f'Saved "{filename}".')
        else:
            self.windows.append(signals)
            if labels is not None:
                self.labels.append(labels)
    def save_concat_data(self, ):
        windows = np.concatenate(self.windows, axis=0).transpose((0, 2, 1))
        if len(self.labels) > 0:
            labels = np.concatenate(self.labels, axis=0)
            data = np.array([(window, label) for window, label in zip(windows, labels)])
        else:
            data = windows
        
        if self.ratio > 0.0:
            train_data, test_data = train_test_split(
                data,
                test_size=int(len(data)*self.ratio),
                shuffle=True,
                random_state=self.random_state,
            )
            np.save(os.path.join(self.output_dir, 'train.npy'), train_data)
            np.save(os.path.join(self.output_dir, 'test.npy'), test_data)
        else:
            np.save(os.path.join(self.output_dir, 'train.npy'), data)


##
## main process
##
class Process_:
    def __init__(self, args):
        self.args = args
        self.save_file = SaveFile(args.datafiles, args.output_dir, ratio=args.test_ratio, save_per_file=args.save_per_file)
        self.splits = None
        self.windows = []
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def __call__(self):
        for i, datafile in tqdm(enumerate(self.args.datafiles)):
            self.data_load(datafile)
            self.split_window()
            self.save_file(i, datafile, self.windows)
        
        if not self.args.save_per_file:
            self.save_file.save_concat_data()
    
    def data_load(self, datafile):
        raise NotImplementedError()
    def split_window(self):
        resolution_fn = self.low_resolution if self.args.HZ >= self.args.hz else self.high_resolution
        self.windows = []
        for split in tqdm(self.splits):
            index, value = np.split(split, [1], axis=-1)
            for start in time_range(0, len(index)-1, step=self.args.window_step_len):
                window = resolution_fn(value, start)
                self.windows.append(window)
            self.remove_lastwindow()
        self.windows = np.stack(self.windows, axis=0).astype(np.float16)
    def low_resolution(self, value, start):
        value_ = value[start:start+self.args.window_len]
        steped_idx = np.round(np.arange(0, len(value_), self.args.HZ/self.args.hz)).astype(np.int32)
        if steped_idx[-1] >= len(value_):
            steped_idx[-1] = -1
        return value_[steped_idx]
    def high_resolution(self, value, start):
        ratio = self.args.hz/self.args.HZ
        d, i = math.modf(ratio)
        window = value[start:start+self.args.window_len]
        idx = np.sort(np.concatenate([
            np.concatenate([np.arange(len(window)) for _ in range(int(i))]),
            np.array(random.sample(range(len(window)), int(len(window)*ratio)-(len(window)*int(i))) if d > 0. else []),
        ])).astype(int)
        return window[idx]
    def remove_lastwindow(self):
        if self.windows[-1].shape[0] != int(self.args.hz*self.args.window_sec):
            self.windows = self.windows[:-1]
            self.remove_lastwindow()


class ProcessSHL(Process_): # SHL
    def __init__(self, args):
        super().__init__(args)
        self.save_file.is_SHL = True
    def data_load(self, datafile, eps=1e-4):
        df = np.loadtxt(datafile)
        df[:, 0] *= 1e-3 # ms → s
        time_epoch = df[:, 0]
        skip_indexes = [i for i in range(len(time_epoch)-1) if time_epoch[i+1]-time_epoch[i]>(0.01+eps)]
        self.splits = np.split(df, skip_indexes)


class ProcessCapture24(Process_): # capture24
    def __init__(self, args):
        super().__init__(args)
    def data_load(self, datafile, eps=1e-4):
        self.windows = []
        df = pd.read_csv(datafile, low_memory=False)
        df['time'] = pd.to_datetime(df['time']).map(pd.Timestamp.timestamp)
        df = df.astype({'x': 'float16', 'y': 'float16', 'z': 'float16'})
        df = df[['time', 'x', 'y', 'z']].values
        time_epoch = df[:, 0]
        skip_indexes = [i for i in range(len(time_epoch)-1) if time_epoch[i+1]-time_epoch[i]>(0.01+eps)]
        self.splits = np.split(df, skip_indexes)


class ProcessOurs(Process_): # ours data
    def __init__(self, args):
        super().__init__(args)
    def data_load(self, datafile):
        df = pd.read_csv(datafile, index_col='time', low_memory=False)
        rm = 'hh.mm.ss.csv.bz2'
        df.index = pd.to_datetime(
            datafile.split('/')[-1][:-len(rm)]+df.index, format='%Y-%m-%d%H:%M:%S:%f'
        ).map(pd.Timestamp.timestamp)
        df = df.astype({'ax': 'float16', 'ay': 'float16', 'az': 'float16'})
        df = df.drop_duplicates()
        df = df.drop(df.index[np.where(df.index.duplicated())[0]])
        self.index = df.index.values
        self.value = df[['ax', 'ay', 'az']].values



def convert_files(datafiles):
    # case for datafiles = ['/home/.../P01*.csv', '/home/.../P02*.csv', ..]
    if isinstance(datafiles, list):
        files = []
        for datafile in datafiles:
            if '*' in datafile:
                files.extend(glob.glob(datafile))
            else:
                files.append(datafile)
        return files
    else:
        return glob.glob(datafiles)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hz', type=int, default=50)
    parser.add_argument('--window-sec', type=float, default=6.0)
    parser.add_argument('--overlap-sec', type=float, default=0.0)
    parser.add_argument('--datafiles', type=str, default='./capture24/P*.csv.gz')
    parser.add_argument('--output-dir', type=str, default='./signal_data/experiment_data')
    parser.add_argument('--save-per-file', type=bool_flag, default=True)
    parser.add_argument('--test-ratio', type=float, default=0.0)
    parser.add_argument('--dataset', default=None, choices=['SHL', 'capture24', 'ours'])
    parser.set_defaults(
        hz=30,
        window_sec=10.0,
        overlap_sec=0.0,
        # datafiles='/home/ukita/human-activity-recognition/signal_data/SHL/train/*/Acc.txt',
        # dataset='SHL',
        # datafiles='/home/ukita/human-activity-recognition/signal_data/capture24/P*.csv.gz',
        # dataset='capture24',
    )
    args = parser.parse_args([])
    
    HZs = {'SHL': 100, 'capture24': 100, 'ours': 333} # 333?
    args.HZ = HZs[args.dataset]
    args.window_len = int(args.HZ*args.window_sec)
    args.overlap_len = int(args.HZ*args.overlap_sec)
    args.window_step_sec = args.window_sec - args.overlap_sec
    args.window_step_len = args.window_len - args.overlap_len
    args.datafiles = convert_files(args.datafiles)
    return args


def main(args):
    # output dir: '/<output dir>/50hz_10.0s_overlap0.0s/<dataset name>/'
    args.output_dir = os.path.join(args.output_dir, str(args.hz)+'hz_'+str(args.window_sec)+'s_overlap'+str(args.overlap_sec)+'s', args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # dataset 毎の処理
    if args.dataset == 'SHL':
        process_for_SHL = ProcessSHL(args)
        process_for_SHL()
    elif args.dataset == 'capture24':
        process_for_capture24 = ProcessCapture24(args)
        process_for_capture24()
    elif args.dataset == 'ours':
        process_for_ours = ProcessOurs(args)
        process_for_ours()
    else:
        print('Error: dataset name is wrong')

if __name__ == '__main__':
    args = get_args()
    main(args)