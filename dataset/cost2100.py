import os
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader']


class Cost2100DataLoader:

    def __init__(self, root, batch_size, num_workers, scenario):
        self.batch_size = batch_size
        self.num_workers = num_workers

        dir_train = os.path.join(root, f"DATA_Htrain{scenario}.mat")
        dir_val = os.path.join(root, f"DATA_Hval{scenario}.mat")
        dir_test = os.path.join(root, f"DATA_Htest{scenario}.mat")
        channel, nt, nc = 2, 32, 32

        # Training data loading
        data_train = sio.loadmat(dir_train)['HT']
        data_train = torch.tensor(data_train, dtype=torch.float32).view(data_train.shape[0], channel, nt, nc)
        self.train_dataset = TensorDataset(data_train)

        # Validation data loading
        data_val = sio.loadmat(dir_val)['HT']
        data_val = torch.tensor(data_val, dtype=torch.float32).view(data_val.shape[0], channel, nt, nc)
        self.val_dataset = TensorDataset(data_val)

        # Test data loading, including the sparse data and the raw data
        data_test = sio.loadmat(dir_test)['HT']
        data_test = torch.tensor(data_test, dtype=torch.float32).view(data_test.shape[0], channel, nt, nc)
        self.test_dataset = TensorDataset(data_test)
        return_H = torch.tensor(data_test, dtype=torch.float32).contiguous().view(data_test.shape[0], channel, nt, nc)
        self.inputimage = return_H

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False)
        inputimage = self.inputimage

        return train_loader, val_loader, test_loader, inputimage