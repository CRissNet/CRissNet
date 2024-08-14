import time
import os
import torch
from collections import namedtuple
import numpy as np

from utils import logger
from utils.statics import AverageMeter, evaluator

__all__ = ['Trainer', 'Tester']

field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field)


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', print_freq=30, val_freq=2, test_freq=2):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_nmse = 10

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None

    def loop(self, epochs, train_loader, val_loader, test_loader, path):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
            path (str): the path where model save
        """

        self.all_epoch = epochs
        loss_list1 = []
        loss_list2 = []
        final_nmse = 0

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)
            loss_list1.append(self.train_loss)
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)

            if ep % self.test_freq == 0:
                self.test_loss, nmse, best_nmse = self.test(test_loader, path, self.best_nmse)
                loss_list2.append(self.test_loss)
                self.best_nmse = best_nmse
            else:
                nmse = None, None

            # conduct saving, visualization and log printing
        return loss_list1, loss_list2, self.best_nmse

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" exam the model with validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader, path, best_nmse):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
            path (str): the path where model save
            best_nmse: best nmse
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, path, best_nmse, verbose=False)

    def _iteration(self, data_loader):
        loss_list = []
        a = 0
        for batch_idx, (sparse_gt,) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                loss_list.append(loss)
                print(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                      f'[{batch_idx + 1}/{len(data_loader)}] '
                      f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                      f'MSE loss: {loss:.3e} | ')
        for i in range(len(loss_list)):
            a = a + loss_list[i]
        a = a/len(loss_list)
        a = a.cpu().detach().numpy()
        del sparse_gt, sparse_pred
        torch.cuda.empty_cache()

        return a


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, path, best_nmse, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, nmse, best_nmse = self._iteration(test_data, path, best_nmse)

        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'NMSE: {nmse:.3e}\n')
        return loss, nmse, best_nmse

    def _iteration(self, data_loader, path, best_nmse):

        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt,) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            nmse = evaluator(sparse_pred, sparse_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                print(f'[{batch_idx + 1}/{len(data_loader)}] '
                      f'loss: {iter_loss.avg:.3e} |'
                      f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')
        print(iter_nmse.avg)
        print(best_nmse)
        iter_loss.avg = iter_loss.avg.cpu().detach().numpy()
        if iter_nmse.avg < best_nmse:
            best_nmse = iter_nmse.avg
            torch.save(self.model.state_dict(), path)
        print(f'=> Test NMSE: {iter_nmse.avg:.3e}\n'
              f'=> best NMSE: {best_nmse:.3e}\n')

        return iter_loss.avg, iter_nmse.avg, best_nmse