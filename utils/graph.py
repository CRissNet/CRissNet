import numpy as np
import matplotlib.pyplot as plt
import torch

__all__ = ['painting']


class painting:
    def __init__(self, epoch, listA, listB, model, path):
        self.epochs = epoch
        self.listA = listA  # TRAIN_LOSS
        self.listB = listB  # TEST_LOSS
        self.model = model
        self.path = path

    def paint(self):
        x1 = np.linspace(1, self.epochs, len(self.listA))
        x2 = np.linspace(1, self.epochs, len(self.listB))
        self.listA = self.listA
        self.listB = self.listB
        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(1, 1, 1)
        ax1.set_title("Train_loss")
        plt.semilogy(x1, self.listA, label='Train_loss', color='green')
        plt.semilogy(x2, self.listB, label='Test_loss', color='blue')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.savefig(self.path)
        plt.close()
