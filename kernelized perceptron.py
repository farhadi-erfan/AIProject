import pickle
from math import sqrt

import numpy as np
import idx2numpy


class Util:
    @classmethod
    def return_arr(cls, addr, img=False, length = 0):
        a = idx2numpy.convert_from_file(addr)
        if img:
            a = a.reshape((length, 784))
            a = cls.format(a.copy())
        return a

    @classmethod
    def normalize(cls, x):
        return x / 256

    @classmethod
    def linear_kernel(cls, x, y):
        return np.dot(x, y)

    @classmethod
    def quadratic_kernel(cls, x, y):
        return (np.dot(x, y) + 1) ** 2

    @classmethod
    def rbf_kernel(cls, x, y):
        diff = x - y
        print(diff, np.dot(diff, diff), np.exp(-np.dot(diff, diff)))
        return np.exp(-np.dot(diff, diff))

    @classmethod
    def format(cls, train_data):
        train_data[train_data > 0] = 1
        return train_data


class Cell:
    def __init__(self):
        self.alpha = np.zeros(train_size)

    def upgrade(self, index):
        self.alpha[index] += 1

    def downgrade(self, index):
        self.alpha[index] -= 1

    def compute_output(self, kernel_row):
        return np.sum(self.alpha * kernel_row)

    def print_cell(self):
        print(self.alpha)
        print('****')

    def save(self, i):
        with open("kernel" + str(i) + ".file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, i):
        with open("kernel" + str(i) + ".file", "rb") as f:
            dump = pickle.load(f)
            self.alpha = dump.alpha


class KernelizedPerceptron:
    def __init__(self):
        self.cells = [Cell() for i in range(10)]
        self.img, self.lbl = Util.return_arr('../train-images-idx3-ubyte', img=True, length=60000), Util.return_arr('../train-labels-idx1-ubyte')
        self.test_img, self.test_lbl = Util.return_arr('../t10k-images-idx3-ubyte', img=True, length=10000), Util.return_arr('../t10k-labels-idx1-ubyte')
        print('Kernelizing train:')
        for i in range(train_size):
            if i % 200 == 0:
                print(i)
            for j in range(train_size):
                kernel[i][j] = kernel_func(self.img[i], self.img[j])
        print('Kernelizing test:')
        for i in range(test_size):
            if i % 200 == 0:
                print(i)
            for j in range(train_size):
                kernel_test[i][0][j] = kernel_func(self.test_img[i], self.img[j])

    def compute_all_cells(self, kernel_row):
        a = []
        for i in range(10):
            a += [self.cells[i].compute_output(kernel_row)]
        ind = a.index(max(a))
        return ind

    def print_all_cells(self):
        for i in range(10):
            self.cells[i].print_cell()

    def train(self):
        print('training:')
        for i in range(train_size):
            if i % 200 == 0:
                print(i)
            res = self.compute_all_cells(kernel[:, i])
            if res != self.lbl[i]:
                self.cells[res].downgrade(i)
                self.cells[self.lbl[i]].upgrade(i)

    def test(self):
        acc = 0
        for i in range(test_size):
            res = self.compute_all_cells(kernel_test[i])
            if res == self.test_lbl[i]:
                acc += 1
        print('success rate:', acc / test_size)

    def save(self):
        with open("kernel_perceptron" + ".file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open("kernel_perceptron" + ".file", "rb") as f:
            dump = pickle.load(f)
            self.cells = dump.cells
            self.img = dump.img
            self.lbl = dump.lbl
            self.test_img = dump.test_img
            self.test_lbl = dump.test_lbl


kernel_func = Util.quadratic_kernel
train_size = 7500
test_size = 10000
kernel = np.zeros((train_size, train_size))
kernel_test = np.zeros((test_size, 1, train_size))

kp = KernelizedPerceptron()
kp.train()
kp.save()
kp.load()
kp.test()