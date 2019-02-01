import pickle
from math import sqrt
import numpy as np
import idx2numpy


class Util:
    @classmethod
    def return_arr(cls, addr):
        return idx2numpy.convert_from_file(addr)


class Cell:
    def __init__(self, vector_size, learning_rate):
        self.vec_size = vector_size
        self.vector = np.random.uniform(0, 1, size=(28, 28))
        self.learning_rate = learning_rate
        np.set_printoptions(linewidth=np.nan)

    @staticmethod
    def map_binary(x):
        return 1 if x > 0 else 0

    def compute_output(self, input):
        res = 0
        for i in range(int(sqrt(self.vec_size))):
            for j in range(int(sqrt(self.vec_size))):
                res += self.vector[i][j] * input[i][j]

        return res / self.vec_size

    def downgrade(self, input):
        self.vector -= input

    def upgrade(self, input):
        self.vector += input

    def print_cell(self):
        print(self.vector)
        print('****')


class Perceptron:
    def __init__(self):
        self.cells = [Cell(28*28, 1) for i in range(10)]
        self.img, self.lbl = Util.return_arr('../train-images-idx3-ubyte'), Util.return_arr('../train-labels-idx1-ubyte')
        self.test_img, self.test_lbl = Util.return_arr('../t10k-images-idx3-ubyte'), Util.return_arr('../t10k-labels-idx1-ubyte')

    def compute_all_cells(self, img):
        a = []
        for i in range(10):
            a += [self.cells[i].compute_output(img)]
        ind = a.index(max(a))
        return ind

    def print_all_cells(self):
        for i in range(10):
            self.cells[i].print_cell()

    def train(self):
        print('training:')
        for i in range(60000):
            if i % 2000 == 0:
                print(i)
            res = self.compute_all_cells(self.img[i])
            if not res == self.lbl[i]:
                self.cells[res].downgrade(self.img[i])
                self.cells[self.lbl[i]].upgrade(self.img[i])
        self.print_all_cells()

    def test(self):
        acc = 0
        for i in range(10000):
            res = self.compute_all_cells(img=self.test_img[i])
            if res == self.test_lbl[i]:
                acc += 1
        print(acc / 10000)

    def save(self):
        with open("perceptron.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open("perceptron.file", "rb") as f:
            dump = pickle.load(f)
            self.cells = dump.cells
            self.lbl = dump.lbl
            self.img = dump.img
            self.test_img = dump.test_img
            self.test_lbl = dump.test_lbl


p = Perceptron()
p.train()
# p.save()
# p.load()
p.test()