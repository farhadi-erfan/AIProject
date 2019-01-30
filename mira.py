from math import sqrt

import numpy as np
import idx2numpy


def normalize(x):
    return x / 256


class Util:
    @classmethod
    def return_arr(self, addr, img=False):
        a = idx2numpy.convert_from_file(addr)
        if img:
            a = np.array(list(map(np.vectorize(normalize), a)))
        return a


class Cell:
    def __init__(self, vector_size, learning_rate):
        self.vec_size = vector_size
        self.vector = np.random.uniform(0, 1, size=(28, 28)).astype(np.float64)
        self.learning_rate = learning_rate
        np.set_printoptions(linewidth=np.nan)

    def compute_output(self, input):
        res = 0
        for i in range(int(sqrt(self.vec_size))):
            for j in range(int(sqrt(self.vec_size))):
                res += self.vector[i][j] * input[i][j]

        return res / self.vec_size

    def upgrade(self, input, learning_rate):
        self.vector += np.multiply(learning_rate, input)

    def downgrade(self, input, learning_rate):
        self.vector -= np.multiply(learning_rate, input)

    def print_cell(self):
        print(self.vector)
        print('****')


class MIRA:
    def __init__(self):
        self.cells = [Cell(28*28, 1) for i in range(10)]
        self.img, self.lbl = Util.return_arr('../train-images-idx3-ubyte', img=True), Util.return_arr('../train-labels-idx1-ubyte')
        self.test_img, self.test_lbl = Util.return_arr('../t10k-images-idx3-ubyte', img=True), Util.return_arr('../t10k-labels-idx1-ubyte')

    def compute_all_cells(self, img):
        a = []
        for i in range(10):
            a += [self.cells[i].compute_output(img)]
        ind = a.index(max(a))
        return ind

    def print_all_cells(self):
        for i in range(10):
            self.cells[i].print_cell()


    @classmethod
    def normalize(cls, vec):
        vec2 = vec.copy()
        for i in range(len(vec)):
            for j in range(len(vec)):
                vec2[i][j] /= 255
        return vec2

    def train(self):
        for i in range(60000):
            if i % 2000 == 0:
                print(i)
            res = self.compute_all_cells(self.img[i])
            # self.img[i] = MIRA.normalize(self.img[i])
            if not res == self.lbl[i]:
                learning_rate = (np.dot(self.cells[res].vector.ravel() -
                                        self.cells[self.lbl[i]].vector.ravel(), self.img[i].ravel()) + 1) / (2 *
                                                                np.dot(self.img[i].ravel(), self.img[i].ravel()) + 1)
                # learning_rate = min(learning_rate, 10)
                self.cells[res].downgrade(self.img[i], learning_rate)
                self.cells[self.lbl[i]].upgrade(self.img[i], learning_rate)
        self.print_all_cells()

    def test(self):
        acc = 0
        for i in range(10000):
            res = self.compute_all_cells(img=self.test_img[i])
            if res == self.test_lbl[i]:
                acc += 1
        print(acc / 10000)


m = MIRA()
m.train()
m.test()