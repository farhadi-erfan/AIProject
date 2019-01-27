import numpy as np
import idx2numpy


class Util:
    def __init__(self, images_addr, labels_addr):
        self.img_addr = images_addr
        self.labels_addr = labels_addr

    def img_label(self):
        images = idx2numpy.convert_from_file(self.img_addr)
        labels = idx2numpy.convert_from_file(self.labels_addr)
        return images, labels


class Cell:
    def __init__(self, vector_size, learning_rate):
        self.vec_size = vector_size
        self.vector = [0] * vector_size
        self.learning_rate = learning_rate

    def compute_output(self, input):
        res = 0
        for i in range(self.vec_size):
            res += self.vector[i] * input[i]

        return res / self.vec_size

    def update_vector(self, input, err):
        for i in range(self.vec_size):
            self.vector += self.learning_rate * input[i] * err


class Perceptron:
    def __init__(self):
        self.cells = [Cell() for i in range(10)]
        u = Util('../train-images-idx3-ubyte', '../train-labels-idx1-ubyte')
        self.img, self.lbl = u.img_label()

    def compute_all_cells(self, img):
        a = []
        for i in range(10):
            a += [self.cells[i].compute_output(img)]
        ind = a.index(max(a))
        return ind

    def update_all_cells(self, img, err_arr):
        for i in range(10):
            self.cells[i].update_vector(img, err_arr[i])

    @staticmethod
    def construct_err_array(acc, rej):
        a = [0] * 10
        a[acc] = 1
        a[rej] = -1
        return a

    def train(self):
        for i in range(60000):
            res = self.compute_all_cells(self.img[i])
            if not res == self.lbl[i]:
                err_arr = Perceptron.construct_err_array(self.lbl[i], res)
                self.update_all_cells(self.img[i], err_arr)

# np.set_printoptions(linewidth=np.nan)
# print(img[12])