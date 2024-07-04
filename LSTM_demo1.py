import numpy as np
import matplotlib.pyplot as plt


def create_data(n_n, n_m):
    features = np.random.randn(n_n, n_m)
    tag = np.random.randn(n_n, n_m)
    tag = np.where(tag < np.max(tag, axis=1, keepdims=True), 0, 1)

    return features, tag


def loss(inputs, tag):
    return np.sum(- tag * np.log(inputs))


def accuracy(inputs, tag):
    accuracy_inputs = np.where(inputs < np.max(inputs, axis=1, keepdims=True), 0, 1)
    return np.sum(accuracy_inputs * tag) / tag.shape[0]


def draw(loss_list, accuracy_list):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    axes[0].plot(range(len(loss_list)), loss_list)
    axes[0].set_title('Loss')
    axes[1].plot(range(len(accuracy_list)), accuracy_list)
    axes[1].set_title('Accuracy')

    plt.show()


class Sigmoid:
    @staticmethod
    def activate(inputs):
        return 1 / (1 + np.exp(- inputs))

    @staticmethod
    def derivative(inputs):
        return inputs * (1 - inputs)


class Tanh:
    @staticmethod
    def activate(inputs):
        return (1 - np.exp(- 2 * inputs)) / (1 + np.exp(- 2 * inputs))

    @staticmethod
    def derivative(inputs):
        return 1 - inputs**2


class SoftMax:
    @staticmethod
    def activate(inputs):
        exp_inputs = np.exp(inputs)
        sum_inputs = np.sum(exp_inputs, axis=1, keepdims=True)
        return exp_inputs / sum_inputs

    @staticmethod
    def derivative(inputs):
        return inputs * (1 - inputs)


class Normalize:
    @staticmethod
    def activate(inputs):
        inputs_std = np.std(inputs, axis=1, keepdims=True)
        inputs_mean = np.mean(inputs, axis=1, keepdims=True)
        return (inputs - inputs_mean) / inputs_std

    @staticmethod
    def derivative(inputs):
        return 1 / np.std(inputs, axis=1, keepdims=True)


class LSTM:
    def __init__(self, n_h, n_m):
        self.n_h = n_h
        self.n_m = n_m
        self.length = None

        self.wf = np.random.randn(n_m + n_h, n_h)
        self.bf = np.random.randn()

        self.wi = np.random.randn(n_m + n_h, n_h)
        self.bi = np.random.randn()

        self.wc = np.random.randn(n_m + n_h, n_h)
        self.bc = np.random.randn()

        self.wo = np.random.randn(n_m + n_h, n_h)
        self.bo = np.random.randn()

        self.v = np.random.randn(n_h, n_m)
        self.b = np.random.randn()

        self.c = [np.zeros((1, n_h))]
        self.h = [np.zeros((1, n_h))]
        self.x_and_h = []
        self.f = []
        self.i = []
        self.candidate = []
        self.o = []
        self.y = []
        self.nor_sig_f = []
        self.nor_sig_i = []
        self.nor_tanh_candidate = []
        self.nor_sig_o = []
        self.nor_soft_y = []

    def lstm_forward(self, features, tag):
        self.length = features.shape[0]
        for j in range(self.length):
            x_and_h = np.concatenate((features[j:j + 1, :], self.h[-1]), axis=1)

            nor_sig_f = np.dot(x_and_h, self.wf) + self.bf
            f = Sigmoid.activate(Normalize.activate(nor_sig_f))

            nor_sig_i = np.dot(x_and_h, self.wi) + self.bi
            i = Sigmoid.activate(Normalize.activate(nor_sig_i))

            nor_tanh_candidate = np.dot(x_and_h, self.wc) + self.bc
            candidate = Tanh.activate(Normalize.activate(nor_tanh_candidate))

            nor_sig_o = np.dot(x_and_h, self.wo) + self.bo
            o = Sigmoid.activate(Normalize.activate(nor_sig_o))

            c = self.c[-1] * f + i * candidate
            h = Tanh.activate(c) * o
            nor_soft_y = np.dot(h, self.v) + self.b
            y = SoftMax.activate(Normalize.activate(nor_soft_y))

            self.c.append(c)
            self.h.append(h)
            self.x_and_h.append(x_and_h)
            self.f.append(f)
            self.i.append(i)
            self.candidate.append(candidate)
            self.o.append(o)
            self.y.append(y)
            self.nor_sig_f.append(nor_sig_f)
            self.nor_sig_i.append(nor_sig_i)
            self.nor_tanh_candidate.append(nor_tanh_candidate)
            self.nor_sig_o.append(nor_sig_o)
            self.nor_soft_y.append(nor_soft_y)

        array_y = np.squeeze(np.array(self.y))

        return loss(array_y, tag), accuracy(array_y, tag)

    def lstm_backward(self, tag):
        dL_dv = np.zeros((self.n_h, self.n_m))
        dL_db = 0
        dL_dwo = np.zeros((self.n_m + self.n_h, self.n_h))
        dL_dbo = 0
        dL_dwf = np.zeros((self.n_m + self.n_h, self.n_h))
        dL_dbf = 0
        dL_dwi = np.zeros((self.n_m + self.n_h, self.n_h))
        dL_dbi = 0
        dL_dwc = np.zeros((self.n_m + self.n_h, self.n_h))
        dL_dbc = 0

        dL_dh_pre = np.zeros((1, self.n_h))
        dL_dc_pre = np.zeros((1, self.n_h))

        for i in range(len(self.y)):
            dL_dy = - tag[- (i + 1):- (i + 2):- 1] / self.y[- (i + 1)]
            dL_dsoft_y = dL_dy * SoftMax.derivative(self.y[-(i + 1)])
            dL_dnor_soft_y = dL_dsoft_y * Normalize.derivative(self.nor_soft_y[- (i + 1)])
            dL_dv += np.dot(self.h[- (i + 1)].T, dL_dnor_soft_y)
            dL_db += np.sum(dL_dnor_soft_y)

            dL_dh = np.dot(dL_dnor_soft_y, self.v.T) + dL_dh_pre
            dL_do = dL_dh * Tanh.activate(self.c[- (i + 1)])
            dL_dsig_o = dL_do * Sigmoid.derivative(self.o[- (i + 1)])
            dL_dnor_sig_o = dL_dsig_o * Normalize.derivative(self.nor_sig_o[- (i + 1)])
            dL_dwo += np.dot(self.x_and_h[- (i + 1)].T, dL_dnor_sig_o)
            dL_dbo += np.sum(dL_dnor_sig_o)

            dL_dc = dL_dh * self.o[- (i + 1)] * Tanh.derivative(self.c[- (i + 1)]) + dL_dc_pre

            dL_df = dL_dc * self.c[- (i + 2)]
            dL_dsig_f = dL_df * Sigmoid.derivative(self.f[- (i + 1)])
            dL_dnor_sig_f = dL_dsig_f * Normalize.derivative(self.nor_sig_f[- (i + 1)])
            dL_dwf += np.dot(self.x_and_h[- (i + 1)].T, dL_dnor_sig_f)
            dL_dbf += np.sum(dL_dnor_sig_f)

            dL_di = dL_dc * self.candidate[- (i + 1)]
            dL_dsig_i = dL_di * Sigmoid.derivative(self.i[- (i + 1)])
            dL_dnor_sig_i = dL_dsig_i * Normalize.derivative(self.nor_sig_i[- (i + 1)])
            dL_dwi += np.dot(self.x_and_h[- (i + 1)].T, dL_dnor_sig_i)
            dL_dbi += np.sum(dL_dnor_sig_i)

            dL_dcandidate = dL_dc * self.i[- (i + 1)]
            dL_dtanh_candidate = dL_dcandidate * Tanh.derivative(self.candidate[- (i + 1)])
            dL_dnor_tanh_candidate = dL_dtanh_candidate * Normalize.derivative(self.nor_tanh_candidate[- (i + 1)])
            dL_dwc += np.dot(self.x_and_h[- (i + 1)].T, dL_dnor_tanh_candidate)
            dL_dbc += np.sum(dL_dnor_tanh_candidate)

            dL_dc_pre = dL_dc * self.f[- (i + 1)]
            dL_dh_pre = (np.dot(dL_dnor_sig_o, self.wo.T) + np.dot(dL_dnor_tanh_candidate, self.wc.T) + np.dot(dL_dnor_sig_i, self.wi.T) + np.dot(dL_dnor_sig_f, self.wf.T))[:, self.n_m:]

        dL_dv /= self.length
        dL_db /= self.length
        dL_dwo /= self.length
        dL_dbo /= self.length
        dL_dwf /= self.length
        dL_dbf /= self.length
        dL_dwi /= self.length
        dL_dbi /= self.length
        dL_dwc /= self.length
        dL_dbc /= self.length

        self.v -= LEARNING_RATE * dL_dv
        self.b -= LEARNING_RATE * dL_db
        self.wo -= LEARNING_RATE * dL_dwo
        self.bo -= LEARNING_RATE * dL_dbo
        self.wf -= LEARNING_RATE * dL_dwf
        self.bf -= LEARNING_RATE * dL_dbf
        self.wi -= LEARNING_RATE * dL_dwi
        self.bi -= LEARNING_RATE * dL_dbi
        self.wc -= LEARNING_RATE * dL_dwc
        self.bc -= LEARNING_RATE * dL_dbc

    def train(self, epochs, features, tag):
        loss_list = []
        accuracy_list = []
        for i in range(epochs):
            loss, accuracy = self.lstm_forward(features, tag)
            self.lstm_backward(tag)

            loss_list.append(loss)
            accuracy_list.append(accuracy)

            self.c = [np.zeros((1, self.n_h))]
            self.h = [np.zeros((1, self.n_h))]
            self.x_and_h = []
            self.f = []
            self.i = []
            self.candidate = []
            self.o = []
            self.y = []
            self.nor_tanh_candidate = []
            self.nor_sig_i = []
            self.nor_sig_f = []
            self.nor_sig_o = []
            self.nor_soft_y = []

            print(f'epochs:{i}——loss:{loss}--accuracy:{accuracy}')

        return loss_list, accuracy_list


def main():
    features, tag = create_data(N_N, N_M)
    lstm = LSTM(N_N, N_M)
    epochs = input('Number to train, Others to end:')
    while epochs.isnumeric():
        loss_list, accuracy_list = lstm.train(int(epochs), features, tag)

        draw(loss_list, accuracy_list)

        epochs = input('Number to train, Others to end:')


N_M = 64
N_H = 128
N_N = 256
LEARNING_RATE = 10

if __name__ == '__main__':
    main()
