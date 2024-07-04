# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def draw(loss_list, accuracy_list):
#     fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
#     axs[0].plot(range(len(loss_list)), loss_list)
#     axs[0].set_title('Loss')
#     axs[1].plot(range(len(accuracy_list)), accuracy_list)
#     axs[1].set_title('Accuracy')
#     plt.show()
#
#
# def create_data():
#     features = np.random.randn(N_N, N_M)
#     tag = np.random.randn(N_N, N_M)
#     tag[tag < np.max(tag, axis=1, keepdims=True)] = 0
#     tag[tag != 0] = 1
#
#     return features, tag
#
#
# def activation_tanh(inputs):
#     return (1 - np.exp(- 2 * inputs)) / (1 + np.exp(- 2 * inputs))
#
#
# def activation_softmax(inputs):
#     exp_inputs = np.exp(inputs)
#     sum_inputs = np.sum(exp_inputs)
#
#     return exp_inputs / sum_inputs
#
#
# def normalize(inputs):
#     std = np.std(inputs, axis=1, keepdims=True)
#     mean = np.mean(inputs, axis=1, keepdims=True)
#     return (inputs - mean) / std
#
#
# def loss(tag, inputs):
#     return np.sum(- tag * np.log(inputs))
#
#
# def accuracy(tag, inputs):
#     inputs[inputs < np.max(inputs, axis=1, keepdims=True)] = 0
#     inputs[inputs != 0] = 1
#     return np.sum(tag * inputs) / tag.shape[0]
#
#
# class RNN:
#     def __init__(self):
#         self.u = np.random.randn(N_M, N_H)
#         self.dL_du = None
#
#         self.w = np.random.randn(N_H, N_H)
#         self.dL_dw = None
#
#         self.v = np.random.randn(N_H, N_M)
#         self.dL_dv = None
#
#         self.bx = np.random.randn()
#         self.dL_dbx = None
#
#         self.by = np.random.randn()
#
#         self.dL_dby = None
#
#         self.s = []
#         self.s.append(np.zeros((1, N_H)))
#         self.o = []
#         self.length = None
#         self.inputs = None
#         self.tag = None
#
#     def rnn_forward(self, inputs, tag):
#         self.inputs = inputs
#         self.tag = tag
#         self.length = inputs.shape[0]
#
#         for i in range(self.length):
#             s = activation_tanh(normalize(np.dot(inputs[i:i + 1, :], self.u) + np.dot(self.s[- 1], self.w) + self.bx))
#             o = activation_softmax(normalize(np.dot(s, self.v) + self.by))
#
#             self.s.append(s)
#             self.o.append(o)
#
#         return loss(tag, np.squeeze(np.array(self.o))), accuracy(tag, np.squeeze(np.array(self.o)))
#
#     def rnn_backward(self):
#         dL_ds_pre = 0
#         for i in range(len(self.o)):
#             dL_do = - self.tag[- (i + 1)] / self.o[- (i + 1)]
#             dL_dsoft = dL_do * self.o[- (i + 1)] * (1 - self.o[- (i + 1)])
#             dL_dnor_o = dL_dsoft / np.std(np.dot(self.s[- (i + 1)], self.v) + self.by, axis=1, keepdims=True)
#             self.dL_dv += np.dot(self.s[- (i + 1)].T, dL_dnor_o)
#             self.dL_dby += np.sum(dL_dnor_o)
#
#             dL_ds = np.dot(dL_dnor_o, self.v.T)
#             dL_dtan = dL_ds * (1 - self.s[- (i + 1)]**2)
#             dL_dnor_s = dL_dtan / np.std(np.dot(self.inputs[- (i + 1):-(i + 2):- 1, :], self.u) + np.dot(self.s[- (i + 1)], self.w) + self.bx, axis=1, keepdims=True)
#             self.dL_dw += np.dot(self.s[-(i + 2)].T, dL_dnor_s)
#             self.dL_du += np.dot(self.inputs[- (i + 1):- (i + 2):- 1].T, dL_dnor_s)
#             self.dL_dbx += np.sum(dL_dnor_s)
#
#             dL_ds_pre += np.dot(dL_dnor_s, self.w)
#
#         self.dL_dv /= self.length
#         self.dL_dby /= self.length
#         self.dL_dw /= self.length
#         self.dL_du /= self.length
#         self.dL_dbx /= self.length
#
#         self.v -= LEARNING_RATE * self.dL_dv
#         self.by -= LEARNING_RATE * self.dL_dby
#         self.w -= LEARNING_RATE * self.dL_dw
#         self.u -= LEARNING_RATE * self.dL_du
#         self.bx -= LEARNING_RATE * self.dL_dbx
#
#     def train(self, epochs, features, tag):
#         loss_list = []
#         accuracy_list = []
#         for i in range(epochs):
#             self.s = []
#             self.s.append(np.zeros((1, N_H)))
#             self.o = []
#             self.dL_du = np.zeros((N_M, N_H))
#             self.dL_dw = np.zeros((N_H, N_H))
#             self.dL_dv = np.zeros((N_H, N_M))
#             self.dL_dbx = 0
#             self.dL_dby = 0
#
#             rnn_loss, rnn_accuracy = self.rnn_forward(features, tag)
#             loss_list.append(rnn_loss)
#             accuracy_list.append(rnn_accuracy)
#
#             print(f'loss:{loss_list[-1]}--accuracy:{accuracy_list[-1]}')
#
#             self.rnn_backward()
#
#         return loss_list, accuracy_list
#
#
# def main():
#     rnn = RNN()
#
#     features, tag = create_data()
#
#     epochs = input('Number to train, Others to end\n')
#     while epochs.isnumeric():
#         loss_list, accuracy_list = rnn.train(int(epochs), features, tag)
#
#         draw(loss_list, accuracy_list)
#
#         epochs = input('Number to train, Others to end\n')
#
#
# N_H = 512
# N_M = 256
# N_N = 128
# LEARNING_RATE = 10
#
# if __name__ == '__main__':
#     main()
#
import numpy as np
import matplotlib.pyplot as plt


def draw(loss_list, accuracy_list):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    axs[0].plot(range(len(loss_list)), loss_list)
    axs[0].set_title('Loss')
    axs[1].plot(range(len(accuracy_list)), accuracy_list)
    axs[1].set_title('Accuracy')
    plt.show()


def create_data():
    features = np.random.randn(N_N, N_M)
    tag = np.random.randn(N_N, N_M)
    tag[tag < np.max(tag, axis=1, keepdims=True)] = 0
    tag[tag != 0] = 1

    return features, tag


def activation_tanh(inputs):
    return (1 - np.exp(- 2 * inputs)) / (1 + np.exp(- 2 * inputs))


def activation_softmax(inputs):
    exp_inputs = np.exp(inputs)
    sum_inputs = np.sum(exp_inputs, axis=1, keepdims=True)

    return exp_inputs / sum_inputs


def normalize(inputs):
    std = np.std(inputs, axis=1, keepdims=True)
    mean = np.mean(inputs, axis=1, keepdims=True)
    return (inputs - mean) / std


def loss(tag, inputs):
    return np.sum(- tag * np.log(inputs))


def accuracy(tag, inputs):
    inputs[inputs < np.max(inputs, axis=1, keepdims=True)] = 0
    inputs[inputs != 0] = 1
    return np.sum(tag * inputs) / tag.shape[0]


class RNN:
    def __init__(self):
        self.u = np.random.randn(N_M, N_H)

        self.w = np.random.randn(N_H, N_H)

        self.v = np.random.randn(N_H, N_M)

        self.bx = np.random.randn()

        self.by = np.random.randn()

        self.s = []
        self.s.append(np.zeros((1, N_H)))
        self.o = []
        self.length = None
        self.inputs = None
        self.tag = None

    def rnn_forward(self, inputs, tag):
        self.inputs = inputs
        self.tag = tag
        self.length = inputs.shape[0]

        for i in range(self.length):
            s = activation_tanh(normalize(np.dot(inputs[i:i + 1, :], self.u) + np.dot(self.s[- 1], self.w) + self.bx))
            o = activation_softmax(normalize(np.dot(s, self.v) + self.by))

            self.s.append(s)
            self.o.append(o)

        return loss(tag, np.squeeze(np.array(self.o))), accuracy(tag, np.squeeze(np.array(self.o)))

    def rnn_backward(self):
        dL_ds_pre = 0
        dL_du = np.zeros((N_M, N_H))
        dL_dw = np.zeros((N_H, N_H))
        dL_dv = np.zeros((N_H, N_M))
        dL_dbx = 0
        dL_dby = 0
        for i in range(len(self.o)):
            dL_do = - self.tag[- (i + 1)] / self.o[- (i + 1)]
            dL_dsoft = dL_do * self.o[- (i + 1)] * (1 - self.o[- (i + 1)])
            dL_dnor_o = dL_dsoft / np.std(np.dot(self.s[- (i + 1)], self.v) + self.by, axis=1, keepdims=True)
            dL_dv += np.dot(self.s[- (i + 1)].T, dL_dnor_o)
            dL_dby += np.sum(dL_dnor_o)

            dL_ds = np.dot(dL_dnor_o, self.v.T) + dL_ds_pre
            dL_dtan = dL_ds * (1 - self.s[- (i + 1)] ** 2)
            dL_dnor_s = dL_dtan / np.std(
                np.dot(self.inputs[- (i + 1):-(i + 2):- 1, :], self.u) + np.dot(self.s[- (i + 1)], self.w) + self.bx,
                axis=1, keepdims=True)
            dL_dw += np.dot(self.s[-(i + 2)].T, dL_dnor_s)
            dL_du += np.dot(self.inputs[- (i + 1):- (i + 2):- 1].T, dL_dnor_s)
            dL_dbx += np.sum(dL_dnor_s)

            dL_ds_pre = np.dot(dL_dnor_s, self.w)

        dL_dv /= self.length
        dL_dby /= self.length
        dL_dw /= self.length
        dL_du /= self.length
        dL_dbx /= self.length

        self.v -= LEARNING_RATE * dL_dv
        self.by -= LEARNING_RATE * dL_dby
        self.w -= LEARNING_RATE * dL_dw
        self.u -= LEARNING_RATE * dL_du
        self.bx -= LEARNING_RATE * dL_dbx

    def train(self, epochs, features, tag):
        loss_list = []
        accuracy_list = []
        for i in range(epochs):
            self.s = []
            self.s.append(np.zeros((1, N_H)))
            self.o = []

            rnn_loss, rnn_accuracy = self.rnn_forward(features, tag)
            loss_list.append(rnn_loss)
            accuracy_list.append(rnn_accuracy)

            print(f'loss:{loss_list[-1]}--accuracy:{accuracy_list[-1]}')

            self.rnn_backward()

        return loss_list, accuracy_list


def main():
    rnn = RNN()

    features, tag = create_data()

    epochs = input('Number to train, Others to end\n')
    while epochs.isnumeric():
        loss_list, accuracy_list = rnn.train(int(epochs), features, tag)

        draw(loss_list, accuracy_list)

        epochs = input('Number to train, Others to end\n')


N_H = 512
N_M = 256
N_N = 128
LEARNING_RATE = 10

if __name__ == '__main__':
    main()
