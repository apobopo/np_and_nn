import numpy as np
import copy
import matplotlib.pyplot as plt


SHAPE = [2, 100, 50, 20, 1]
LEARNING_RATE = 0.001
BATCH = 8192


def draw(data):
    plt.scatter(x=data[:, 0], y=data[:, 1], c=['red' if i == 0 else 'yellow' for i in data[:, 2]], s=20)
    plt.show()


def create_batch(num):
    entry_list = []

    for i in range(num):
        y = np.random.uniform(-3, 3)
        x = np.random.uniform(-3, 3)
        # tag = 1 if (x**2 + y**2 > 1) & (x**2 + y**2 < 4) else 0
        tag = 1 if x**2 + y**2 > 1 else 0
        entry_list.append([x, y, tag])

    return np.array(entry_list)


def create_weights(n_y, n_h):
    return np.random.randn(n_y, n_h)


def create_biases(n_y):
    return np.random.randn(1, n_y)


def activation_ReLu(inputs):
    return np.maximum(0, inputs)


# def activation_SoftMax(inputs):
#     exp_inputs = np.exp(inputs)
#     sum_inputs = np.sum(exp_inputs, axis=1, keepdims=True)
#     outputs = exp_inputs / sum_inputs
#
#     return outputs


def activation_Sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def normalize(inputs):
    std = np.std(inputs, axis=1, keepdims=True)
    mean = np.mean(inputs, axis=1, keepdims=True)
    outputs = (inputs - mean) / std

    return outputs


def loss(tag, outputs):
    epsilon = 1e-15
    safe_outputs = np.clip(outputs, epsilon, 1 - epsilon)
    return np.sum(- tag * np.log(safe_outputs) - (1 - tag) * np.log(1 - safe_outputs)).round(2)


class Layer:

    def __init__(self, n_h, n_x):
        self.weights = create_weights(n_h, n_x)
        self.biases = create_biases(n_h)
        self.inputs = 0
        self.z = 0

    def layer_forward(self, inputs):
        z = np.dot(inputs, self.weights.T) + self.biases

        self.z = z
        self.inputs = inputs

        return z

    def layer_backward(self, dL):
        dw = (1 / BATCH) * np.dot(dL.T, self.inputs)
        db = (1 / BATCH) * np.sum(dL)
        dx = np.dot(dL, self.weights)

        self.weights -= LEARNING_RATE*dw
        self.biases -= LEARNING_RATE*db

        return dx


class NetWork:
    def __init__(self, network_shape):
        self.layers = []
        self.outputs = []
        self.loss = None
        self.copy_layers = None

        for i in range(len(network_shape) - 1):
            self.layers.append(Layer(network_shape[i+1], network_shape[i]))

    def network_forward(self, inputs):
        self.outputs = []
        self.outputs.append(inputs)
        for i in range(len(self.layers)):

            if i != len(self.layers) - 1:
                inputs = activation_ReLu(self.layers[i].layer_forward(inputs))
            else:
                inputs = activation_Sigmoid(self.layers[i].layer_forward(inputs))

            self.outputs.append(inputs)

        return self.outputs

    def network_backward(self, tag):
        self.loss = loss(tag, self.outputs[-1])
        self.copy_layers = copy.deepcopy(self.layers)

        dL = self.outputs[-1] - tag

        for i in range(len(self.layers)):
            layer = self.layers[-(i + 1)]
            dL = layer.layer_backward(dL)

            if i != 0:
                dL *= np.where(dL > 0, 1, 1)

        self.network_forward(self.outputs[0])
        update_loss = loss(tag, self.outputs[-1])

        if self.loss < update_loss:
            self.layers = self.copy_layers
            print(self.loss)
        else:
            print(update_loss)

    def diversified_train(self, epochs):
        for i in range(epochs):
            batch_data = create_batch(BATCH)
            batch_features = batch_data[:, 0:2]
            batch_tag = batch_data[:, 2:3]

            self.repetition_train(batch_features, batch_tag)
            print('-------------------')

    def repetition_train(self, batch_features, batch_tag):
        for i in range(2):
            self.network_forward(batch_features)
            self.network_backward(batch_tag)


def main():
    initialized_batch = create_batch(BATCH)
    initialized_features = initialized_batch[:, 0:2]

    draw(initialized_batch)

    while True:
        network = NetWork(SHAPE)
        initialized_final_outputs = np.rint(network.network_forward(initialized_features)[-1])
        initialized_result = np.concatenate((initialized_features, initialized_final_outputs), axis=1)

        draw(initialized_result)

        use_this_initialization = input("Use this initialization? Number to train, Others to initialize again\n")

        if use_this_initialization.isnumeric():
            break

    epochs = int(use_this_initialization)

    while True:
        network.diversified_train(epochs)
        update_final_outputs = np.rint(network.network_forward(initialized_features)[-1])
        update_result = np.concatenate((initialized_features, update_final_outputs), axis=1)

        draw(update_result)

        epochs = input("Over? Number to train again, Others to end\n")

        if epochs.isnumeric():
            epochs = int(epochs)
        else:
            break


if __name__ == '__main__':
    main()
