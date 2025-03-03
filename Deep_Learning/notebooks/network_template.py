import numpy as np
import matplotlib.pyplot as plt
from helper_functions import softmax, softmax_dLdZ, relu, relu_prime, cross_entropy
from helper_functions import load_data_cifar

class Network(object):
    def __init__(self, sizes, optimizer="sgd", reg=False, l_schedule=False):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.random.default_rng().standard_normal((x, 1)) * 0.01 for x in sizes[1:]] #[np.zeros((x, 1)) for x in sizes[1:]]
        self.batch_size = None
        self.reg = reg
        self.l_schedule = l_schedule
        self.optimizer = optimizer
        if self.optimizer == "adam":
            # Adam optimizer parameters
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0

    def train(self, training_data,training_class, val_data, val_class, epochs, mini_batch_size, eta, lmbd=0.001, decay=0.999):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation
        loss_train_list = []
        acc_train_list = []
        loss_val_list = []
        acc_val_list = []
        
        iteration_index = 0
        eta_current = eta

        self.batch_size = mini_batch_size

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch"+str(j))
            if self.l_schedule and j > 0:
                #eta_current = eta * np.e**(-decay*j) # exponential
                eta_current = eta * (1 - j/epochs) # linear
                #eta_current = eta / np.sqrt(j) # inverse square
                #eta_current = 1/2 * eta * (1 + np.cos(j*np.pi/epochs))# cosine
            print("Current learning rate: ", eta_current)

            indices = np.arange(n)
            np.random.shuffle(indices)
            shuffled_data = training_data[:, indices]
            shuffled_class = training_class[:, indices]

            loss_avg = 0.0
            #mini_batches = [
            #    (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
            #    for k in range(0, n, mini_batch_size)]
            mini_batches = [(shuffled_data[:, k:k + mini_batch_size], shuffled_class[:, k:k + mini_batch_size]) 
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = net.backward_pass(mini_batch[1], Zs, As)

                self.update_network(gw, gb, eta_current, lmbd*np.log(j+1))

                # Implement the learning rate schedule for Task 5
                eta_current = eta
                iteration_index += 1

                loss = cross_entropy(mini_batch[1], As[-1])
                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            print("On train dataset")
            loss_res, acc_res = self.eval_network(train_data, train_class)
            loss_train_list.append(loss_res)
            acc_train_list.append(acc_res)
            print("On eval dataset")
            loss_res, acc_res = self.eval_network(val_data, val_class)
            loss_val_list.append(loss_res)
            acc_val_list.append(acc_res)
            
        return loss_train_list, acc_train_list, loss_val_list, acc_val_list

    def eval_network(self, validation_data,validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            Zs, As = self.forward_pass(example)
            output_num = np.argmax(As[-1], axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, As[-1])
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))
        return loss_avg/n, tp/n

    def update_network(self, gw, gb, eta, lmbd=0.1):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                if self.reg:
                    self.weights[i] = (1 - eta*lmbd/self.batch_size)*self.weights[i] - eta * gw[i] 
                else:
                    self.weights[i] = self.weights[i] - eta * gw[i]
                self.biases[i] = self.biases[i] - eta * gb[i]
        elif self.optimizer == "adam":
            self.t += 1
            for i in range(len(self.weights)):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gw[i] ** 2)
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                if self.reg:
                    self.weights[i] = (1 - eta * lmbd / self.batch_size) * self.weights[i] - eta * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                else:
                    self.weights[i] -= eta * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb[i] ** 2)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                self.biases[i] -= eta * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            raise ValueError('Unknown optimizer:'+self.optimizer)

    def forward_pass(self, x):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        a = x
        Zs = []
        As = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b  
            Zs.append(z)

            if len(As) < len(self.weights): 
                a = relu(z) # sigmoid_t(z)
            else: 
                a = softmax(z) # handle last layer
            As.append(a)
        return Zs, As

    def backward_pass(self, target, Zs, As):
        gb = [np.zeros(b.shape) for b in self.biases]
        gw = [np.zeros(w.shape) for w in self.weights]

        for layer in range(-1, -len(self.weights)-1, -1):
            if layer == -1:
                delta = softmax_dLdZ(As[layer], target)
            else:
                z = Zs[layer]
                delta = np.dot(self.weights[layer+1].T, delta) * relu_prime(z) #sigmoid_t_prime(z)

            gb[layer] = np.array([np.sum(delta, axis=1)/self.batch_size]).T
            gw[layer] = np.dot(delta, As[layer-1].T)/self.batch_size

        return gw, gb

if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(train_data.shape[1] * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    
    net = Network([train_data.shape[0], 100, 100, 100, 10], optimizer="adam", reg=True, l_schedule=False)
    loss_train, acc_train, loss_val, acc_val = net.train(train_data, train_class, val_data, val_class, 20, 64, eta=0.0002, lmbd=0.5)
    net.eval_network(test_data, test_class)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(loss_train)
    ax[0].plot(loss_val)
    ax[0].legend(["Train loss", "Eval loss"])
    ax[0].set_title("Loss per epoch")
    ax[1].plot(acc_train)
    ax[1].plot(acc_val)
    ax[1].legend(["Train acc", "Eval acc"])
    ax[1].set_title("Accuracy per epoch")
    fig.show()
    plt.savefig('learning_result.png')
    """
    #print(val_class[:5], val_class.shape)

    x = np.array([[0.5, 0.5, 0.1]]).T

    y = np.array([[0.9, 0.1]]).T

    w1 = np.array([[0.1, 0.2, 0.5], 
                [0.4, 0.1, 0.5]])

    b1 = np.array([[0.2, 0.2]]).T
    b2 = np.array([[0.2, 0.1]]).T
    w2 = np.array([[0.6, 0.1], 
                [0.5, 0.2]])
    
    w = [w1, w2]
    b = [b1, b2]
    X = np.array([[0.5, 0.5, 0.1]])
    net = Network([3, 2, 2], w, b, optimizer="sgd")
    Zs, As = net.forward_pass(X.T)
    gw, gb = net.backward_pass(y, Zs, As)
    net.update_network(gw, gb, 0.1)
    print(net.weights)"""
