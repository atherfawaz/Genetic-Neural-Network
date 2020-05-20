import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
import copy
import time

# GLOBALS
CROSS = 0.5
GENERATIONS = 200
N_POPS = 20
NETWORK = [4, 10, 3]
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.4
RETENTION_RATE = 0.4
SLEEP_TIMEOUT = 1

class NeuralNetwork:

    def __init__(self, features, classes):
        self.n_pops = N_POPS
        self.net_size = NETWORK
        self.network = [Connections(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE
        self.retain_rate = RETENTION_RATE
        self.features = features[:]
        self.classes = classes[:]

    def set_data(self, features, classes):
        self.features = features[:]
        self.classes = classes[:]

    def generate_neural_network(features, classes):
        return NeuralNetwork(features, classes)

    def print_processing(self, generation):
        if (generation % 50 is 0):
            print('Generation: ', generation)
            print('Accuracy: ', self.get_all_accuracy()[0])

    def begin(self):
        print('Starting new phase...')
        for generation in range(GENERATIONS):
            print('Generation: ' + str(generation) +
                  ', accuracy: ' + str(self.get_highest_accuracy()))
            self.evolve()
        print('Phase complete...')
        time.sleep(SLEEP_TIMEOUT)

    def crossover(self, father, mother):
        temp_network = copy.deepcopy(father)

        # cross-over bias
        for _ in range(self.network[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # replace genetic (bias) with mother's value
            if (random.uniform(0, 1) < self.crossover_rate):
                temp_network.biases[layer][point] = mother.biases[layer][point]

        # cross-over weight
        for _ in range(self.network[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # replace genetic (weight) with mother's value
            if random.uniform(0, 1) < self.crossover_rate:
                temp_network.weights[layer][point] = mother.weights[layer][point]

        return temp_network

    def mutation(self, child):
        temp_network = copy.deepcopy(child)

        # mutate bias
        for _ in range(self.network[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                temp_network.biases[layer][point] += random.uniform(-0.5, 0.5)

        # mutate weight
        for _ in range(self.network[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                temp_network.weights[layer][point[0],
                                            point[1]] += random.uniform(-0.5, 0.5)

        return temp_network

    def evolve(self):
        score_list = list(zip(self.network, self.get_all_scores()))
        score_list.sort(key=lambda x: x[1])
        score_list = [object[0] for object in score_list]

        best_retain = int(self.n_pops * self.retain_rate)
        score_list_best = score_list[:best_retain]

        retain_ordinary = int((self.n_pops - best_retain) * self.retain_rate)
        for _ in range(random.randint(0, retain_ordinary)):
            score_list_best.append(random.choice(score_list[best_retain:]))

        while(len(score_list_best) < self.n_pops):
            parent = random.choice(score_list_best)
            another_parent = random.choice(score_list_best)

            if (parent != another_parent):
                reproduce = self.crossover(parent, another_parent)
                reproduce = self.mutation(reproduce)
                score_list_best.append(reproduce)

        self.network = score_list_best

    def get_highest_accuracy(self):
        val = [network.accuracy(self.features, self.classes)
               for network in self.network][0]
        formatted_val = "{:.2f}".format(val)
        return val

    def get_all_accuracy(self):
        val = [network.accuracy(self.features, self.classes)
               for network in self.network]
        return val

    def get_all_scores(self):
        return [network.score(self.features, self.classes) for network in self.network]

    def get_random_point(self, type):
        temp = self.network[0]
        layer, point = random.randint(0, temp.layer_count - 2), 0
        if (type == 'weight'):
            row = random.randint(0, temp.weights[layer].shape[0] - 1)
            col = random.randint(0, temp.weights[layer].shape[1] - 1)
            point = (row, col)
        elif (type == 'bias'):
            point = random.randint(0, temp.biases[layer].size - 1)
        return (layer, point)


class Connections:

    def __init__(self, size):
        self.layer_count = len(size)
        self.size = size
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(size[:-1], size[1:])]

        self.bias_nitem = sum(size[1:])
        self.weight_nitem = sum(
            [self.weights[i].size for i in range(self.layer_count - 2)])

    def forward_propagation(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def score(self, features, classes):
        score = 0
        for i in range(features.shape[0]):
            predicted = self.forward_propagation(features[i].reshape(-1, 1))
            actual = classes[i].reshape(-1, 1)
            score += np.sum(np.power(predicted - actual, 2) / 2)
        return score

    def accuracy(self, features, classes):
        accuracy = 0
        for i in range(features.shape[0]):
            output = self.forward_propagation(features[i].reshape(-1, 1))
            accuracy += int(np.argmax(output) == np.argmax(classes[i]))
        return accuracy / features.shape[0] * 100

    def __str__(self):
        return "\nBias:\n\n" + str(self.biases) + "\nWeights:\n\n" + str(self.weights) + "\n\n"


def fetch_dataset():

    dataset = pd.read_excel('dataset.xlsx')

    sepal_width = np.array(dataset['Sepal width'])
    sepal_length = np.array(dataset['Sepal length'])
    petal_length = np.array(dataset['Petal length'])
    petal_width = np.array(dataset['Petal width'])
    flower_class = np.array(dataset['Species'])

    idx = 0
    for x in flower_class:
        if (x == 'Setosa'):
            flower_class[idx] = 1
        elif (x == 'Versicolor'):
            flower_class[idx] = 2
        elif (x == 'Virginica'):
            flower_class[idx] = 3
        idx += 1

    data = np.random.rand(flower_class.size, 5)
    data[:, 0] = sepal_length
    data[:, 1] = sepal_width
    data[:, 2] = petal_length
    data[:, 3] = petal_width
    data[:, 4] = flower_class

    return data


def distribute_data(data):

    features = data[:, :4]
    classes = data[:, 4]
    classes = classes.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(classes)
    classes = enc.transform(classes).toarray()

    return (features, classes)


def separate_data(features, classes):

    size = int(CROSS * features.shape[0])

    features_a = features[: size, :]
    classes_a = classes[: size, :]

    features_b = features[size:, :]
    classes_b = classes[size:, :]

    return (features_a, classes_a, features_b, classes_b)


def main():

    data = fetch_dataset()

    # shuffle
    np.take(data, np.random.permutation(data.shape[0]), axis=0, out=data)
    data = np.array(data)

    # get features and classes separated
    (features, classes) = distribute_data(data)

    (features_a, classes_a, features_b, classes_b) = separate_data(features, classes)

    # establish a neural network
    neural_network = NeuralNetwork.generate_neural_network(
        features_a, classes_a)

    # start training
    neural_network.begin()

    # get training accuracy on first training set
    print("Accuracy on training dataset (Phase 1): ",
          neural_network.get_highest_accuracy())

    # update parameters of features and classes
    neural_network.set_data(features_b, classes_b)

    # test on testing dataset
    print("Accuracy on testing dataset (Phase 1): ",
          neural_network.get_highest_accuracy())

    # train on previous testing data
    neural_network.begin()

    # get training accuracy on second training set
    print("Accuracy on training dataset (Phase 2): ",
          neural_network.get_highest_accuracy())

    # update parameters of features and classes
    neural_network.set_data(features_a, classes_a)

    # test on testing dataset
    print("Accuracy on testing dataset (Phase 2): ",
          neural_network.get_highest_accuracy())


if __name__ == "__main__":
    main()