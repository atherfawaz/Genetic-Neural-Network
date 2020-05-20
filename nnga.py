import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
import copy
import time

# GLOBALS
CROSS = 0.5
GENERATIONS = 200
SELECTION_RATE = 20
NETWORK = [4, 10, 3]
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.3
RETENTION_RATE = 0.3
SLEEP_TIMEOUT = 1


class NeuralNetwork:

    def __init__(self, features, classes):
        self.selection_rate = SELECTION_RATE
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE
        self.retention_rate = RETENTION_RATE
        self.dimensions = NETWORK
        self.network = [Connections(self.dimensions)
                        for i in range(self.selection_rate)]
        self.features = features[:]
        self.classes = classes[:]

    def set_data(self, features, classes):
        self.features = features[:]
        self.classes = classes[:]

    def generate(features, classes):
        return NeuralNetwork(features, classes)

    def print_processing(self, generation):
        if (generation % 50 is 0):
            print('Generation: ', generation)
            print('Accuracy: ', self.accuracy()[0])

    def train(self):
        print('Starting new phase...')
        for generation in range(GENERATIONS):
            print('Generation: ' + str(generation) +
                  ' ; Accuracy: ' + str(self.calculate_accuracy()), '%')
            self.natural_selection()
        print('Phase complete...')
        time.sleep(SLEEP_TIMEOUT)

    def genetic_crossover(self, parent_a, parent_b):
        temp_network = copy.deepcopy(parent_a)

        for _ in range(self.network[0].bias_sum):
            layer, point = self.fetch_point('bias')
            if (random.uniform(0, 1) < self.crossover_rate):
                temp_network.biases[layer][point] = parent_b.biases[layer][point]

        for _ in range(self.network[0].weight_sum):
            layer, point = self.fetch_point('weight')
            if (random.uniform(0, 1) < self.crossover_rate):
                temp_network.weights[layer][point] = parent_b.weights[layer][point]

        return temp_network

    def genetic_mutation(self, child):
        temp_network = copy.deepcopy(child)

        for _ in range(self.network[0].bias_sum):
            layer, point = self.fetch_point('bias')
            if (random.uniform(0, 1) < self.mutation_rate):
                temp_network.biases[layer][point] += random.uniform(-0.5, 0.5)

        for _ in range(self.network[0].weight_sum):
            layer, point = self.fetch_point('weight')
            if (random.uniform(0, 1) < self.mutation_rate):
                temp_network.weights[layer][point[0],
                                            point[1]] += random.uniform(-0.5, 0.5)

        return temp_network

    def natural_selection(self):
        score_list = list(zip(self.network, self.cost_function()))
        score_list.sort(key=lambda x: x[1])
        score_list = [object[0] for object in score_list]

        best_retain = int(self.selection_rate * self.retention_rate)
        score_list_best = score_list[:best_retain]

        retain_ordinary = int(
            (self.selection_rate - best_retain) * self.retention_rate)

        for _ in range(random.randint(0, retain_ordinary)):
            score_list_best.append(random.choice(score_list[best_retain:]))

        while(len(score_list_best) < self.selection_rate):
            parent_a = random.choice(score_list_best)
            parent_b = random.choice(score_list_best)

            if (parent_a != parent_b):
                reproduce = self.genetic_crossover(parent_a, parent_b)
                reproduce = self.genetic_mutation(reproduce)
                score_list_best.append(reproduce)

        self.network = score_list_best

    def calculate_accuracy(self):
        val = [network.accuracy(self.features, self.classes)
               for network in self.network][0]
        formatted_val = "{:.2f}".format(val)
        return formatted_val

    def accuracy(self):
        val = [network.accuracy(self.features, self.classes)
               for network in self.network]
        return val

    def cost_function(self):
        return [network.score(self.features, self.classes) for network in self.network]

    def fetch_point(self, type):
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

        self.bias_sum = sum(size[1:])
        self.weight_sum = sum(
            [self.weights[i].size for i in range(self.layer_count - 2)])

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
        
    def forward_propagation(self, val):
        for biases, weights in zip(self.biases, self.weights):
            val = self.sigmoid(np.dot(weights, val) + biases)
        return val

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

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
    encoder = OneHotEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes).toarray()

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

    # shuffle dataset
    np.take(data, np.random.permutation(data.shape[0]), axis=0, out=data)
    data = np.array(data)

    # get features and classes separated
    (features, classes) = distribute_data(data)

    # separate for two-fold cross validation
    (features_a, classes_a, features_b, classes_b) = separate_data(features, classes)

    # generate a neural network
    neural_network = NeuralNetwork.generate(features_a, classes_a)

    # start training
    neural_network.train()

    """
                TWO-FOLD CROSS VALIDATION
    1. Train
    2. Get training accuracy on first training set
    3. Update parameters of features and classes
    4. Test on testing dataset
    5. Train on previous testing data
    6. Get training accuracy on second training set
    7. Update parameters of features and classes
    8. Test on previous training dataset
    
    """

    # get training accuracy on first training set
    print("Accuracy on training dataset (Phase 1): ",
          neural_network.calculate_accuracy())

    # update parameters of features and classes
    neural_network.set_data(features_b, classes_b)

    # test on testing dataset
    print("Accuracy on testing dataset (Phase 1): ",
          neural_network.calculate_accuracy())

    # train on previous testing data
    neural_network.train()

    # get training accuracy on second training set
    print("Accuracy on training dataset (Phase 2): ",
          neural_network.calculate_accuracy())

    # update parameters of features and classes
    neural_network.set_data(features_a, classes_a)

    # test on previous training dataset
    print("Accuracy on testing dataset (Phase 2): ",
          neural_network.calculate_accuracy())


if __name__ == "__main__":
    main()
