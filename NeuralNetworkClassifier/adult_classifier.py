# coding: utf-8
# # ANN, by Seale
import numpy as np
from numpy import exp, array, random, dot
#from bigfloat import *


def workclass(text):
    if text == "Private":
        return 1
    elif text == "Self-emp-not-inc":
        return 2
    elif text == "Self-emp-inc":
        return 3
    elif text == "Federal-gov":
        return 4
    elif text == "Local-gov":
        return 5
    elif text == "State-gov":
        return 6
    elif text == "Without-pay":
        return 7
    elif text == "Never-worked":
        return 8
    else:
        return 0

def education(text):
    if text == "Bachelors":
        return 1
    elif text == "Some-college":
        return 2
    elif text == "11th":
        return 3
    elif text == "HS-grad":
        return 4
    elif text == "Prof-school":
        return 5
    elif text == "Assoc-acdm":
        return 6
    elif text == "Assoc-voc":
        return 7
    elif text == "9th":
        return 8
    elif text == "7th-8th":
        return 9
    elif text == "12th":
        return 10
    elif text == "Masters":
        return 11
    else:
        return 0

def  maritalStatus(text):
    if text == "Married-civ-spouse":
        return 1
    elif text == "Divorced":
        return 2
    elif text == "Never-married":
        return 3
    elif text == "Separated":
        return 4
    elif text == "Widowed":
        return 5
    elif text == "Married-spouse-absent":
        return 6
    elif text == "Married-AF-spouse":
        return 7
    else:
        return 0

def occupation(text):
    if text == "Tech-support":
        return 1
    elif text == "Craft-repair":
        return 2
    elif text == "Other-service":
        return 3
    elif text == "Sales":
        return 4
    elif text == "Exec-managerial":
        return 5
    elif text == "Prof-specialty":
        return 6
    elif text == "Handlers-cleaners":
        return 7
    elif text == "Machine-op-inspct":
        return 8
    elif text == "Adm-clerical":
        return 9
    elif text == "Farming-fishing":
        return 10
    elif text == "Transport-moving":
        return 11
    elif text == "Priv-house-serv":
        return 12
    elif text == "Protective-serv":
        return 13
    elif text == "Armed-Forces":
        return 14
    else:
        return 0

def relationship(text):
    if text == "Husband":
        return 3
    elif text == "Not-in-family":
        return 4
    elif text == "Other-relative":
        return 5
    elif text == "Unmarried":
        return 6
    else:
        return 0

def race(text):
    if text == "White":
        return 1
    elif text == "Black":
        return 2
    elif text == "Asian-Pac-Islander":
        return 3
    elif text == "Amer-Indian-Eskimo":
        return 4
    elif text == "Other":
        return 5
    else:
        return 0

def sex(text):
    if text == "Female":
        return 1
    elif text == "Male":
        return 2
    else:
        return 0

def nativeCountry(text):
    if text == "United-States":
        return 1
    elif text == "Cambodia":
        return 2
    elif text == "England":
        return 3
    elif text == "Puerto-Rico":
        return 4
    elif text == "Canada":
        return 5
    elif text == "Germany":
        return 6
    elif text == "Outlying-US(Guam-USVI-etc)":
        return 7
    elif text == "India":
        return 8
    elif text == "Japan":
        return 9
    elif text == "Greece":
        return 10
    elif text == "South":
        return 11
    elif text == "China":
        return 12
    elif text == "Cuba":
        return 13
    elif text == "Iran":
        return 14
    elif text == "Honduras":
        return 15
    elif text == "Philippines":
        return 16
    elif text == "Italy":
        return 17
    elif text == "Poland":
        return 18
    elif text == "Jamaica":
        return 19
    elif text == "Vietnam":
        return 20
    elif text == "Mexico":
        return 21
    elif text == "Portugal":
        return 22
    elif text == "Ireland":
        return 23
    elif text == "France":
        return 24
    elif text == "Dominican-Republic":
        return 25
    elif text == "Laos":
        return 26
    elif text == "Ecuador":
        return 27
    elif text == "Taiwan":
        return 28
    elif text == "Haiti":
        return 29
    elif text == "Columbia":
        return 30
    elif text == "Hungary":
        return 31
    elif text == "Guatemala":
        return 32
    elif text == "Nicaragua":
        return 33
    elif text == "Scotland":
        return 34
    elif text == "Thailand":
        return 35
    elif text == "Yugoslavia":
        return 36
    elif text == "El-Salvador":
        return 37
    elif text == "Trinadad&Tobago":
        return 38
    elif text == "Peru":
        return 39
    elif text == "Hong":
        return 40
    elif text == "Holand-Netherlands":
        return 41
    else:
        return 0

def numerize(name, column):
    name = name.strip()
    if column == 1:
        return workclass(name)
    elif column == 3:
        return education(name)
    elif column == 5:
        return maritalStatus(name)
    elif column == 6:
        return occupation(name)
    elif column == 7:
        return relationship(name)
    elif column == 8:
        return race(name)
    elif column == 9:
        return sex(name)
    elif column == 13:
        return nativeCountry(name)
    elif name == ">50K":
        return 1
    elif name == "<=50K":
        return 0
    else:
        return float(name)

def load_dataset(data='../data/adult.data'):
    """
    Loads and returns train and test datasets.
    """
    data = np.loadtxt(fname=data, delimiter=', ', dtype=str)
    counter = 0
    arr = [[0.0 for j in range(len(data[i]))] for i in range(len(data))]
    for i in range(len(data)):
        counter += 1
        #print(i)
        for j in range(len(data[i])) :
            arr[i][j] = numerize(data[i][j], j)
        #print(i)
        #print("Process Person " + str(counter))
    data = np.asarray(arr)
    y = data[:,data.shape[1]-1] #.astype(np.int)
    x = data[:,:data.shape[1]-1] #/ 255.0
    return (x, y)

class Layer:
    def __init__(self, num_notes, num_inputs_per_note):
        self.nodes = num_notes
        self.inputs = num_inputs_per_note
        self.weights = 2 * random.random((num_inputs_per_note, num_notes)) - 1

    def print_weights(self):
        print(self.self.weights)


class Network:
    def __init__(self, layer1, layer2):
        self.layers = []
        self.layer1 = layer1
        self.layer2 = layer2

    def add_layer(self, layer):
        self.layers.append(layer)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_data, training_outputs, num_iterations):
        # print("\tTraining...")
        for counter in range(num_iterations):
            if(counter%100 == 0):
                print("\tTrain Iteration: " + str(counter))
            output_from_layer_1, output_from_layer_2 = self.classify(training_data)

            #print(training_outputs.shape," VS ", output_from_layer_2.shape)
            layer2_error = training_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # if(counter<20):
            #     print(np.sum(np.abs(layer1_error)))
            #     print(np.sum(np.abs(layer2_error)))

            layer1_adjustment = training_data.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment

    def classify(self, inputs):
        output_from_layer1 = self.sigmoid( dot(inputs, self.layer1.weights) )
        output_from_layer2 = self.sigmoid( dot(output_from_layer1, self.layer2.weights) )

        return output_from_layer1, output_from_layer2

    def print_weights(self):
        print("\t Layer 1 ({} neurons, each with {} inputs): ".format(self.layer1.nodes, self.layer1.inputs))
        print(self.layer1.weights)
        print()
        print("\t Layer 2 ({} neuron, with {} inputs):".format(self.layer2.nodes, self.layer2.inputs))
        print(self.layer2.weights)
        print()

if __name__ == "__main__":
    random.seed(1)

    x, y = load_dataset()
    #x = map(float, x)
    #y = map(float, y)
    # print("========Dataset Size==========")
    # print(x, len(x))
    # print(y)
    # print("==============================")

    _x_ = []
    _y_ = []
    for i in range(20000):
        _x_.append(x[i])
        _y_.append(y[i])
    _x_ = np.asarray(_x_)
    _y_ = np.asarray(_y_)
    _y_ = _y_[:,np.newaxis]
    # print(len(_x_[0]))
    #
    # print("Creating Network...")
    # (x, y) => x neurons with y inputs each
    layer1 = Layer(8, 14)
    layer2 = Layer(1, 8)

    neural_network = Network(layer1, layer2)
    # print("Before: ")
    neural_network.print_weights()
    # print("Training Network...")
    neural_network.train(_x_, _y_, 10000)
    # print("\nAfter: ")
    neural_network.print_weights()

    # print("Testing...")
    correct, wrong = [0, 0], [0, 0]
    under, over = 0, 0
    last_index = len(x)-1

    for j in range(1,4):
        test_size = 500*j
        print("============Test for {} inputs============".format(test_size))
        for i in range(test_size):
            hidden, out = neural_network.classify(x[last_index-i])
            actual = y[last_index-i]
            if actual == 0:
                under +=1
            else:
                over += 1
            _out = round(out[0])
            if _out == actual: #correct classification
                if _out == 1: # correct classification of >50k
                    correct[0] += 1
                else: # correct classification of <=50k
                    correct[1] += 1
            else: #wrong classification
                if _out == 1: # wrong classification of >50k
                    wrong[0] += 1
                else: # wrong classification of <=50k
                    wrong[1] += 1

        print("<=50k: ", under)
        print(" >50k: ", over)
        print("\tCorrect\t| Wrong")
        print("under|\t{} \t| {}".format(correct[0],wrong[0]))
        print("over |\t{} \t| {}".format(correct[1],wrong[1]))
        print()
