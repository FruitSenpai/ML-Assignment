
# coding: utf-8

# # ANN, by Seale

# In[2]:


import numpy as np
from numpy import exp, array, random, dot


# * First, we read in the data

# In[33]:


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
    


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


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


# In[38]:


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


# In[39]:


def sex(text):
    if text == "Female":
        return 1
    elif text == "Male":
        return 2
    else:
        return 0


# In[40]:


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


# In[50]:


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


# In[51]:


def load_dataset(data='./adult.data'):
    """
    Loads and returns train and test datasets.
    """
    data = np.loadtxt(fname=data, delimiter=', ', dtype=str)
    for i in data:
        for j in range(len(i)) :
            i[j] = numerize(i[j], j)
            
    y = data[:,data.shape[1]-1] #.astype(np.int)
    x = data[:,:data.shape[1]-1] #/ 255.0
    return (x, y)


# In[52]:


x, y = load_dataset()


# In[10]:


class Layer:
    def __init__(self, num_notes, num_inputs):
        self.weights = 2 * random.random((num_inputs, num_notes)) - 1
        
    def print_weights(self):
        print(self.self.weights)


# In[11]:


class Network:
    def __init__(self):
        random.seed(1)
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self):
        print("training")
    
    def classify(self, inputs):
        output_1 = sigmoid( dot(inputs, self.synaptic_weights) )

