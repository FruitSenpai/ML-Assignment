from numpy import exp, array, random, dot

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# print "======Training Inputs======"
# print training_set_inputs

training_set_outputs = array([[0, 1, 1, 0]]).T
# print "======Training Outputs======"
# print training_set_outputs

random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
# print "======Synaptic Weights======"
# print synaptic_weights

for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

# print
# print synaptic_weights

# print
print "======Result======"
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))

print "======Error======"
print (training_set_outputs-output)
