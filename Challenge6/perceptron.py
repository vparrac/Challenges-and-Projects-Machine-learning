import numpy as np


class NeuralLayer():
    def __init__(self,conexionNumber,neuronsNumber,activationFunction):
        self.activationFunction = activationFunction
        self.bias = (np.random.rand(1,neuronsNumber)[0])*2-1
        self.weigths = np.random.rand(conexionNumber,neuronsNumber)*2-1

### Topology: An Array of [neurons] per layer [10,2,1]
### ActivationFunctiosn an array with the af per layer 
def createNeuralNet(topology,activationFunctions):
    neuralNet = []
    for layer in range(len(topology)-1):
        neuralNet.append(NeuralLayer(topology[layer],topology[layer+1],activationFunctions[layer]))
        
    return neuralNet

