from .Neuron import Neuron
#Network Class
class Network:
    __Layers = []

    def __init__(self,topology):
        numLayers=len(topology)
        for layerNum in range(numLayers):
            self.__Layers.append([])
            numOutputs = 0 if layerNum == len(topology)-1 else topology[layerNum+1]
            for neuronNum in range(topology[layerNum]):
                print(f"Neuron: {neuronNum} in Layer: {layerNum} was born.")
                self.__Layers[-1].append(Neuron(numOutputs,neuronNum))

    def feedForward(self,inputValues):

        for i in range(len(inputValues)):
            self.__Layers[0][i].outputValue=inputValues[i]
        for i in range(1,len(self.__Layers)):
            perviousLayer=self.__Layers[i-1]
            for j in range(len(self.__Layers[i])):
                self.__Layers[i][j].feedForward(perviousLayer)
