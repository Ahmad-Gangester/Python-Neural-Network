if(__name__ == "__main__"):
    exit()

from .Neuron import Neuron
import math
#Network Class
class Network:

    #region Variables
    __Layers = []
    __Error=0.0
    __AvgError=0.0
    __numberToAvg=100
    #endregion

    #region Constructor
    def __init__(self,topology):
        numLayers=len(topology)
        for layerNum in range(numLayers):
            self.__Layers.append([])
            numOutputs = 0 if layerNum == len(topology)-1 else topology[layerNum+1]
            for neuronNum in range(topology[layerNum]):
                self.__Layers[-1].append(Neuron(numOutputs,neuronNum))
    #endregion

    #region Public Functions

    def feedForward(self,inputValues):
        for i in range(len(inputValues)):
            self.__Layers[0][i].outputValue=inputValues[i]
        for i in range(1,len(self.__Layers)):
            perviousLayer=self.__Layers[i-1]
            for j in range(len(self.__Layers[i])):
                self.__Layers[i][j].feedForward(perviousLayer)

    def backPropagation(self,targetValues):
        self.__Error=0.0
        outputLayer = self.__Layers[-1]
        for(neuron,target) in zip(outputLayer,targetValues):
            delta=neuron.outputValue-target
            self.__Error+=delta**2
        self.__Error/=len(outputLayer)
        self.__Error=math.sqrt(self.__Error)
        self.__AvgError=(self.__AvgError * self.__numberToAvg + self.__Error)/(self.__numberToAvg + 1.0)

        for (i,_),target in zip(enumerate(outputLayer),targetValues):
            outputLayer[i].outputGradient(target)

        for layerNum in range(len(self.__Layers)-2,-1,-1):
            hiddenLayer=self.__Layers[layerNum]
            nextLayer=self.__Layers[layerNum+1]
            for i,_ in enumerate(hiddenLayer):
                hiddenLayer[i].hiddenGradient(nextLayer)
        for layerNum,_ in reversed(list(enumerate(self.__Layers))):
            if(layerNum==0):
                break
            layer=self.__Layers[layerNum]
            perviousLayer=self.__Layers[layerNum-1]
            for neuronNum,_ in enumerate(layer):
                layer[neuronNum].updateWeights(perviousLayer)

    def getResult(self):
        return self.__Layers[-1][0].outputValue
    #endregion

    #region Private Functions

    #endregion

    #region Gets and Sets
    @property
    def AvgError(self):
        return self.__AvgError

    #endregion
