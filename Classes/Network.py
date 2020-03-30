from .Neuron import Neuron
import math
#Network Class
class Network:

    #region Variables
    __Layers = []
    __Error=None
    __AvgError=None
    __numberToAvg=100
    #endregion

    #region Constructor
    def __init__(self,topology):
        numLayers=len(topology)
        for layerNum in range(numLayers):
            self.__Layers.append([])
            numOutputs = 0 if layerNum == len(topology)-1 else topology[layerNum+1]
            for neuronNum in range(topology[layerNum]):
                print(f"Neuron: {neuronNum} in Layer: {layerNum} was born.")
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
        # for i in range(len(outputLayer)):
        #     delta=outputLayer[i].outputValue -targetValues[i]
        #     self.__Error+=delta**2
        for(neuron,target) in zip(outputLayer,targetValues):
            delta=neuron.outputValue-target
            self.__Error+=delta**2
        self.__Error/=len(outputLayer)
        self.__Error=math.sqrt(self.__Error)
        self.__AvgError=(self.__AvgError * self.__numberToAvg + self.__Error)/(self.__numberToAvg + 1.0)

        for(neuron,target) in zip(outputLayer,targetValues):
            neuron.outputGradient(target)

        for layerNum in range(len(self.__Layers)-2,-1,-1):
            hiddenLayer=self.__Layers[layerNum]
            nextLayer=self.__Layers[layerNum+1]
            for neuron in hiddenLayer:
                neuron.hiddenGradient(nextLayer)
        for layerNum,l in reversed(list(enumerate(self.__Layers))):
            layer=l
            perviousLayer=self.__Layers[layerNum-1]
            for neuron in layer:
                neuron.updateWeights(perviousLayer)

    #endregion

    #region Private Functions

    #endregion

    #region Gets and Sets
    @property
    def AvgError(self):
        return self.__AvgError

    #endregion
