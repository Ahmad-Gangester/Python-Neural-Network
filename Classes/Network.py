if(__name__ == "__main__"):
    exit()

from .Neuron import Neuron
import math
#Network Class
class Network:

    #region Class Variables
    __numberToAvg=100
    #endregion

    #region Constructor
    def __init__(self,topology):
        self.__Layers=[]
        self.__Error=0.0
        self.__AvgError=0.0
        numLayers=len(topology)
        for layerNum in range(numLayers):
            # print(f"Layer {layerNum} =>",end=" ")
            self.__Layers.append([])
            numOutputs = 0 if layerNum == len(topology)-1 else topology[layerNum+1]
            for neuronNum in range(topology[layerNum]):
                # print(f"Neuron {neuronNum} =>")
                x=Neuron(numOutputs,neuronNum)
                self.__Layers[-1].append(x)
        # print("\n\n")
    #endregion

    #region Public Functions

    def feedForward(self,inputValues):
        for i in range(len(inputValues)):
            self.__Layers[0][i].outputValue=inputValues[i]
            # print(f"Layer 0 => Neuron {i} => output={self.__Layers[0][i].outputValue}")
        for i in range(1,len(self.__Layers)):
            # print(f"Layer {i} =>")
            perviousLayer=self.__Layers[i-1]
            for j in range(len(self.__Layers[i])):
                self.__Layers[i][j].feedForward(perviousLayer)
                # print(f"Neuron {j} => output = {self.__Layers[i][j].outputValue}")
            # print("\n")

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
            # print(f"Layer 2 => Neuron {i} => Gradient={self.__Layers[-1][i].Gradient}")

        for layerNum in range(len(self.__Layers)-2,0,-1):
            hiddenLayer=self.__Layers[layerNum]
            nextLayer=self.__Layers[layerNum+1]
            # print(f"Layer {layerNum} =>")
            for i,_ in enumerate(hiddenLayer):
                hiddenLayer[i].hiddenGradient(nextLayer)
                # print(f"Neuron {i} => Gradient = {self.__Layers[layerNum][i].Gradient}")
            # print("\n")
        for layerNum,_ in reversed(list(enumerate(self.__Layers))):
            if(layerNum==0):
                break
            # layer=self.__Layers[layerNum]
            # perviousLayer=self.__Layers[layerNum-1]
            for neuronNum,_ in enumerate(self.__Layers[layerNum]):
                self.__Layers[layerNum][neuronNum].updateWeights(self.__Layers[layerNum-1])

    def getResult(self):
        Results=[]
        for r in self.__Layers[-1]:
            Results.append(r.outputValue)
        return Results
    #endregion

    #region Private Functions

    #endregion

    #region Gets and Sets
    @property
    def AvgError(self):
        return self.__AvgError

    #endregion
