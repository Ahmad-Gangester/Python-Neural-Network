if(__name__ == "__main__"):
    exit()

from dataclasses import dataclass
import random
import math

@dataclass
class Connection:
    Weight = 0.0
    DeltaWeight =0.0



#Neuron Class

class Neuron:

    #region Variables
    __outputWeights = []
    __outputValue =0.0
    __bias=1.0
    __Gradient = 0.0
    __traningRate = 0.20
    __Momentum = 0.001
    #endregion

    #region Constructor
    def __init__(self,numberOfOutputs,Index):
        self.__Index=Index
        for i in range(numberOfOutputs):
            self.__outputWeights.append(Connection())
            self.__outputWeights[i].Weight=self.__randomWight()
            #endregion

    #region Private Functions

    @staticmethod
    def __randomWight():
        return random.uniform(-0.5,0.5)

    @staticmethod
    def __activationFunction(x):
        return math.tanh(x)

    @staticmethod
    def __activationFunctionDir(x):
        return 1-x*x

    def __getOutputWeightOf(self,i):
        return self.__outputWeights[i].Weight

    @staticmethod
    def __SumDOW(nextLayer):
        sum=0.0
        for i,n in enumerate(nextLayer):
            sum+=n.__outputWeights[i].Weight * n.__Gradient
        return sum

    #endregion

    #region Public Functions
    def feedForward(self,perviousLayer):
        sum=0.0
        for n in perviousLayer:
            sum+=n.outputValue * n.__getOutputWeightOf(self.__Index)
        sum+=self.__bias
        self.outputValue=self.__activationFunction(sum)

    def outputGradient(self,targetValue):
        Error=targetValue-self.outputValue
        self.__Gradient=Error*self.__activationFunctionDir(self.__outputValue)

    def hiddenGradient(self,nextLayer):
        dow=self.__SumDOW(nextLayer)
        self.__Gradient=dow*self.__activationFunctionDir(self.__outputValue)

    def updateWeights(self,perviousLayer):
        for n,neuron in enumerate(perviousLayer):
            oldDelta=neuron.__outputWeights[self.__Index].DeltaWeight
            newDelta= neuron.outputValue * self.__Gradient * self.__traningRate + self.__Momentum * oldDelta
            perviousLayer[n].__outputWeights[self.__Index].DeltaWeight=newDelta
            perviousLayer[n].__outputWeights[self.__Index].Weight+=newDelta

    #endregion

    #region Gets and Sets
    def setOutputValue(self,outputValue):
        self.__outputValue=outputValue
    def getOutputValue(self):
        return self.__outputValue
    outputValue=property(getOutputValue,setOutputValue)
    @property
    def Gradient(self):
        return self.__Gradient
    #endregion