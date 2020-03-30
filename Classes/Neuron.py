from dataclasses import dataclass
import random
import math

@dataclass
class Connection:
    Weight = None
    DeltaWeight =None



#Neuron Class

class Neuron:
    __outputWeights = []
    __outputValue =0
    __bias=1
    __Gradient = 0
    __traningRate = 0.20
    __Momentum = 0.001

    def __init__(self,numberOfOutputs,Index):
        self.__Index=Index
        for i in range(numberOfOutputs):
            self.__outputWeights.append(Connection())
            self.__outputWeights[i].Weight=self.__randomWight()
            print(f"Connection {i} in Neuron {Index} was created.")

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

    def feedForward(self,perviousLayer):
        sum=0.0
        for n in perviousLayer:
            sum+=n.outputValue * n.__getOutputWeightOf(self.__Index)
        sum+=self.__bias
        self.outputValue=self.__activationFunction(sum)


    def setOutputValue(self,outputValue):
        self.__outputValue=outputValue
    def getOutputValue(self):
        return self.__outputValue
    outputValue=property(getOutputValue,setOutputValue)
