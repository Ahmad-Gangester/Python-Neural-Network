if(__name__ == "__main__"):
    exit()

from .DataHandler import DataHandler
from .Network import Network

class Trainer:

    def __init__(self,path):
        self.__Data=DataHandler(path)

    def Train(self):
        self.__Data.read()
        self.__Topology=self.__Data.Topology
        net = Network(self.__Topology)
        for _ in range (self.__Data.NumberOfTestCases):
            Inputs=self.__Data.nextInputs()
            Targets=self.__Data.nextTargets()
            net.feedForward(Inputs)
            Results=net.getResult()
            self.__Data.writeResults(Results,net.AvgError)
            net.backPropagation(Targets)