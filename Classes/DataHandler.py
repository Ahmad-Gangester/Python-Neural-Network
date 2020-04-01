if(__name__ == "__main__"):
    exit()


class DataHandler:

        def __init__(self,path):
            self.__DataFolderPath=path
            self.__Topology=[]
            self.__numberOfTestCases=0
            self.__passNumber=0
            self.__Lines=[]


        def read(self):
            file=open(self.__DataFolderPath+"\\inputs.txt",'r')
            open(self.__DataFolderPath+"\\outputs.txt",'w+').write('')
            self.__Lines=file.readlines()
            str=self.__Lines.pop(0)
            str=str.replace("Topology: ",'')
            str=str.split(" ")
            self.__numberOfTestCases=int(str.pop())
            for s in str:
                self.__Topology.append(int(s))

        def nextInputs(self):
            str=self.__Lines.pop(0)
            file=open(self.__DataFolderPath+"\\outputs.txt",'a+')
            file.write(f"Pass {self.__passNumber}:\n{str}")
            str=str.replace("In: ",'')
            str=str.split(" ")
            Inputs=[]
            for s in str:
                Inputs.append(float(s))
            return Inputs

        def nextTargets(self):
            str=self.__Lines.pop(0)
            file=open(self.__DataFolderPath+"\\outputs.txt",'a+')
            file.write(f"{str}")
            str=str.replace("Target: ",'')
            str=str.split(" ")
            Targets=[]
            for s in str:
                Targets.append(float(s))
            self.__passNumber+=1
            return Targets

        def writeResults(self,Results,AvgError):
            file=open(self.__DataFolderPath+"\\outputs.txt",'a+')
            file.write("\nResults: ")
            for i,r in enumerate(Results):
                file.write(f"{i+1}:{r}")
            file.write(f"\nAvgError:{AvgError}\n\n")

        @property
        def Topology(self):
            return self.__Topology
        @property
        def NumberOfTestCases(self):
            return self.__numberOfTestCases
