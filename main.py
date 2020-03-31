from Classes.Network import Network
import random

def main():
    topology = [2,4,1]
    inputs=[]
    targets=[]
    net = Network(topology)
    for i in range(4000):
        inputs.clear()
        targets.clear()
        print(f"Pass: {i}")
        n1=random.randint(0,1)
        n2=random.randint(0,1)
        t=n1^n2
        inputs.append(n1)
        inputs.append(n2)
        targets.append(t)
        net.feedForward(inputs)
        print(f"inputs: {n1} {n2}")
        r=net.getResult()
        print(f"results: {r}")
        net.backPropagation(targets)
        print(f"target: {t}")
        print(f"Avg error : {net.AvgError}\n")

if(__name__ == "__main__"):
    main()
