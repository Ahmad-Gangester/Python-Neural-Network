from Classes.Network import Network

def main():
    topology = [2,4,1]
    net = Network(topology)
    inputs=[1,1]
    net.feedForward(inputs)


if(__name__ == "__main__"):
    main()
