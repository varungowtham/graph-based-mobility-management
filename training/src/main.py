from src.TrainConvEModel import trainConvE

graphPath = "graphs/ibn_demo.ttl"
graphIndicesPath = "graphs/graphIndices.json"
convEModel = "models/conve-0.1.pth"

if __name__ == '__main__':
    trainConvE(graphPath, graphIndicesPath, 50, convEModel)