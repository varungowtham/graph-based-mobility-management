from src.TrainConvEModel import trainConvE
from src.train_ho_model import trainRSRQHODecision

graphPath = "graphs/ibn_demo.ttl"
graphIndicesPath = "graphs/graphIndices.json"
convEModelPath = "models/conve-0.1.pth"
rsrqHOModelPath = "models/rsrqHOModel_T13.pth"

if __name__ == '__main__':
    # trainConvE(graphPath, graphIndicesPath, 50, convEModel)
    trainRSRQHODecision(1000, graphIndicesPath, rsrqHOModelPath, convEModelPath)