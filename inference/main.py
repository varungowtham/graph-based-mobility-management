import torch
from typing import List
from src.model import ConvE, IBNRSRQHandoverDecision
from src.utils import loadJSONFromFile

graphPath = "graphs/ibn_demo.ttl"
graphIndicesPath = "graphs/graphIndices.json"
convEModelPath = "models/conve-0.1.pth"
rsrqHOModelPath = "models/rsrqHOModel.pth"

def construcLabel(prefix: str, val: str) -> str:
    return "http://purl.org/toco/" + prefix + "_" + val + "_dB"


def runRSRQHOInference(rsrq: int):
    
    # load relevant json files
    graphIndicesJSON = loadJSONFromFile(graphIndicesPath)
    
    rsrqStr = str(rsrq) # integer string
    rsrqLabel = construcLabel('UERSRQ', rsrqStr) # Entity label
    
    # load the convE model
    numElem = graphIndicesJSON['numElem']
    numRel = graphIndicesJSON['numRel']
    convEModel = ConvE(200, numElem, numRel)
    convEModel.load_state_dict(torch.load(convEModelPath))
    convEModel.eval()
    convEModel.cpu()
    
    rsrqHOModel = IBNRSRQHandoverDecision()
    rsrqHOModel.load_state_dict(torch.load(rsrqHOModelPath))
    rsrqHOModel.eval()
    rsrqHOModel.cpu()
    
    rsrqIndex = graphIndicesJSON['elemLabelsToIndexMap'][rsrqLabel]
    rsrqEmbedding: torch.Tensor = convEModel.emb_e(torch.LongTensor([rsrqIndex])).cpu()
    
    prediction = rsrqHOModel.forward(rsrqEmbedding)
    print(prediction.size())
    prediction = prediction.tolist()[0]
    print(prediction)
    decision = prediction.index(max(prediction))
    
    if decision == 0:
        print("No Handover")
    else:
        print("Perform Handover")

if __name__=='__main__':
    runRSRQHOInference(-21)