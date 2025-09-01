import torch
from src.model import ConvE, IBNRSRQHandoverDecision
from src.utils import *
from src.datasetHelpers import construcLabel
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

def generateDataSet(graphIndicesPath: str, convEModel: ConvE):
    grIndexPropertiesJSON = loadJSONFromFile(graphIndicesPath)
    
    rsrqValues = [x for x in range(-40, 41)]
    
    trainingData = []
    for rsrq in rsrqValues:
        data = {}
        rsrqStr = str(rsrq)
        inputLabel = construcLabel('UERSRQ', rsrqStr)
        inputIndex = grIndexPropertiesJSON['elemLabelsToIndexMap'][inputLabel]
        inputTensor: Tensor = convEModel.emb_e(torch.LongTensor([inputIndex]))
        inputTensor = inputTensor.detach()
        
        outputTensor = [1, 0]
        if rsrq < 19:
            outputTensor = [0, 1]
        
        data['rsrqIndex'] = inputIndex
        data['rsrqValue'] = rsrq
        data['rsrqTensor'] = inputTensor
        data['hoDecision'] = torch.tensor(outputTensor, requires_grad=False).view(1,2)
        
        trainingData.append(data)
    return trainingData

class TrainRSRQHoData(Dataset):
    def __init__(self, graphIndicesPath: str, convEModel: ConvE):
        
        self.dataset = generateDataSet(graphIndicesPath, convEModel)
        self.n_samples = len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
def trainRSRQHODecision(epochs: int, graphIndicesPath: str, rsrqHOModelPath: str, convEModelPath: str):
    
    # load relevant json files
    graphIndicesJSON = loadJSONFromFile(graphIndicesPath)

    # load the convE model
    numElem = graphIndicesJSON['numElem']
    numRel = graphIndicesJSON['numRel']
    convEModel = ConvE(200, numElem, numRel)
    convEModel.load_state_dict(torch.load(convEModelPath))
    convEModel.eval()
    
    rsrqHandoverModel = IBNRSRQHandoverDecision()
    rsrqHandoverModel.train()
    rsrqHandoverModel.cuda()
    
    dataset = TrainRSRQHoData(graphIndicesPath, convEModel)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=10)
    opt = torch.optim.Adam(rsrqHandoverModel.parameters(), lr=0.003)
    for epoch in range(epochs):
        rsrqHandoverModel.train()
        for i, batchElems in enumerate(dataloader):
            opt.zero_grad()
            rsrqValue = batchElems['rsrqValue']
            inputTensor = batchElems['rsrqTensor']
            inputTensorCuda = inputTensor.cuda()
            output = batchElems['hoDecision']
            outputTensor = output.float().cuda()
            pred = rsrqHandoverModel.forward(inputTensorCuda)
            outputTensor = ((1-0.1)*outputTensor) + (0.01)
            loss = rsrqHandoverModel.loss(pred, outputTensor)
            
            loss.backward()
            opt.step()
            
            print(rsrqValue[0], pred[0].tolist())
            print('Epoch:{0}, loss: {1}', epoch, loss.cpu())
            
    rsrqHandoverModel.cpu()
    torch.save(rsrqHandoverModel.state_dict(), rsrqHOModelPath)