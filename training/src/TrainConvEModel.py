import torch
import numpy as np
from rdflib import Graph, URIRef
from rdflib.term import Node, URIRef
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from functools import reduce
from src.model import ConvE
from src.utils import *
from torch.utils.data import Dataset, DataLoader
import math

def generateDataset(graphFilePath: str):
    trainingData = []
    gr = getGraph(graphFilePath)
    grProperties = generateGraphProperties(gr)
    for elemKey, elemIndex in grProperties.elemLabelToIndexDict.items():
        for relKey, relIndex in grProperties.relLabelToIndexDict.items():
            data = {}
            data['e1'] = torch.LongTensor([elemIndex])
            data['rel'] = torch.LongTensor([relIndex])
            data['e2_multi_binary'] = torch.tensor(grProperties.labeledE2MultiVector[(elemKey, relKey)])
            
            trainingData.append(data)
            
    return trainingData
    
class GraphData(Dataset):
    def __init__(self, graphFilePath: str):
        self.dataset = generateDataset(graphFilePath)
        self.n_samples = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

def trainConvE(graphFilePath: str, graphIndicesFilePath: str, epochs: int, finalModelPath: str):    
    gr = getGraph(graphFilePath)
    grProperties = generateGraphProperties(gr)
    grIndexProperties = generateGraphIndexProperties(grProperties)
    
    grIndexPropertiesJSON = json.dumps(grIndexProperties, cls=EnhancedJSONEncoder)
    
    numElem = grProperties.numElem
    numRel = grProperties.numRel
    
    model = ConvE(200, numElem, numRel)
    model.init()
    model.cuda()
        
    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))
    
    dataset = GraphData(graphFilePath)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=10)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(epochs):
        model.train()
        for i, str2var in enumerate(dataloader):
#            for relKey, relIndex in grProperties.rel.items():
            opt.zero_grad()
            e1 = str2var['e1'].cuda()
            rel = str2var['rel'].cuda()
            e2_multi = str2var['e2_multi_binary'].float().cuda()
            e2_multi = ((1.0-0.1)*e2_multi) + (1.0/e2_multi.size(1))
            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            # loss = model.loss(e2_multi, pred)
            loss.backward()
            opt.step()
                
            print("Epoch:{0}, loss: {1}", epoch, loss.cpu())
    model.cpu()
    with open(graphIndicesFilePath, 'w') as f:
        json.dump(grIndexPropertiesJSON, f)
    print("saving to {0}", finalModelPath)
    print(model.emb_e(torch.LongTensor([1])))
    torch.save(model.state_dict(), finalModelPath)
    
