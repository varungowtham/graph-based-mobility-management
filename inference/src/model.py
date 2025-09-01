import torch
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

class IBNRSRQHandoverDecision(nn.Module):
    def __init__(self):
        super(IBNRSRQHandoverDecision, self).__init__()
        self.firstLayer = nn.Linear(200,200)
        self.secondLayer = nn.Linear(200,15000)
        self.finalLayer = nn.Linear(15000, 2)
        self.hiddenDropout = torch.nn.Dropout(0.3)
#        self.bn = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', Parameter(torch.zeros(2)))
        self.loss = nn.BCELoss()

    def forward(self, maxIngressRate: Tensor) -> Tensor:
        x = self.firstLayer(maxIngressRate)
#        x = self.bn(x)
        x = F.relu(x)
        x = self.secondLayer(x)
        x = self.hiddenDropout(x)
#        x = self.bn(x)
        x = F.relu(x)
        x = self.finalLayer(x)
#        x = self.bn(x)
        x += self.b.expand_as(x)
        x = F.sigmoid(x)
        return x

# ConvE model copied from Tim Dettmers
class ConvE(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2) # input_drop
        self.hidden_drop = torch.nn.Dropout(0.3) # hidden_drop
        self.feature_map_drop = torch.nn.Dropout2d(0.2) # feat_drop
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 20 # embedding_shape1
        self.emb_dim2 = embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(9728, embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e.weight.data, gain=50.0)
        xavier_normal_(self.emb_rel.weight.data, gain=50.0)
    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
