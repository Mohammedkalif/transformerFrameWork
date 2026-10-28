import math
import torch
import torch.nn as nn
from abc import ABC , abstractmethod

class positionalEncoding(nn.Module , ABC):
    """
    This class is to implement different tecniques of positional encodings , the encodings that will be implemented in this class is following
    
    ->Sinudoidal positional encoding
    ->Leared positional encoding
    """
    def __init__(self, dModel, maxLen, dropout=0.1):
        super().__init__()

        self.dModel = dModel
        self.maxLen = maxLen
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self , x):
        pass



class SinudoinalPositionalEncoding(positionalEncoding):
    def __init__(self, dModel, maxLen, dropout=0.1):
        super().__init__(dModel , maxLen , dropout)
        position = torch.arange(maxLen , dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0 , dModel , 2).float()*(-math.log(10000.0) / dModel))

        pe = torch.zeros(maxLen , 1 , dModel)
        pe[: , 0 , 0::2] = torch.sin(position*divTerm)
        pe[: , 0 , 1::2] = torch.cos(position*divTerm)

        self.register_buffer("pe" , pe)
    def forward(self, x):
       x = x+self.pe[:x.size(0)]
       return self.dropout(x)

    
class LearnedPositionalEncodig(positionalEncoding):
    def __init__(self, dModel, maxLen, dropout=0.1):
        super().__init__(dModel, maxLen, dropout)

        self.positionalEmbedding = nn.Embedding(maxLen , dModel)
    def forward(self,x):
        seqLen = x.size(0)
        positions = torch.arange(seqLen , device=x.device)

        posEmbeddings = self.positionalEmbedding(positions).unsqueeze(1)
        x = x+posEmbeddings

        return self.dropout(x)
