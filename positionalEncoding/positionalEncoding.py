import math
import torch
import torch.nn as nn
from abc import ABC , abstractclassmethod

class positionalEncoding(ABC):
    """
    This class is to implement different tecniques of positional encodings , the encodings that will be implemented in this class is following
    
    ->Sinudoidal positional encoding
    ->Leared positional encoding
    ->Rotatory positional embedding (RoPE)
    ->Relative positional encoding
    """
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def forward(self ,  query , value , key , mask = None):
        pass


class SinudoinalPositionalEncoding(positionalEncoding , nn.Module):
    def forward(self, query, value, key, mask=None):
       pass
