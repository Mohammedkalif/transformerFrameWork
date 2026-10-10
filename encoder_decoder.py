import math
import torch
import torch.nn as nn

class positionalEncoding(nn.Module):
    def __init__(self, dModle , maxLen = 50000):
        super().__init__()
        pe = torch.zeros(maxLen , dModle)

        positions = torch.arange(maxLen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , dModle , 2)*(-math.log(10000.0)/dModle))
        pe[: , 0::2] = torch.sin(positions*div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe" , pe.unsqueeze(0))

    def forward(self , x):
        return x + self.pe[: , :x.size(1) , :]

class multiHeadAttentionLayer(nn.Module):
    def __init__(self,dModel , numHead , dropout = 0.1):
        super().__init__()
        assert dModel%numHead == 0
        self.dModel = dModel
        self.numHead = numHead
        self.headDim = dModel//numHead

        self.wQ = nn.Linear(dModel , dModel)
        self.wK = nn.Linear(dModel , dModel)
        self.wV = nn.Linear(dModel , dModel)
        self.wO = nn.Linear(dModel , dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self , query , key , value , mask = None):
                batchSize = query.size(0)
                q = self.wQ(query)
                k = self.wK(key)
                v = self.wV(value)
    
                q = q.view(batchSize , -1 , self.numHead , self.headDim).transpose(1,2)
                k = k.view(batchSize , -1 , self.numHead , self.headDim).transpose(1,2)
                v = v.view(batchSize , -1 , self.numHead , self.headDim).transpose(1,2)
    
                score = torch.matmul(q , k.transpose(-2,-1))
                score = score/math.sqrt(self.headDim)
    
                if mask is not None:
                    score = score.masked_fill(mask , float('-inf'))
    
                weights = torch.softmax(score , dim=-1)
                weights = self.dropout(weights)
    
                outPut = torch.matmul(weights , v)
                outPut = outPut.transpose(2,1).contiguous()
                outPut = outPut.view(batchSize , -1 , self.dModel)
    
                return self.wO(outPut)

class feedForwardNN(nn.Module):
    def __init__(self, dModel , dff , dropout = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dModel , dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff , dModel)
        )

    def forward(self , x):
        return self.network(x)

class EncoderLayer(nn.Module):
    def __init__(self, dModel , numHead , dff , dropout = 0.1):
        super().__init__()
        self.selfAttention = multiHeadAttentionLayer(dModel , numHead , dropout)
        self.feedForward = feedForwardNN(dModel , dff , dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x , srcMask = None):
        attentionOutput = self.selfAttention(x,x,x,srcMask)
        x = self.norm1(x+self.dropout(attentionOutput))
        ffOutput = self.feedForward(x)
        x = self.norm2(x + self.dropout(ffOutput))

        return x

class DecoderLayer(nn.Module):
    def __init__(self , dModel , numHead , dff , dropout = 0.1):
        super().__init__()
        self.selfAttention = multiHeadAttentionLayer(dModel , numHead , dropout)
        self.crossAttention = multiHeadAttentionLayer(dModel , numHead , dropout)
        self.feedForward = feedForwardNN(dModel , dff , dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.norm3 = nn.LayerNorm(dModel)

        self.dropout = nn.Dropout(dropout)

    def forward(self , x , encoderOutput , tgtMask = None):
        attentionOutput =  self.selfAttention(x,x,x,tgtMask)
        x = self.norm1(x+self.dropout(attentionOutput))

        crossOutput = self.crossAttention(x , encoderOutput , encoderOutput)
        x = self.norm2(x+self.dropout(crossOutput))

        ffOutput = self.feedForward(x)
        x = self.norm3(x + self.dropout(ffOutput))

        return x

class Encoder(nn.Module):
    def __init__(self , vocbSize , dModel , numHeads , dff , numLayers , dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocbSize , dModel)
        self.positionalEncoding = positionalEncoding(dModel)
        self.encoderLayers = nn.ModuleList(
            [EncoderLayer(dModel , numHeads , dff , dropout) for _ in range(numLayers)] 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self , src , srcMask = None):
        x = self.embedding(src)*math.sqrt(self.embedding.embedding_dim)
        x = self.positionalEncoding(x)
        x = self.dropout(x)

        for layer in self.encoderLayers:
            x = layer(x , srcMask)

        return x

class Decoder(nn.Module):
    def __init__(self, vocabSize , dModel , numHead , dff, numLayers , dropout = 0.1):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize , dModel)
        self.positionalEncoding = positionalEncoding(dModel)
        self.decoderLayers = nn.ModuleList(
            [DecoderLayer(dModel , numHead , dff , dropout) for _ in range(numLayers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self , tgt , encoderOutput , tgtMask = None):
        x = self.embeddings(tgt)*math.sqrt(self.embeddings.embedding_dim)
        x = self.positionalEncoding(x)
        x = self.dropout(x)

        for layer in self.decoderLayers:
            x = layer(x , encoderOutput , tgtMask)

        return x
class Transformer(nn.Module):
    def __init__(self, scrVocabSize , tgtVocabSize , dModel = 128 , numHead = 4 , dff = 512 , numEncoderLayer = 3 , numDecoderLayer = 3 , dropout = 0.1):
        super().__init__()
        self.encoder = Encoder(scrVocabSize , dModel , numHead , dff , numEncoderLayer , dropout)
        self.decoder = Decoder(tgtVocabSize , dModel , numHead , dff , numDecoderLayer , dropout)
        self.output = nn.Linear(dModel , tgtVocabSize)
    def generateCasualMask(self , seqLen , device):
        return torch.triu(torch.ones(seqLen , seqLen , device=device , dtype=torch.bool) , diagonal=1)
    
    def forward(self , src , tgt):
        tgtMask = self.generateCasualMask(tgt.size(1) , tgt.device)
        encoderOutput = self.encoder(src)
        decoderOutput = self.decoder(tgt , encoderOutput , tgtMask)
        logits = self.output(decoderOutput)

        return logits

def main():
    torch.manual_seed(42)
    srcVocabSize = 1000
    tgtVocabSize = 1200

    model = Transformer(
        srcVocabSize , tgtVocabSize
    )

    batchSize = 2
    srcLen = 10
    tgtLen = 7

    src = torch.randint(0,srcVocabSize , (batchSize , srcLen))
    tgt = torch.randint(0 , tgtVocabSize , (batchSize , tgtLen))

    model.eval()
    with torch.no_grad():
        logits = model(src , tgt)

    print("Source:" , src.shape)
    print("Target:" , tgt.shape)
    print("Logits:" , logits.shape)
    print("Number of parameters:" , sum(p.numel() for p in model.parameters()))
    print("Encoder:" , len(model.encoder.encoderLayers))
    print("Decoder:" , len(model.decoder.decoderLayers))

if __name__ == "__main__":
    main()