import torch
import torch.nn as nn
import math

class imagePatching(nn.Module):
    def __init__(self,imgSize = 224 , patchSize = 16 , inChannel = 3 , dModel = 768):
        super().__init__()
        assert imgSize%patchSize == 0
        self.numPatches = (imgSize//patchSize)**2

        self.projection = nn.Conv2d(
            in_channels=inChannel,
            out_channels=dModel,
            kernel_size=patchSize,
            stride=patchSize
        )

    def forward(self , x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1,2)

        return x

class multiHeadAttention(nn.Module):
    def __init__(self , dModel , numHead , dropout = 0.1):
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
    def forward(self , x):
        batchSize , sequenceLen , _ = x.shape
        k = self.wK(x)
        v = self.wV(x)
        q = self.wQ(x)

        q = q.reshape(batchSize , sequenceLen , self.numHead , self.headDim).transpose(1, 2)
        v = v.reshape(batchSize , sequenceLen , self.numHead , self.headDim).transpose(1, 2)
        k = k.reshape(batchSize , sequenceLen , self.numHead , self.headDim).transpose(1, 2)

        score = torch.matmul(q , k.transpose(-2,-1))/math.sqrt(self.headDim)
        attention = torch.softmax(score , dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention , v).transpose(1,2)
        output = output.contiguous()
        output = output.reshape(batchSize , sequenceLen , self.dModel)

        return self.wO(output)
    

class feedForwardNN(nn.Module):
    def __init__(self, dModel , dff , dropout = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dModel , dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff , dModel),
            nn.Dropout(dropout)
        )

    def forward(self , x):
        return self.network(x)

class encoderLayer(nn.Module):
    def __init__(self, dModel , dff , numHead , dropout = 0.1):
        super().__init__()
        self.attention = multiHeadAttention(dModel , numHead , dropout)
        self.feefForwardNN = feedForwardNN(dModel , dff , dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)

    def forward(self ,x):
        attentionOutput = self.attention(self.norm1(x))
        x = x+attentionOutput
        ffnnOutPut = self.feefForwardNN(self.norm2(x))
        x = x+ffnnOutPut
        return x
    

class visionTransformer(nn.Module):
    def __init__(self,imageSize = 224 , patchSize = 16 , inChannel = 3 , numclasses = 10 , dModel = 768 , numHead = 12 , dff = 3072 , numLayer = 12 , dropout = 0.1):
        super().__init__()
        self.patchEmbadding = imagePatching(imageSize , patchSize , inChannel , dModel)
        numPatches = self.patchEmbadding.numPatches
        self.clsTocken = nn.Parameter(torch.zeros(1,1,dModel))
        self.posEmbedding = nn.Parameter(torch.zeros(1 , numPatches+1 , dModel))
        self.dropout = nn.Dropout(dropout)
        self.encoderLayer = nn.ModuleList([encoderLayer(dModel , dff , numHead , dropout) for _ in range(numLayer)])
        self.norm = nn.LayerNorm(dModel)
        self.classificationHead = nn.Linear(dModel , numclasses)

        nn.init.trunc_normal_(self.clsTocken , std=0.02)
        nn.init.trunc_normal_(self.posEmbedding , std=0.02)

    def forward(self,x):
        batchSize = x.size(0)
        x = self.patchEmbadding(x)
        clsTocken = self.clsTocken.expand(batchSize , -1 , -1)
        x = torch.cat((clsTocken , x) , dim=1)
        x = x+self.posEmbedding
        x = self.dropout(x)

        for layer in self.encoderLayer:
            x = layer(x)

        x = self.norm(x)

        clsOutput = x[: , 0 , :]
        logits = self.classificationHead(clsOutput)

        return logits
    

def main():
    torch.manual_seed(42)

    model = visionTransformer(imageSize=224,patchSize=16,inChannel=3,numclasses=10,dModel=768,numHead=12,dff=3072,numLayer=12,dropout=0.1)
    batchSize = 2
    images = torch.randn(batchSize, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", logits.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
   main()