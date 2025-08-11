import torch 
import torch.nn as nn

class IOU(nn.Module):
    def __init__(self,weight=None,size_avg=True):
        super(IOU,self).__init__()
    def forward(self,inputs,targets,smooth=1):
        inputs=torch.sigmoid(inputs)

        inputs=inputs.view(-1)
        targets=targets.view(-1)
        
        intersection=(inputs*targets).sum()
        total=(inputs+targets).sum()
        union=total-intersection

        IOU=(intersection+smooth)/(union+smooth)

        return IOU
        