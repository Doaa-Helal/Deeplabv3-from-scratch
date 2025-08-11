import torch
import torch.nn as nn 
class DiceBCELoss(nn.Module):
    def __init__(self,weight=None,size_avg=True):
        super(DiceBCELoss,self).__init__()
        self.bce_loss=nn.BCEWithLogitsLoss()

    def forward(self,inputs,targets,smooth=1):
        bce=self.bce_loss(inputs,targets) 
        
        inputs=torch.sigmoid(inputs)
        
        inputs=inputs.view(-1)
        targets=targets.view(-1)

        intersection=(inputs*targets).sum()
        dice_loss=1-(2.0*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)
        Dice_BCE=dice_loss+bce
        
        return Dice_BCE
        
        