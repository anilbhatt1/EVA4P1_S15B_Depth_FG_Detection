import torch
import torch.nn as nn
import torch.nn.functional as F

class Freezer():
    def freeze_mask_layers(self,model):
        print('Freezing Mask Layers')
        response = False
        for param in model.convA.parameters():
          print('XYZ')
          param.requires_grad = False
        for param in model.convB.parameters():
          param.requires_grad = False
        for param in model.convC.parameters():
          param.requires_grad = False
        for param in model.convD.parameters():
          param.requires_grad = False
        for param in model.convE.parameters():
          param.requires_grad = False
        #    
        for param in model.blck1.parameters():
          param.requires_grad = True
        for param in model.blck2.parameters():
          param.requires_grad = True
        for param in model.blck3.parameters():
          param.requires_grad = True
        for param in model.blck4.parameters():
          param.requires_grad = True
        for param in model.blck5.parameters():
          param.requires_grad = True  
        for param in model.blck6.parameters():
          param.requires_grad = True
        for param in model.blck7.parameters():
          param.requires_grad = True
        for param in model.blck8.parameters():
          param.requires_grad = True
        for param in model.convLast.parameters():
          param.requires_grad = True
        response = True  
        return response          

    def freeze_depth_layers(self,model):
        print('Freezing Depth Layers')
        response = False        
        for param in model.convA.parameters():
          print('ABCD')        
          param.requires_grad = True
        for param in model.convB.parameters():
          param.requires_grad = True
        for param in model.convC.parameters():
          param.requires_grad = True
        for param in model.convD.parameters():
          param.requires_grad = True
        for param in model.convE.parameters():
          param.requires_grad = True  
        #    
        for param in model.blck1.parameters():
          param.requires_grad = False
        for param in model.blck2.parameters():
          param.requires_grad = False
        for param in model.blck3.parameters():
          param.requires_grad = False
        for param in model.blck4.parameters():
          param.requires_grad = False
        for param in model.blck5.parameters():
          param.requires_grad = False  
        for param in model.blck6.parameters():
          param.requires_grad = False
        for param in model.blck7.parameters():
          param.requires_grad = False
        for param in model.blck8.parameters():
          param.requires_grad = False
        for param in model.convLast.parameters():
          param.requires_grad = False
        response = True  
        return response           