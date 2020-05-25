from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from kornia.losses import SSIM
from kornia.losses import DiceLoss

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

      def test_loss_calc(self,model, device, test_loader, optimizer, epoch, criterion1, criterion2, path_name, scheduler=None, img_save_idx =500):
          self.model        = model
          self.device       = device
          self.test_loader  = test_loader
          self.optimizer    = optimizer
          self.epoch        = epoch
          self.criterion1   = criterion1
          self.criterion1   = criterion2           
          self.scheduler    = scheduler
          self.path_name    = path_name
          self.img_save_idx = img_save_idx

          model.eval()  
          test_loss1, test_loss2, test_loss = 0, 0, 0
          pbar = tqdm(test_loader)
          cuda0 = torch.device('cuda:0')

          with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
              data['f1'] = data['f1'].to(device)
              data['f2'] = data['f2'].to(device)
              data['f3'] = data['f3'].to(device)
              data['f4'] = data['f4'].to(device)      
              data['f3O'] = torch.tensor(data['f3'],dtype= torch.int64, device= cuda0)      
            
              output = model(data)

              loss1 = criterion1(output[0], data['f3O'])
              loss2 = criterion2(output[1], data['f4'])
              loss  = 2*loss1 + loss2
              test_loss1 += loss1
              test_loss2 += loss2
              test_loss  += loss

              pbar.set_description(desc = f'{int(epoch)} {int(batch_idx)} tl={loss.item():.4f} tl1={loss1.item():.5f} tl2={loss2.item():.5f}')   
              
              if batch_idx % save == 0:
                  print('Test Epoch: {} [{}/{} ({:.0f}%)]\tTest_Loss: {:.6f} Mask_Loss: {:.5f} Dpth_Loss: {:.5f}'.format(
                        epoch, batch_idx * len(data), len(test_loader.dataset), (100. * batch_idx / len(test_loader)),
                        test_loss, test_loss1, test_loss2))
                  draw_and_save_tst(output[0].detach().cpu(), f'{path_name}Test_{epoch}_{batch_idx}_MP_{loss.item():.5f}.jpg')
                  draw_and_save_tst(data['f3'].detach().cpu(), f'{path_name}Test_{epoch}_{batch_idx}_MA_{loss.item():.5f}.jpg')
                  draw_and_save_tst(output[1].detach().cpu(), f'{path_name}Test_{epoch}_{batch_idx}_DP_{loss.item():.5f}.jpg')
                  draw_and_save_tst(data['f4'].detach().cpu(), f'{path_name}Test_{epoch}_{batch_idx}_DA_{loss.item():.5f}.jpg')
                  draw_and_save_tst(data['f1'].detach().cpu(), f'{path_name}Test_{epoch}_{batch_idx}_FGBG_{loss.item():.5f}.jpg')       
            
          test_loss /= len(test_loader.dataset)
          mask_loss = test_loss1/len(test_loader.dataset)
          dpth_loss = test_loss2/len(test_loader.dataset)
          return test_loss, mask_loss, dpth_loss     
              
      def draw_and_save_tst(tensors, name, figsize=(15,15), *args, **kwargs):
          try:
             tensors = tensors.detach().cpu()
          except:
             pass
          grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
          grid_image  = grid_tensor.permute(1, 2, 0)
          plt.figure(figsize = figsize)
          plt.imshow(grid_image)
          plt.xticks([])
          plt.yticks([])

          plt.savefig(name, bbox_inches='tight')
          plt.close()
         #plt.show()              