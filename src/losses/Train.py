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

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Train_loss:
    
    def draw_show_and_save(tensors, name, figsize=(15,15), *args, **kwargs):
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
        plt.show()
        
    def draw_and_save(tensors, name, figsize=(15,15), *args, **kwargs):
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
    

    def train_loss_calc(self,model, device, train_loader, optimizer, epoch, criterion1, criterion2, batch_size, path_name,path_model_save,
                        scheduler=None, model_save_idx=500, img_save_idx=500,maxlr=0):
                        
          self.model        = model
          self.device       = device
          self.train_loader = train_loader
          self.optimizer    = optimizer
          self.epoch        = epoch
          self.criterion1   = criterion1
          self.criterion2   = criterion2
          self.scheduler    = scheduler
          self.model_save_idx    = model_save_idx
          self.img_save_idx      = img_save_idx
          self.maxlr        = maxlr
          self.batch_size   = batch_size
          self.path_name    = path_name
          self.path_model_save = path_model_save
        
          model.train()
          train_loss1, train_loss2, train_loss = 0, 0, 0
          pbar = tqdm(train_loader)
          num_batches = len(train_loader.dataset)/batch_size
          cuda0 = torch.device('cuda:0')           

          for batch_idx, data in enumerate(pbar):
            data['f1'] = data['f1'].to(device)
            data['f2'] = data['f2'].to(device)
            data['f3'] = data['f3'].to(device)
            data['f4'] = data['f4'].to(device)
            data['f3O'] = torch.tensor(data['f3'],dtype= torch.int64, device= cuda0)

            optimizer.zero_grad()
            output = model(data)

            loss1 = criterion1(output[0], data['f3O'])
            loss2 = criterion2(output[1], data['f4'])
            loss  = 2*loss1 + loss2
            train_loss1 += loss1
            train_loss2 += loss2
            train_loss  += loss            
            

            pbar.set_description(desc = f'{int(epoch)} {int(batch_idx)} l={loss.item():.4f} l1={loss1.item():.5f} l2={loss2.item():.5f}')   
            loss.backward()
            optimizer.step()
            
            saved = False
            if batch_idx == 0 or batch_idx == int(num_batches-1):
                print('Train Epoch: {} Batch_ID: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f} Mask_Loss: {:.5f} Dpth_Loss: {:.5f}'.format(
                       epoch, batch_idx, batch_idx * batch_size, len(train_loader.dataset),
                       (100. * batch_idx / len(train_loader)),
                       loss, loss1, loss2))
                draw_show_and_save(output[0].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_MP_{loss.item():.5f}.jpg')
                draw_show_and_save(data['f3'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_MA_{loss.item():.5f}.jpg')
                draw_show_and_save(output[1].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_DP_{loss.item():.5f}.jpg')
                draw_show_and_save(data['f4'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_DA_{loss.item():.5f}.jpg')
                draw_show_and_save(data['f1'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_FGBG_{loss.item():.5f}.jpg')      
                saved = True

            if batch_idx % img_save_idx == 0 and not saved:
                print('Train Epoch: {} Batch_ID: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f} Mask_Loss: {:.5f} Dpth_Loss: {:.5f}'.format(
                       epoch, batch_idx, batch_idx * batch_size, len(train_loader.dataset), 
                       (100. * batch_idx / len(train_loader)),
                       loss, loss1, loss2))
                draw_and_save(output[0].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_MP_{loss.item():.5f}.jpg')
                draw_and_save(data['f3'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_MA_{loss.item():.5f}.jpg')
                draw_and_save(output[1].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_DP_{loss.item():.5f}.jpg')
                draw_and_save(data['f4'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_DA_{loss.item():.5f}.jpg')
                draw_and_save(data['f1'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_FGBG_{loss.item():.5f}.jpg')    
              
            if batch_idx % model_idx == 0:
              torch.save(model.state_dict(),path_model_save)
              print('MODEL SAVED:',path_model_save, 'Epoch & Batch-ID:', epoch, batch_idx)
              
          train_loss /= len(train_loader.dataset)
          mask_train_loss = test_loss1/len(test_loader.dataset)
          dpth_train_loss = test_loss2/len(test_loader.dataset)
          return train_loss, mask_train_loss, dpth_train_loss                    
