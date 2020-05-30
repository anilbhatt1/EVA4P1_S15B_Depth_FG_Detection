from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from kornia.losses import SSIM
from kornia.losses import DiceLoss

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Training_loss1: 
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
          train_loss1, train_loss2, train_loss, train_mask_iou_cum, train_depth_iou_cum = 0, 0, 0, 0, 0
          pbar = tqdm(train_loader)
          num_batches = len(train_loader.dataset)/batch_size
          cuda0 = torch.device('cuda:0')
          log_path  = path_name + 'train_log.txt'
          log_file  = open(f'{log_path}', "a")          

          for batch_idx, data in enumerate(pbar):
            data['f1'] = data['f1'].to(cuda0)
            data['f2'] = data['f2'].to(cuda0)
            data['f3'] = data['f3'].to(cuda0)
            data['f4'] = data['f4'].to(cuda0)
            #data['f3O'] = torch.tensor(data['f3'],dtype= torch.int64, device= cuda0)

            optimizer.zero_grad()
            output = model(data)

            loss1 = criterion1(output[0], data['f3'])
            loss2 = criterion2(output[1], data['f4'])
            loss  = 2*loss1 + loss2
            train_loss1 += loss1
            train_loss2 += loss2
            train_loss  += loss
            mask_iou   = self.calculate_iou(data['f3'].detach().cpu().numpy(), output[0].detach().cpu().numpy())
            depth_iou  = self.calculate_iou(data['f4'].detach().cpu().numpy(),  output[1].detach().cpu().numpy())
            train_mask_iou_cum  += mask_iou
            train_depth_iou_cum += depth_iou

            pbar.set_description(desc = f'TR{int(epoch)}|{int(batch_idx)}|{loss1:.3f}|{loss2:.3f}|{mask_iou:.3f}|{depth_iou:.3f}') 
                                          
            loss.backward()
            optimizer.step()
            
            if batch_idx % img_save_idx == 0 or batch_idx == int(num_batches-1):
                print('Train Epoch:{} Batch_ID: {} [{}/{} ({:.0f}%)]\tLoss:{:.5f} Mask_Loss:{:.5f} Dpth_Loss:{:.5f} Mask_IOU:{:.5f} Dpth_IOU: {:.5F}'
                      .format(epoch, batch_idx, batch_idx * batch_size, len(train_loader.dataset), (100. * batch_idx / len(train_loader)),
                       loss, loss1, loss2, mask_iou, depth_iou))
                flg = self.draw_and_save(output[0].detach().cpu(),  f'{path_name}{epoch}_{batch_idx}_MP_{loss.item():.5f}.jpg')
                flg = self.draw_and_save(data['f3'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_MA_{loss.item():.5f}.jpg')
                flg = self.draw_and_save(output[1].detach().cpu(),  f'{path_name}{epoch}_{batch_idx}_DP_{loss.item():.5f}.jpg')
                flg = self.draw_and_save(data['f4'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_DA_{loss.item():.5f}.jpg')
                flg = self.draw_and_save(data['f1'].detach().cpu(), f'{path_name}{epoch}_{batch_idx}_FGBG_{loss.item():.5f}.jpg')
                string = f' Train Epoch-{int(epoch)}|Batch-{int(batch_idx)}|Loss-{loss:.5f}|MaskLoss-{loss1:.5f}|DepthLoss-{loss2:.5f}|MaskIOU-{mask_iou:.5f}|DepthIOU-{depth_iou:.5f}'
                wrt = self.log_write(string, log_file)                
              
            if batch_idx % model_save_idx == 0:
              torch.save(model.state_dict(),path_model_save)
              print('MODEL SAVED:',path_model_save, 'Epoch & Batch-ID:', epoch, batch_idx)
              
          #train_loss       /= len(train_loader.dataset)
          train_loss       /= num_batches
          train_mask_loss   = train_loss1/num_batches
          train_depth_loss  = train_loss2/num_batches
          train_mask_iou    = train_mask_iou_cum/num_batches
          train_depth_iou   = train_depth_iou_cum/num_batches
          string = f'*Train Epoch-{int(epoch)}|Batch-{int(batch_idx)}|Loss-{train_loss:.5f}|MaskLoss-{train_mask_loss:.5f}|DepthLoss-{train_depth_loss:.5f}|MaskIOU-{train_mask_iou:.5f}|DepthIOU-{train_depth_iou:.5f}'
          wrt    = self.log_write(string, log_file)
          log_file.close()          
          return train_loss, train_mask_loss, train_depth_loss, train_mask_iou, train_depth_iou   

    def calculate_iou(self, target, prediction, thresh=0.5):
        '''
        Calculate intersection over union value
        :param target: ground truth
        :param prediction: output predicted by model
        :param thresh: threshold
        :return: iou value
        '''
        intersection = np.logical_and(np.greater(target, thresh), np.greater(prediction, thresh))
        union = np.logical_or(np.greater(target, thresh), np.greater(prediction, thresh))
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
        
    def draw_and_save(self, tensors, name, figsize=(15,15), *args, **kwargs):
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
          flag = True
         #plt.show()
          return flag    

    def log_write(self, string, log_file):
          wrt = False
          write_str = string + '\n'
          log_file.write(write_str)
          wrt = True
          return wrt      