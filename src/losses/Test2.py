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

fg_bg_mean, fg_bg_stdev                    = [0.56670278, 0.49779153, 0.43632878], [0.25049532, 0.2468085, 0.25520498]
mask_mean,  mask_stdev                     = [0.20249742], [0.39961225]
depth_mean, depth_stdev                    = [0.32939295], [0.24930712]
bg_mean, bg_stdev                          = [0.58245822, 0.51269352, 0.43691653], [0.24252189, 0.24318804, 0.25401604]
# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Testing_loss1:

      def test_loss_calc(self,model, device, test_loader, optimizer, epoch, criterion1, criterion2, batch_size, path_name, scheduler=None, img_save_idx =500):
          self.model        = model
          self.device       = device
          self.test_loader  = test_loader
          self.optimizer    = optimizer
          self.epoch        = epoch
          self.criterion1   = criterion1
          self.criterion1   = criterion2           
          self.scheduler    = scheduler
          self.batch_size   = batch_size
          self.path_name    = path_name
          self.img_save_idx = img_save_idx

          model.eval()  
          test_loss1, test_loss2, test_loss, test_mask_iou_cum, test_depth_iou_cum = 0, 0, 0, 0, 0
          pbar = tqdm(test_loader)
          num_batches = len(test_loader.dataset)/batch_size
          cuda0 = torch.device('cuda:0')
          log_path  = path_name + 'test_log.txt'
          log_file  = open(f'{log_path}', "a")

          with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
              data['f1'] = data['f1'].to(cuda0)
              data['f2'] = data['f2'].to(cuda0)
              data['f3'] = data['f3'].to(cuda0)
              data['f4'] = data['f4'].to(cuda0)      
              #data['f3O'] = torch.tensor(data['f3'],dtype= torch.int64, device= cuda0)      
            
              output = model(data)

              loss1 = criterion1(output[0], data['f3'])
              loss2 = criterion2(output[1], data['f4'])
              loss  = 2*loss1 + loss2
              test_loss1 += loss1
              test_loss2 += loss2
              test_loss  += loss
              mask_iou   = self.calculate_iou(data['f3'].detach().cpu().numpy(), output[0].detach().cpu().numpy())
              depth_iou  = self.calculate_iou(data['f4'].detach().cpu().numpy(),  output[1].detach().cpu().numpy())
              test_mask_iou_cum  += mask_iou
              test_depth_iou_cum += depth_iou

              pbar.set_description(desc = f'TS{int(epoch)}|{int(batch_idx)}|{loss1:.3f}|{loss2:.3f}|{mask_iou:.3f}|{depth_iou:.3f}')   
              
              if batch_idx % img_save_idx == 0 or batch_idx == int(num_batches-1):
                  print('Test Epoch: {} [{}/{} ({:.0f}%)]\tTest_Loss: {:.6f} Mask_Loss: {:.5f} Dpth_Loss: {:.5f} Mask_IOU: {:.5f} Dpth_IOU: {:.5F}'
                         .format(epoch, batch_idx * len(data), len(test_loader.dataset), (100. * batch_idx / len(test_loader)),
                                 loss.item(), loss1.item(), loss2.item(),mask_iou, depth_iou ))
                  flg = self.draw_and_save(output[0].detach().cpu(),mask_mean, mask_stdev, f'{path_name}Test_{epoch}_{batch_idx}_MP_{loss.item():.5f}.jpg')
                  flg = self.draw_and_save(data['f3'].detach().cpu(),mask_mean, mask_stdev, f'{path_name}Test_{epoch}_{batch_idx}_MA_{loss.item():.5f}.jpg')
                  flg = self.draw_and_save(output[1].detach().cpu(),depth_mean, depth_stdev, f'{path_name}Test_{epoch}_{batch_idx}_DP_{loss.item():.5f}.jpg')
                  flg = self.draw_and_save(data['f4'].detach().cpu(),depth_mean, depth_stdev, f'{path_name}Test_{epoch}_{batch_idx}_DA_{loss.item():.5f}.jpg')
                  flg = self.draw_and_save(data['f1'].detach().cpu(), fg_bg_mean, fg_bg_stdev, f'{path_name}Test_{epoch}_{batch_idx}_FGBG_{loss.item():.5f}.jpg')
                  string = f' Test Epoch-{int(epoch)}|Batch-{int(batch_idx)}|Loss-{loss:.5f}|MaskLoss-{loss1:.5f}|DepthLoss-{loss2:.5f}|MaskIOU-{mask_iou:.5f}|DepthIOU-{depth_iou:.5f}'
                  wrt = self.log_write(string, log_file)
            
          #test_loss      /= len(test_loader.dataset)
          test_loss      /= num_batches
          test_mask_loss  = test_loss1/num_batches
          test_depth_loss = test_loss2/num_batches
          test_mask_iou   = test_mask_iou_cum/num_batches
          test_depth_iou  = test_depth_iou_cum/num_batches
          string = f'*Test Epoch-{int(epoch)}|Batch-{int(batch_idx)}|Loss-{test_loss:.5f}|MaskLoss-{test_mask_loss:.5f}|DepthLoss-{test_depth_loss:.5f}|MaskIOU-{test_mask_iou:.5f}|DepthIOU-{test_depth_iou:.5f}'
          wrt    = self.log_write(string, log_file)
          log_file.close()
          return test_loss, test_mask_loss, test_depth_loss, test_mask_iou, test_depth_iou

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
        
      def draw_and_save(self, tensors, mean, stdev, name, figsize=(15,15), *args, **kwargs):
          try:
            tensors = tensors.detach().cpu()
          except:
            pass
            
          tensor_mean   = mean
          tensor_stdev  = stdev
          for i in range(tensors.shape[0]):
              if tensors.shape[1] ==3:
                 for j in range(0,3):  
                     tensors[i,j] = ((tensors[i,j] * tensor_stdev[j]) + tensor_mean[j])
              if tensors.shape[1] ==1:
                 for j in range(0,1):  
                     tensors[i,j] = ((tensors[i,j] * tensor_stdev[j]) + tensor_mean[j])            
            
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