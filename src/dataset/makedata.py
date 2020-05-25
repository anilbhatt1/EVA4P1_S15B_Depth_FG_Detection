import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class Makedata(Dataset):
  def __init__(self,image_path,content_list, start_rec, end_rec, transform1, transform2, transform3, transform4, do_transform=True):
    self.interim      = list(image_path.glob('*.jpg'))
    self.start_rec    = start_rec
    self.end_rec      = end_rec
    self.f1_files     = self.interim[start_rec:end_rec]
    #self.f1_files     = list(f1.glob('*.jpg'))
    self.content_list = content_list
    self.fg_bg_transform   = transform1 
    self.bg_transform      = transform2
    self.mask_transform    = transform3
    self.depth_transform   = transform4
    self.do_transform      = do_transform

  def __len__(self):
    return len(self.f1_files)  

  def __getitem__(self,index):
    fg_bg_name   = self.f1_files[index].stem
    fg_bg_idx    = fg_bg_name.split('_')[-1]
    bg_name      = '/content/BG_and_Its_Flip/' + self.content_list[int(fg_bg_idx)-1].split(',')[1]
    mask_name    = '/content/FG_BG_Mask_400K/' + 'Img_fg_bg_mask' + str(fg_bg_idx) + '.jpg'
    depth_name   = '/content/FG_BG_Depth_0_400K/' + 'Img_fg_bg_' + str(fg_bg_idx) + '_depth.jpg'
    f1_image = Image.open(self.f1_files[index])  
    f2_image = Image.open(f'{bg_name}')
    f3_image = Image.open(f'{mask_name}')
    f4_image = Image.open(f'{depth_name}')

    if self.do_transform:
      f1_image = self.fg_bg_transform(f1_image)
      f2_image = self.bg_transform(f2_image)
      f3_image = self.mask_transform(f3_image)
      f4_image = self.depth_transform(f4_image)

    return {'f1':f1_image, 'f2':f2_image, 'f3':f3_image, 'f4':f4_image}
