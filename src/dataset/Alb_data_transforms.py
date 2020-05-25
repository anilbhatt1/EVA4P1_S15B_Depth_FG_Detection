from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


class Albumentations_transform:
  """
  Class to create test and train transforms using Albumentations. ToTensor() will be appended at the end of transforms list
  """
  def __init__(self, transforms_list=[]):
    transforms_list.append(AP.ToTensor())
    
    self.transforms = A.Compose(transforms_list)

  def __call__(self):
    return self.transforms
