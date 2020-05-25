from torchvision import transforms

# # class for Transformations ## 
class Transforms_custom:
      def __init__(self, normalize=False, mean=None, stdev=None):
      
          self.normalize = normalize
          self.mean      = mean      ## Make sure you pass the meand and stdev whenever normalization is set to true 
          self.stdev     = stdev            
      
      def data_transforms(self , height=None, width=None, before_norm=None, after_norm=None):
          transforms_list = [transforms.Resize((height,width))]
          if before_norm:
             transforms_list.extend(before_norm)
           
          transforms_list.append(transforms.ToTensor())
             
          if (self.normalize):
             transforms_list.append(transforms.Normalize(self.mean,self.stdev))
           
          if after_norm:
             transforms_list.extend(after_norm)
           
          return transforms.Compose(transforms_list)
