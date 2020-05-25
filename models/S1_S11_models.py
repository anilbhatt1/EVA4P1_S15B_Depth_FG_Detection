import torch
import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.05

# Model for trainining CIFAR 10 with depthwise and Dilated Convolution

class CIFAR10Net_Dilation(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 32x32x3 , out = 32x32x32, RF = 3

        self.dilated1B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 32x32x32 , out = 32x32x64, RF = 7
 
        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # in = 32x32x64 , out = 16x16x64, RF = 8
        self.tran1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 16x16x64 , out = 16x16x32, RF = 6        

        # CONVOLUTION BLOCK 2
        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 16x16x32 , out = 16x16x64, RF = 12
        self.depthwise2B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 16
        self.pointwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 16x16x64 , out = 16x16x128, RF = 16   

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # in = 16x16x128 , out = 8x8x128, RF = 18
        self.tran2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 8x8x128 , out = 8x8x32, RF = 18        

        # CONVOLUTION BLOCK 3
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x32 , out = 8x8x64, RF = 26
        
        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2) # in = 8x8x64 , out = 4x4x64, RF = 30
        self.tran3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False)
        ) # in = 4x4x64 , out = 4x4x32, RF = 30
        
        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # in = 4x4x32 , out = 1x1x32, RF = 54	
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 1x1x32 , out = 1x1x10, RF = 54

    def forward(self, x):
        x = self.dilated1B(self.convblock1A(x))
        x = self.tran1(self.pool1(x))
        x = self.pointwise2C(self.depthwise2B(self.convblock2A(x)))
        x = self.tran2(self.pool2(x))
        x = self.convblock3A(x)
        x = self.tran3(self.pool3(x))
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

## ResNet 18 model for training CIFAR10 Using Softmax (softmax - given in model itself)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.Gap1   = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc     = nn.Conv2d(512*block.expansion, num_classes, kernel_size=1, stride = 1, padding=0, bias=False)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.Gap1(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return F.log_softmax(out, dim=-1)
            
def ResNet_18():
    return ResNet(BasicBlock, [2,2,2,2])

# Model for training CIFAR10 with normal 3x3 convolutions. Used for S9 Quiz

class CIFAR10Net_S9(nn.Module):

    def __init__(self):
        super(CIFAR10Net_S9, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 32x32x3 , out = 32x32x32, RF = 3

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 32x32x32 , out = 32x32x64, RF = 5
 
        # TRANSITION BLOCK 1
        self.pool4 = nn.MaxPool2d(2, 2) # in = 32x32x64 , out = 16x16x64, RF = 6

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 16x16x64 , out = 16x16x64, RF = 10
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 14
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 18 

        # TRANSITION BLOCK 2
        self.pool8 = nn.MaxPool2d(2, 2) # in = 16x16x64 , out = 8x8x64, RF = 20
    

        # CONVOLUTION BLOCK 3
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x32, RF = 28
        
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x64, RF = 36 
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x32, RF = 44              
      
        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # in = 8x8x32 , out = 1x1x32, RF = 72	
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 1x1x32 , out = 1x1x10, RF = 72

    def forward(self, x):
        x = self.convblock3(self.convblock2(x))
        x = self.pool4(x)
        x = self.convblock7(self.convblock6(self.convblock5(x)))
        x = self.pool8(x)
        x = self.convblock11(self.convblock10(self.convblock9(x)))
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
## ResNet 18 model for training CIFAR10 with Criterion as CrossEntropy. CrossEntropy criterion called from main file. We will supply o/p of ResNet Model  

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.Gap1   = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc     = nn.Conv2d(512*block.expansion, num_classes, kernel_size=1, stride = 1, padding=0, bias=False)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.Gap1(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out
            
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

## Modified ResNet model for training CIFAR10. "_S11" simply stands for session 11 as this belongs to 11th session of EVA4 program

class BasicBlock_S11(nn.Module):
    expansion  = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_S11, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += F.relu(self.shortcut(x))
        return out
    
class ResNet_mod_S11(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_mod_S11, self).__init__()
        self.in_planes  = [128,512]
        self.layer_call = 0

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.conv2  = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.bn2    = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.conv3  = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.bn3    = nn.BatchNorm2d(256)
        self.conv4  = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.bn4    = nn.BatchNorm2d(512)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=1)
        self.pool4  = nn.MaxPool2d(4, 4)
        self.fc     = nn.Conv2d(512*block.expansion, num_classes, kernel_size=1, stride = 1, padding=0, bias=False)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes[self.layer_call], planes, stride))
            self.in_planes[self.layer_call] = planes * block.expansion
        self.layer_call += 1    
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.pool1(self.conv2(out))))
        out = self.layer1(out)
        out = F.relu(self.bn3(self.pool2(self.conv3(out))))
        out = F.relu(self.bn4(self.pool3(self.conv4(out))))
        out = self.layer2(out)
        out = self.pool4(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return F.log_softmax(out, dim=-1)
            
def ResNet_S11():
    return ResNet_mod_S11(BasicBlock_S11, [1,1])
