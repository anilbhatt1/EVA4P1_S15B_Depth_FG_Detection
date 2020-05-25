class DownSize(nn.Module):
  def __init__(self,inchannels,outchannels):
    super(DownSize, self).__init__()

    self.conv1  = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False)
    self.conv12 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(outchannels)
    self.relu = nn.ReLU(inplace=False)

    self.conv2  = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv22 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(outchannels)

    self.conv1x1down = nn.Conv2d(inchannels, outchannels,kernel_size=1, stride=2, padding=0, bias=False)

  def forward(self,x):
    identity = self.conv1x1down(x)
    out = self.conv1(x)
    out = self.conv12(out)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.conv22(out)
    out = self.bn2(out)

    out += identity
    out = self.relu(out)

    return out 
	
class UpSize(nn.Module):
  def __init__(self, inchannels, outchannels):
    super(UpSize, self).__init__()

    self.conv1 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv12 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(outchannels)
    self.relu = nn.ReLU(inplace = False)

    self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv22 = nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(outchannels)

    self.conv1x1down = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)

  def forward(self, x):
    identity = self.conv1x1down(x)
    out = self.conv1(identity)
    out = self.conv12(out)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.conv22(out)
    out = self.bn2(out)

    out = self.relu(out)  
    return out

class BGDepth(nn.Module):
  def __init__(self):
    super(BGDepth, self).__init__()


    self.convA = nn.Sequential(nn.Conv2d(in_channels=6,out_channels=32,kernel_size=(3,3),stride=1, 
                                              padding=1, bias= False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
    self.convB = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,
                                              padding=1,bias=False,groups=32),
                                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,1),stride=1,
                                              padding=0,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
    self.convC = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,
                                              padding=1,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
    self.convD = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=1,
                                              padding=1,bias=False))
    self.convE = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(3,3),stride=1,
                                              padding=1,bias=False))    


    self.blck1 = DownSize(64,128)
    self.blck2 = DownSize(128,256)
    self.blck3 = DownSize(256,512)
    self.blck4 = DownSize(512,1024)

    self.blck5 = UpSize(1024, 512)
    self.blck6 = UpSize(512, 256)
    self.blck7 = UpSize(256, 128)
    self.blck8 = UpSize(128, 64)

    self.convLast = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),)

  def forward(self, sample):
    fg_bg = sample['f1']
    bg    = sample['f2']
    f     = torch.cat([fg_bg,bg], dim=1)  # (192,192,6)

    outA = self.convA(f)             #192 (192,192,32)
    outB = self.convB(outA)          #192 (192,192,64)  
    outC = self.convC(outB)          #192 (192,192,128)
    outD = self.convD(outC)          #192 (192,192,64)
    outMask = self.convE(outD)       #192 (192,192,1)
	
    down1 = self.blck1(outD)         #96  (96,96,128)
    down2 = self.blck2(down1)        #48  (48,48,256)
    down3 = self.blck3(down2)        #24  (24,24,512)
    down4 = self.blck4(down3)        #12  (12,12,512) 

    scale1 = nn.functional.interpolate(down4, scale_factor=2, mode='bilinear')  #(24,24,1024)

    up1 = self.blck5(scale1)   #(24,24,512)         
    up1 += down3               #(24,24,512)
    scale2 = nn.functional.interpolate(up1, scale_factor=2, mode='bilinear') #(48,48,512)
    up2 = self.blck6(scale2) #(48,48,256)
    up2 += down2             #(48,48,256)
    scale3 = nn.functional.interpolate(up2, scale_factor=2, mode='bilinear') #(96,96,256)
    up3 = self.blck7(scale3) #(96,96,128)
    up3 += down1             #(96,96,128)
    scale4 = nn.functional.interpolate(up3, scale_factor=2, mode='bilinear')  #(192,192,128)
    up4 = self.blck8(scale4)        #(192,192,64)
    outDepth = self.ConvLast(up4)   #(192,192,1)

    return outMask, outDepth		