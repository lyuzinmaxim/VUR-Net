import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True #работает медленнее, но зато воспроизводимость!

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='replicate')
  
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)
  
class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock1, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(in_channels,out_channels)
        
        
        
    def forward(self, x):
        
        branch = x
        branch = self.conv3(branch)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out +=branch
        out = self.relu(out)
        
        return out
 
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock2, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(in_channels,out_channels)
        
        
        
    def forward(self, x):
        
        branch = x
        branch = self.conv3(branch)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out +=branch

        out = self.relu(out)
       
        return out
     
def down_creator(in_channels, out_channels):
    return nn.Sequential( 
       nn.MaxPool2d(kernel_size=2, stride=2),
       ResidualBlock2(in_channels,out_channels)
                        )
  
def conv_up(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2, 
                    stride=2)

class VURnet(torch.nn.Module):
  
  def __init__(self):
    super(VURnet,self).__init__()

    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.block1 = ResidualBlock1(in_channels=1,out_channels=64)
    self.block2 = ResidualBlock1(in_channels=64,out_channels=128)

    self.down1 = down_creator(128,256)
    self.down2 = down_creator(256,512)
    self.down3 = down_creator(512,512)

    self.up1 = conv_up(512,512)
    self.upblock1 = ResidualBlock2(in_channels=1024,out_channels=512)
    
    self.up2 = conv_up(512,256)
    self.upblock2 = ResidualBlock2(in_channels=512,out_channels=256)

    self.up3 = conv_up(256,128)
    self.upblock3 = ResidualBlock1(in_channels=256,out_channels=128)

    self.up4 = conv_up(128,64)
    self.upblock4 = ResidualBlock1(in_channels=128,out_channels=1)
    self.conv = conv3x3(in_channels=1,out_channels=1)

  def forward(self,image):

    #encoder
    x1 = self.block1(image) #
    
    x2 = self.max_pool_2x2(x1)
    x3 = self.block2(x2)  #
    
    x4 = self.down1(x3)   #
    
    x5 = self.down2(x4)   #
 
    x6 = self.down3(x5)   #
    print(x6.size(),'мой вывод')
    x7 = self.down3(x6)
    
    #decoder
    y1 = self.up1(x7)
    y1 = torch.cat([y1,x6],1)
    y1 = self.upblock1(y1)

    y2 = self.up1(y1)
    y2 = torch.cat([y2,x5],1)
    y2 = self.upblock1(y2)

    y3 = self.up2(y2)
    y3 = torch.cat([y3,x4],1)
    y3 = self.upblock2(y3)

    y4 = self.up3(y3)
    y4 = torch.cat([y4,x3],1)
    y4 = self.upblock3(y4)

    y5 = self.up4(y4)
    y5 = torch.cat([y5,x1],1)
    y5 = self.upblock4(y5)
    out = self.conv(y5)

    #print(out.size(),'мой вывод')
    return out  

if __name__ == "__main__":
  image = torch.rand((1,1,256,256))
  model = VURnet()
  print(model(image).size())
