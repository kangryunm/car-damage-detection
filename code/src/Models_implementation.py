import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
import torchvision
import torch.nn.functional as F

# Code reference
# https://amaarora.github.io/2020/09/13/unet.html

class Unet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(256, 256)): # out_sz = (572, 572)
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, kernel_size=1)  ## 1x1 convolution (64, h, w) > (1, h, w)
        self.retain_dim  = retain_dim
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:]) ## encoder 마지막 feature map / 마지막 단계 이전 feature map들
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)

        # out      = self.sigmoid(out)  ## activation 추가 (0 < out < 1 사이의 확률 분포)
        ## hist 구하는 과정에서 np.bincount negative input이 들어가면 안됨
        return out


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) ## padding=1
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1) ## padding=1
        
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))  ## input output shape 유지, add ReKU


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []  # pooling 전의 feature map들을 모두 저장 (Decoder에서 concat할 때 사용)
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        self.bn         = nn.BatchNorm2d([chs[i+1] for i in range(len(chs)-1)])
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
            x        = self.bn(x) # batch norm 추가
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


'''
class Unet(nn.Module):
    # def __init__(self, num_classes,encoder,pre_weight):
    def __init__(self, encoder_name, encoder_weights, in_channels, out_channels, activation):
        super().__init__()
        # self.model = smp.Unet( classes = num_classes,
        #                       encoder_name=encoder,
        #                       encoder_weights=pre_weight,
        #                       in_channels=3)
        
        self.model = smp.Unet(
                                encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                in_channels=in_channels,
                                classes=out_channels, # out_classes
                                activation=activation
                              )


    def forward(self, x):
        y = self.model(x)
        # encoder_weights = "imagenet"
        return y  # output shape: (batch, classes, h, w) = 
'''



