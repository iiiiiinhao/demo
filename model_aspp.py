import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pylab as plt

# from ext import warp
import warp
#from utils import train_utils as tn_utils

class conv_down(nn.Module):
    """
    Conv3d:三维卷积层, 输入的尺度是(N, C_in,D,H,W)，输出尺度（N,C_out,D_out,H_out,W_out）
    BatchNorm3d:在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
    在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。
    在验证时，训练求得的均值/方差将用于标准化验证数据。
    """

    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(conv_down, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(outChan),
                nn.ReLU(inplace=True)
                )
        self.pool = nn.AvgPool3d(pool_kernel)
#        self.pool = nn.MaxPool3d(pool_kernel)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)
        if self.down:
            x = self.pool(x)
        return x


class PPM(nn.Module):
    def __init__(self, inChan, outChan, size):
        super(PPM, self).__init__()
        interchannel = int(outChan / 4)
        self.size = size
        # print(inChan,outChan,interchannel)
        self.conv1 = nn.Sequential(
            nn.Conv3d(inChan, interchannel*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(interchannel*2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(inChan, interchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(interchannel),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(inChan, interchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(interchannel),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(nn.Conv3d(outChan, outChan, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.AvgPool3d(1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    m.bias.data.zero_()

    def pool(self, x, size):
        avge = nn.AvgPool3d(size)
        return avge(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='trilinear', align_corners=True)

    def forward(self, x):

        out1 = self.pool(x, 4)
        out2 = self.pool(x, 2)
        out3 = self.pool(x, 1)

        out1 = self.conv1(out1)
        out2 = self.conv2(out2)
        out3 = self.conv3(out3)

        out1 = self.upsample(out1,self.size)
        out2 = self.upsample(out2, self.size)
        out3 = self.upsample(out3, self.size)

        out5 = torch.cat([out1, out2, out3 ], dim=1)
        out = self.out(out5)

        return out



# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel, outChan):
        super(ASPP, self).__init__()
        self.mean = nn.AvgPool3d((1, 1, 1)) # (1,1,1)means ouput_dim
        outChan_aspp = int(outChan/4)
        self.conv = nn.Conv3d(in_channel, outChan_aspp,kernel_size=3, stride=1, padding=1)
        self.atrous_block1 = nn.Conv3d(in_channel, outChan_aspp, 1, 1)
        self.atrous_block4 = nn.Conv3d(in_channel, outChan_aspp, 3, 1, padding=4, dilation=4)
        self.atrous_block8 = nn.Conv3d(in_channel, outChan_aspp, 3, 1, padding=8, dilation=8)

    def forward(self, x):

        image_features = self.mean(x)
        image_features = self.conv(image_features)

        atrous_block1 = self.atrous_block1(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block8 = self.atrous_block8(x)

        net = torch.cat([image_features, atrous_block1, atrous_block4,
                                              atrous_block8], dim=1)
        return net


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)



class Net(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):  #num of channel
        super(Net, self).__init__()
        """
        net architecture. 
        :param nfea: list of conv filters. right now it needs to be 1x8.
        """



        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])


        self.same1 = conv_down(nfea[3], nfea[4], down=False)
        self.same2 = conv_down(nfea[4], nfea[5], down=False)
        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)
        self.same5 = conv_down(nfea[7], nfea[8], down=False)
        self.channelAttention1 = SELayer(nfea[1])
        self.channelAttention2 = SELayer(nfea[2])
        self.channelAttention3 = SELayer(nfea[3])
        self.channelAttention4 = SELayer(nfea[4])
        self.channelAttention5 = SELayer(nfea[5])
        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])
        self.channelAttention8 = SELayer(nfea[8])
        self.outconv = nn.Conv3d(
                nfea[8], nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        scale=8

        x = self.down1(x)
        x = self.channelAttention1(x)
        x = self.down2(x)
        x = self.channelAttention2(x)
        x = self.down3(x)
        x = self.channelAttention3(x)
        x = self.same1(x)
        x = self.channelAttention4(x)
        x = self.same2(x)
        x = self.channelAttention5(x)
        x = self.same3(x)
        x = self.channelAttention6(x)
        x = self.same4(x)
        x = self.channelAttention7(x)
        x = self.same5(x)
        x = self.channelAttention8(x)
        x = self.outconv(x)

        x = F.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=True)

        return x

class snet(nn.Module):
    def __init__(self,  img_size=[256,256,96]):
        super(snet, self).__init__()
        self.net = Net()
        self.warper = warp.Warper3d(img_size)


    def forward(self, mov, ref):
        input0 = torch.cat((mov, ref), 1)
        flow = self.net(input0)
        warped = self.warper(mov, flow)
        return warped, flow

#a=snet(ndown=344, img_size=[32,32,32])
#in1 = torch.rand((2,1,32,32,32))
#in2 = torch.rand((2,1,32,32,32))
#b,c=a(in1, in2)