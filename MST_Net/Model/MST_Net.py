import torch
import torch.nn as nn
from einops import rearrange
from torchinfo import summary
from ptflops import get_model_complexity_info

class WinEmbed(nn.Module):
    def __init__(self, map_size, win_size, ch_in, ch_out):
        super().__init__()
        self.img_size = map_size
        self.win_size = win_size
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=win_size, stride=win_size)

    def forward(self, x):
        x = self.proj(x)
        return x

class Attention(nn.Module):
    def __init__(self,channel,head=4):
        super(Attention, self).__init__()
        self.qkv = nn.Conv2d(channel,3*channel,1,padding=0,bias=True)
        self.head = head
        self.proj = nn.Conv2d(channel,channel,1)
    def forward(self,x):
        b, c, h, w = x.shape
        x_qkv = self.qkv(x)
        q,k,v = x_qkv.chunk(3,dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)

        attn = (q.transpose(-2, -1) @ k)
        attn = attn.softmax(dim=-1)
        out = (v @ attn)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.head, h=h, w=w)
        out =self.proj(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super(LayerNorm, self).__init__()
        self.proj = nn.LayerNorm(dim)

    def forward(self,x):
        b,c,h,w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj(x)
        x = rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
        return x


class STB(nn.Module):
    def __init__(self,map_size,win_size,ch_in,ch_out,head = 4):
        super(STB, self).__init__()
        self.PE = WinEmbed(map_size, win_size, ch_in, ch_out)
        self.layer_norm = LayerNorm(ch_out)
        self.att = Attention(ch_out,head)

    def forward(self,x):
        x_emd = self.PE(x)
        x = self.layer_norm(x_emd)
        x = self.att(x)
        return x+x_emd

class Adaptive_Block(nn.Module):
    def __init__(self):
        super(Adaptive_Block, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(4)])

    def forward(self, *inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, but got {len(inputs)}.")

        weighted_inputs = [w * x for w, x in zip(self.weights, inputs)]
        output = torch.cat(weighted_inputs, dim=1)
        return output

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    # nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x

class outconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),)
    def forward(self, x):
        x = self.conv(x)
        return x


class MST_Net(nn.Module):
    def __init__(self, img_size = 256,img_ch=1, output_ch=1,filter_dim=32):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filter_dim)
        self.Conv2 = conv_block(ch_in=filter_dim, ch_out=filter_dim*2)
        self.Conv3 = conv_block(ch_in=filter_dim*2, ch_out=filter_dim*4)
        self.Conv4 = conv_block(ch_in=filter_dim*4, ch_out=filter_dim*8)
        self.Conv5 = conv_block(ch_in=filter_dim*8, ch_out=filter_dim*16)

        map_size = [int(img_size / 2 ** i) for i in range(1, 5)]
        win_size = [int(x / map_size[-1]) for x in map_size]
        self.atten2 = STB(map_size[0], win_size[0], ch_in=filter_dim * 2, ch_out=filter_dim * 2)
        self.atten3 = STB(map_size[1], win_size[1], ch_in=filter_dim * 4, ch_out=filter_dim * 2)
        self.atten4 = STB(map_size[2], win_size[2], ch_in=filter_dim * 8, ch_out=filter_dim * 4)
        self.atten5 = STB(map_size[3], win_size[3], ch_in=filter_dim * 16, ch_out=filter_dim * 8)

        self.Up5 = up_conv(ch_in=filter_dim*16, ch_out=filter_dim*8)
        self.Up_conv5 = conv_block(ch_in=filter_dim*16, ch_out=filter_dim*8)
        self.Up4 = up_conv(ch_in=filter_dim*8, ch_out=filter_dim*4)
        self.Up_conv4 = conv_block(ch_in=filter_dim*8, ch_out=filter_dim*4)
        self.Up3 = up_conv(ch_in=filter_dim*4, ch_out=filter_dim*2)
        self.Up_conv3 = conv_block(ch_in=filter_dim*4, ch_out=filter_dim*2)
        self.Up2 = up_conv(ch_in=filter_dim*2, ch_out=filter_dim)
        self.Up_conv2 = conv_block(ch_in=filter_dim*2, ch_out=filter_dim)
        self.Conv11 = outconv(filter_dim, output_ch)

    def forward(self, x):
        # encoder
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        #Attention
        x2_ = self.atten2(x2)
        x3_ = self.atten3(x3)
        x4_ = self.atten4(x4)
        x5_ = self.atten5(x5)
        x5 = torch.cat((x2_, x3_, x4_, x5_), 1)

        # decoder
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        T2 = self.Conv11(d2)

        return T2

if __name__ =="__main__":

    model = MST_Net()



